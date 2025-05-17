import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
import numpy as np
import data.util as util
from models.modules.module_init import weight_xavier_init, weight_orthogonal_init, weight_init
import torchvision.utils as vutils
import cv2 as cv


logger = logging.getLogger('base')

class IRNColorModel(BaseModel):
    def __init__(self, opt):
        super(IRNColorModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        opt_net = opt['network_grey']
        which_model = opt_net['which_model']
        use_robust = which_model['use_robust']
        Gau_channel_scale = which_model['Gau_channel_scale']
        Model_test = which_model['Model_test']
        init_opt = opt_net['init']

        self.train_opt = train_opt
        self.test_opt = test_opt
        self.use_robust = use_robust
        self.Gau_channel_scale = Gau_channel_scale
        self.init_opt = init_opt
        self.Model_test = Model_test

        self.netG = networks.define_grey(opt).to(self.device)
        self.Robust_Net = networks.define_robust(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.Robust_Net = DistributedDataParallel(self.Robust_Net, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.Robust_Net = DataParallel(self.Robust_Net)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()


        if self.is_train:
            self.Robust_Net.train()

            # loss
            # self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # feature loss
            if train_opt['feature_weight'] > 0:
                self.Reconstructionf = ReconstructionLoss(losstype=self.train_opt['feature_criterion'])

                self.l_fea_w = train_opt['feature_weight']
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)
            else:
                self.l_fea_w = 0

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []

                # optim_params.append(v)
                # if "net" in k:
                #     v.requires_grad = False
                #     optim_params.append(v)
                # else:
                #     if self.rank <= 0:
                #         logger.warning('Params [{:s}] will not optimize.'.format(k))

            for k, v in self.Robust_Net.named_parameters():
                # v.requires_grad = True
                optim_params.append(v)

            for k, v in self.netG.named_parameters():
                # v.requires_grad = True
                optim_params.append(v)

            self.optimizer_G = torch.optim.AdamW(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            self.netG.eval()
            for k, v in self.netG.named_parameters():
                v.requires_grad = False

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.LQ = data['LQ'].to(self.device)
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    # def loss_forward(self, out, y, z):
    #
    #     l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
    #
    #     z = z.reshape([out.shape[0], -1])
    #     l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]
    #
    #     return l_forw_fit, l_forw_ce
    def loss_backward(self, xIn, y, LQ_ref):

        x_samples = self.netG(x=y, rev=True)
        x_samples = x_samples[:, :3, :, :]
        x_samples = torch.clip(x_samples, 0, 1.0)
        EN_out = self.Robust_Net(LQ_ref, x_samples)

        if 0:
            for i in range(3):
                images_to_save = x_samples[0:1, i:(i + 1), :, :]
                print(" images_to_save shape", images_to_save.shape)
                # print(" x_samples: \n\n", images_to_save)
                # nrow 参数决定了每行有多少张图片，由于我们想要单独保存，所以设置为1
                vutils.save_image(images_to_save, 'x_samples{}.png'.format(i), nrow=1, normalize=True,
                                  scale_each=True)
            for i in range(3):
                images_to_save = LQ_syn[0:1, i:(i + 1), :, :]
                # print(" LQ_syn: \n\n", images_to_save)
                print(" images_to_save shape", images_to_save.shape)
                # nrow 参数决定了每行有多少张图片，由于我们想要单独保存，所以设置为1
                vutils.save_image(images_to_save, 'LQ_syn{}.png'.format(i), nrow=1, normalize=True,
                                  scale_each=True)

            # assert  1 < 0

        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(xIn, EN_out)
        # l_back_syn = self.train_opt['lambda_en_out'] * self.Reconstruction_back(EN_out, x)

        # feature loss
        if self.l_fea_w > 0:
            l_back_fea = self.train_opt['feature_weight'] * self.feature_loss(xIn, EN_out)
        else:
            l_back_fea = torch.tensor(0)

        return l_back_rec, l_back_fea

    def feature_loss(self, real, fake):
        real_fea = self.netF(real).detach()
        fake_fea = self.netF(fake)
        l_g_fea = self.l_fea_w * self.Reconstructionf(real_fea, fake_fea)

        return l_g_fea

    def save_image(self, tensor, filename):
        # 将张量从Tensor转换为numpy数组
        # 首先确保CPU上进行操作
        tensor = tensor.detach().cpu()

        # 转换为[0,255]
        tensor = tensor.mul(255).byte()

        # 转换为numpy
        numpy_image = tensor.numpy()

        # 调整维度顺序为 [height, width, channels]
        numpy_image = numpy_image.transpose(1, 2, 0)

        # 使用cv2保存图像
        cv.imwrite(filename, cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR))

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward decolorization
        input_real = self.real_H
        # self.output = self.netG(x=self.input)



        zshape = self.LQ.shape
        LQ_ref = self.LQ #.detach()

        # l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], Grey_ref, self.output[:, 3:, :, :])
        #
        # # backward upscaling
        # out_LQ = self.Quantization(self.output[:, :3, :, :])

        if self.train_opt['add_noise_on_y']:
            probability = self.train_opt['y_noise_prob']
            noise_scale = self.train_opt['y_noise_scale']
            prob = np.random.rand()
            if prob < probability:
                # print("LQ_ref original", LQ_ref)
                LQ_ref = LQ_ref + noise_scale * self.gaussian_batch(LQ_ref.shape)
                # print("add noisy LQ_ref original", LQ_ref)

        # LQ_float = self.Robust_Net(LQ_ref)  ## 将整数 通过一个鲁棒模型，将整数转换成浮点数
        LQ_ref = self.Quantization(LQ_ref[:, :3, :, :])
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = torch.cat((LQ_ref, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        l_back_rec, l_back_fea = self.loss_backward(xIn=input_real, y=y_, LQ_ref=LQ_ref)

        # print("l_forw_fit", l_forw_fit.item())
        # print("l_forw_ce", l_forw_ce.item())
        # print("l_back_rec", l_back_rec.item())
        # total loss
        loss = l_back_rec + l_back_fea
        # loss = l_back_rec + l_forw_ce
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.Robust_Net.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        # self.log_dict['l_forw_fit'] = l_forw_fit.item()
        # self.log_dict['l_forw_ce'] = l_forw_ce.item()
        # self.log_dict['l_back_syn'] = l_back_syn.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_back_fea'] = l_back_fea.item()

    def test(self):

        n, _, hh, ww = self.LQ.size()

        zshape = self.LQ.shape
        input_LQ = self.LQ
        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.Robust_Net.eval()
        with torch.no_grad():
            # if not self.Model_test:
                # self.forw_L = self.netG(x=self.input)[:, :3, :, :]
                # self.forw_L = self.Quantization(self.forw_L)
            if 0: ### mlkk
                Path_model = "./Datasets/test/0.0032/ELIC_arch_TEST_LQ_GEN/I_23P0059.png"
                # Path_model = "./test_images/LR_images/rec_image.png"
                # Path_model = "./test_images/LR_images/PSNR_39.64_crop_14_44.bmp"

                LR_img = util.read_img(None, Path_model, None)
                if LR_img.shape[2] == 3:
                    LR_img = LR_img[:, :, [2, 1, 0]]

                LR_img = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_img, (2, 0, 1)))).float()
                print("LR_img1: ", LR_img.shape)
                LR_img = torch.unsqueeze(LR_img, dim=0).cuda()
                print("LR_img2: ", LR_img.shape)
                ##  replace the self.forw_L
                self.forw_L = LR_img

                # if self.Model_test:
                # self.forw_L = self.LQ  ##  屏蔽掉产生模型， 直接送入测试图像


            y_forw = torch.cat((input_LQ, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

            # self.fake_H = torch.clip(self.fake_H, 0, 1.0)
            # self.save_image(self.fake_H[0, :, :, :], "rec.png")
            # self.save_image(input_LQ[0, :, :, :], "LQ_ref.png")
            # self.save_image(self.real_H[0, :, :, :], "GT.png")
            # assert 1 < 0

            self.EN_fake_H = self.Robust_Net(input_LQ, self.fake_H)

        self.Robust_Net.train()

    # def decolorize(self, img):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         Grey_img = self.netG(x=img)#[:, :1, :, :]
    #         Grey_img = self.Quantization(Grey_img)
    #     self.netG.train()
    #
    #     return Grey_img
    #
    # def colorize(self, Grey_img, gaussian_scale=1):
    #     Lshape = Grey_img.shape
    #     zshape = Lshape # [Lshape[0], 2, Lshape[2], Lshape[3]]
    #     y_ = torch.cat((Grey_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
    #
    #     self.netG.eval()
    #     with torch.no_grad():
    #         if self.use_robust:
    #             img, img_EN = self.netG(x=y_, x_LQ=self.LQ, rev=True)
    #             img_EN = img_EN[:,:3,:,:]
    #             img = img[:,:3,:,:]
    #             self.netG.train()
    #             return img, img_EN
    #         else:
    #             img = self.netG(x=y_, x_LQ=self.LQ, rev=True)[:, :3, :, :]
    #             self.netG.train()
    #         return img



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ_ref'] = self.LQ.detach()[0].float().cpu()
        out_dict['HQ_rec'] = self.fake_H.detach()[0].float().cpu()
        out_dict['HQ_rec_EN'] = self.EN_fake_H.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        s, n = self.get_network_description(self.Robust_Net)
        if isinstance(self.Robust_Net, nn.DataParallel) or isinstance(self.Robust_Net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.Robust_Net.__class__.__name__,
                                             self.Robust_Net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.Robust_Net.__class__.__name__)

        if self.rank <= 0:
            logger.info('Network Robust_Net structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_Robust = self.opt['path']['pretrain_model_R']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        else:
            print("init with -- netG -- >", self.init_opt)
            if self.init_opt == 'weight_init':
                weight_init(self.netG)
            elif self.init_opt == 'weight_orthogonal_init':
                weight_orthogonal_init(self.netG)
            else:
                weight_xavier_init(self.netG)

        if load_path_Robust is not None:
            logger.info('Loading model for Robust [{:s}] ...'.format(load_path_Robust))
            self.load_network(load_path_Robust, self.Robust_Net, self.opt['path']['strict_load'])
        else:
            print("init with -- Robust_Net -- >", self.init_opt)
            if self.init_opt == 'weight_init':
                weight_init(self.Robust_Net)
            elif self.init_opt == 'weight_orthogonal_init':
                weight_orthogonal_init(self.Robust_Net)
            else:
                weight_xavier_init(self.Robust_Net)


    def save(self, iter_label):
        self.save_network(self.Robust_Net, 'R', iter_label)
