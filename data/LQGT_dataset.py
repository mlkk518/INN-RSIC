import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

        self.use_grey = False
        if self.opt['use_grey']:
            self.use_grey = True

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        if not self.use_grey:
            self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                    meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        # scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        LQ_path = self.paths_LQ[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        # modcrop in the validation / test phase
        # if self.opt['phase'] != 'train':
        #     img_GT = util.modcrop(img_GT, scale)
        #     img_LQ = util.modcrop(img_LQ, scale)
        # change color space if necessary
        # if self.opt['color']:
        #     img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
        #     img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        # get LQ image
        if not self.use_grey:
            if self.paths_LQ:
                LQ_path = self.paths_LQ[index]
                if self.data_type == 'lmdb':
                    resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
                else:
                    resolution = None
                img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
            else:  # down-sampling on-the-fly
                # randomly scale during training
                if self.opt['phase'] == 'train':
                    random_scale = random.choice(self.random_scale_list)
                    H_s, W_s, _ = img_GT.shape

                    def _mod(n, random_scale, scale, thres):
                        rlt = int(n * random_scale)
                        rlt = (rlt // scale) * scale
                        return thres if rlt < thres else rlt

                    H_s = _mod(H_s, random_scale, scale, GT_size)
                    W_s = _mod(W_s, random_scale, scale, GT_size)
                    img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                    # force to 3 channels
                    if img_GT.ndim == 2:
                        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

                H, W, _ = img_GT.shape
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                img_LQ = cv2.resize(np.copy(img_LQ), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                if not self.use_grey:
                    # img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                    if img_LQ.ndim == 2:
                        img_LQ = np.expand_dims(img_LQ, axis=2)
            # print(" mlkkk  use_grey  dataset ..... \n", self.use_grey)
            # if not self.use_grey:
            H, W, C = img_LQ.shape
            # LQ_size = GT_size // scale
            LQ_size = GT_size
            # print(" img_Q 111 size", img_LQ.shape)
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h ), int(rnd_w )
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            # print(" img_Q 222 size", img_LQ.shape)
            # else:
            #     rnd_h_GT = random.randint(0, max(0, H - GT_size))
            #     rnd_w_GT = random.randint(0, max(0, W - GT_size))
            #     img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            # if not self.use_grey:
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                              self.opt['use_rot'])
            # else:
            #     img_GT = util.augment([img_GT], self.opt['use_flip'], self.opt['use_rot'])[0]

        # change color space if necessary
        # if not self.use_grey:
        if self.opt['color']:
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition
            # print(" img_Q size", img_LQ.shape)
        # if self.use_grey:
        #     img_Grey = cv2.cvtColor(img_GT, cv2.COLOR_BGR2GRAY)
        #     print(" img_Grey size", img_Grey.shape)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            # if not self.use_grey:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # if not self.use_grey:
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        # if self.use_grey:
        #     img_Grey = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img_Grey, 0))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path

        # print("img_LQ shape", img_LQ.shape)
        # print("img_GT shape", img_GT.shape)
        # if not self.use_grey:
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}
        # else:
        #     return {'Grey': img_Grey, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
