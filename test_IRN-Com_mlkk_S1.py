import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch



#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)


for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['LQ_psnr'] = []
    test_results['EN_psnr'] = []
    test_results['LQ_ssim'] = []
    test_results['EN_ssim'] = []

    time_list = []
    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals()
        en_time = visuals['en_time'] # uint8
        time_list.append(en_time)

        HQ_rec_img = util.tensor2img(visuals['HQ_rec'])  # uint8
        gt_img = util.tensor2img(visuals['GT'])  # uint8
        LQ_mid_img = util.tensor2img(visuals['LQ_mid'])  # uint8
        LQ_ref_img = util.tensor2img(visuals['LQ_ref'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LQ_rec.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LQ_rec.png')
        util.save_img(HQ_rec_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_GT.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_GT.png')
        util.save_img(gt_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LQ_mid.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LQ_mid.png')
        util.save_img(LQ_mid_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LQ_ref.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LQ_ref.png')
        util.save_img(LQ_ref_img, save_img_path)

        # calculate PSNR and SSIM
        gt_img = gt_img / 255.
        HQ_rec_img = HQ_rec_img / 255.

        # LQ_mid_img = LQ_mid_img / 255.
        LQ_ref_img = LQ_ref_img / 255.

        # crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
        # if crop_border == 0:
        #     cropped_color_img = HQ_rec_img
        #     cropped_gt_img = gt_img
        #     cropped_LQ_ref_img = LQ_ref_img
        # else:
        #     cropped_color_img = HQ_rec_img[crop_border:-crop_border, crop_border:-crop_border, :]
        #     cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
        #     cropped_LQ_ref_img = LQ_ref_img[crop_border:-crop_border, crop_border:-crop_border, :]

        LQ_psnr = util.calculate_psnr(LQ_ref_img * 255, gt_img * 255)
        EN_psnr = util.calculate_psnr(HQ_rec_img * 255, gt_img * 255)


        LQ_ssim = util.calculate_ssim(LQ_ref_img * 255, gt_img * 255)
        EN_ssim = util.calculate_ssim(HQ_rec_img * 255, gt_img * 255)

        test_results['LQ_psnr'].append(LQ_psnr)
        test_results['EN_psnr'].append(EN_psnr)
        test_results['LQ_ssim'].append(LQ_ssim)
        test_results['EN_ssim'].append(EN_ssim)


        logger.info('{:20s} - LQ-PSNR: {:.2f} dB, En-PSNR: {:.2f} dB; \n LQ-SSIM: {:.6f} En-SSIM: {:.6f}..'.format(img_name, LQ_psnr, EN_psnr, LQ_ssim, EN_ssim))

    print("Avg time ==", np.sum(time_list[1:100])/99)
    # Average PSNR/SSIM results
    ave_psnr_LQ = sum(test_results['LQ_psnr']) / len(test_results['LQ_psnr'])
    ave_psnr_EN = sum(test_results['EN_psnr']) / len(test_results['EN_psnr'])

    ave_ssim_LQ = sum(test_results['LQ_ssim']) / len(test_results['LQ_ssim'])
    ave_ssim_EN = sum(test_results['EN_ssim']) / len(test_results['EN_ssim'])


    logger.info(
            '----Average LQ-PSNR/EN-PSNR  \n  LQ-SSIM/EN_SSIM  results for {}----\n\t LQ-psnr\EN-psnr: {:.2f}/{:.2f} dB; LQ-ssim/EN-ssim: {:.4f}/{:.4f}.\n'.format(
            test_set_name, ave_psnr_LQ, ave_psnr_EN, ave_ssim_LQ, ave_ssim_EN))
