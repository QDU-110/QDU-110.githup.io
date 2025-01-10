import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data_functions import create_dataset
from models import networks
from torchvision.utils import make_grid,save_image
if __name__ == '__main__':

    opt = TrainOptions().parse()   # 导入options
    dataset = create_dataset(opt)  # 创建dataset
    dataset_size = len(dataset)    # 返回dataset尺寸
    print('The number of training images = %d' % dataset_size)
    Generator = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                  norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                  init_gain=opt.init_gain,
                                  where_add=opt.where_add, upsample=opt.upsample).cuda()
    # Generator.load_state_dict(torch.load('./checkpoints/04_20/200_net_G.pth'))
    # the number of training iterations in current epoch, reset to 0 every epoch
    for i, data in enumerate(dataset):  # inner loop within one epoch
        mri = data['MRI'][0]
        fake_pet = Generator(mri)
        break
    print('111')