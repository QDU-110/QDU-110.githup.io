import os

import util.util
from options.test_options import TestOptions
from data_functions import create_dataset
from models import create_model
from itertools import islice
from util import nii_functions,util
from torchvision.utils import make_grid,save_image
import SimpleITK as sitk
import numpy as np
import nibabel as nib
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
# options
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    # create dataset
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print('Loading model %s' % opt.model)
    # sample random z
    z_samples = model.get_z_random(opt.num_test, opt.nz)
    # test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        data_name = data['PET_paths'][0][-25:-7]
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))
        real_A, fake_B, real_B = model.test(z_samples[[i]], encode=True)
        path = os.path.join(opt.checkpoints_dir, opt.name, 'PET_results')
        util.mkdir(path)
        nii_functions.inverseImage(fake_B,data_name,path)
        print('{} generates successfully !'.format(data_name))

    print('Please Check Generation Results In {}'.format(path))
