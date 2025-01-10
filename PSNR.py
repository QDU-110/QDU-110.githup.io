import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from options.test_options import TestOptions
from data_functions import create_dataset
from itertools import islice
import SimpleITK as sitk

def mae(imageA, imageB):
	err = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = TestOptions().parse()
    result_ssim = []
    sd = []
    j = 1

    if j == 1:
        sum_ssim = 0
        real_path_dir = './datasets/testPET'
        # fake_path_dir = os.path.join(opt.checkpoints_dir, opt.name, 'PET_results')
        fake_path_dir = './checkpoints/04_22_macgan/PET_results/'
        vector = []
        label = os.listdir(real_path_dir)
        for j in label:
            vector.append(j[0:18])
        vector = sorted(set(vector), key=vector.index)

        opt = TestOptions().parse()
        opt.num_threads = 1  # test code only supports num_threads=1
        opt.batch_size = 1  # test code only supports batch_size=1
        opt.serial_batches = True  # no shuffle

        # create dataset
        dataset = create_dataset(opt)
        PSNR = 0.0
        for i, (data, name) in enumerate(zip(islice(dataset, opt.num_test), vector)):
            real_pet = data['PET']
            real_pet_data = real_pet.squeeze().cpu().numpy()
            fake_pet_name = name + '_fake.nii.gz'
            fake_pet_path = os.path.join(fake_path_dir + fake_pet_name)
            fake_pet = sitk.ReadImage(fake_pet_path)
            fake_pet_data = sitk.GetArrayFromImage(fake_pet)
            for slice in range(real_pet_data.shape[1]):
                PSNR += compare_psnr(real_pet_data[:, :, slice], fake_pet_data[:, :, slice],data_range=real_pet_data.max()-real_pet_data.min())
            PSNR = PSNR.item() / opt.num_test

            sd.append(PSNR)

        result_sd = np.array(sd)
        arr_mean = np.mean(result_sd)
        arr_std = np.std(result_sd, ddof=1)
        print("Mean Value:%f" % arr_mean)
        print("Standard Deviation:%f" % arr_std)