# 这个文件是SSIM.py MAE.py等文件的汇总
import os
import numpy as np
import torch
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from options.test_options import TestOptions
from data_functions import create_dataset
from itertools import islice
import SimpleITK as sitk


def caculate_mae(imageA, imageB):
    err = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def guassian_kernel(source, target, kernel_mul=3.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def caculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = TestOptions().parse()
    result_ssim = []
    ssim = []
    mae = []
    psnr = []
    mmd = []
    mse = []
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
        SSIM = 0.0
        MAE = 0.0
        MSE = 0.0
        PSNR = 0.0
        MMD = 0.0
        for i, (data, name) in enumerate(zip(islice(dataset, opt.num_test), vector)):
            real_pet = data['PET']
            real_pet_data = real_pet.squeeze().cpu().numpy()
            fake_pet_name = name + '_fake.nii.gz'
            fake_pet_path = os.path.join(fake_path_dir + fake_pet_name)
            fake_pet = sitk.ReadImage(fake_pet_path)
            fake_pet_data = sitk.GetArrayFromImage(fake_pet)
            for slice in range(real_pet_data.shape[1]):
                # 1.SSIM
                SSIM += compare_ssim(real_pet_data[:, slice, :], fake_pet_data[:, slice, :], win_size=11,
                                     gradient=False,
                                     data_range=real_pet_data.max() - real_pet_data.min(),
                                     gaussian_weights=False, full=False, dynamic_range=None)
                # 2.MAE
                MAE += caculate_mae(real_pet_data[:, slice, :], fake_pet_data[:, slice, :])
                # 3. PSNR
                PSNR += compare_psnr(real_pet_data[:, slice, :], fake_pet_data[:, slice, :],
                                     data_range=real_pet_data.max() - real_pet_data.min())
                # 4. MMD
                MMD += mmd_rbf(Variable(torch.Tensor(real_pet_data[:, slice, :])),
                               Variable(torch.Tensor(fake_pet_data[:, slice, :])))
                # 5. MSE
                MSE += caculate_mse(real_pet_data[:, slice, :], fake_pet_data[:, slice, :])

            SSIM = SSIM.item() / opt.num_test
            MAE = MAE / opt.num_test
            PSNR = PSNR.item() / opt.num_test
            MMD = MMD.item() / opt.num_test
            MSE = MSE.item() / opt.num_test
            ssim.append(SSIM)
            mae.append(MAE)
            psnr.append(PSNR)
            mmd.append(MMD)
            mse.append(MSE)

        result_ssim = np.array(ssim)
        ssim_mean = np.mean(ssim)
        ssim_std = np.std(ssim, ddof=1)

        result_mae = np.array(mae)
        mae_mean = np.mean(mae)
        mae_std = np.std(mae, ddof=1)

        result_psnr = np.array(psnr)
        psnr_mean = np.mean(psnr)
        psnr_std = np.std(psnr, ddof=1)

        result_mmd = np.array(mmd)
        mmd_mean = np.mean(mmd)
        mmd_std = np.std(mmd, ddof=1)

        result_mse = np.array(mse)
        mse_mean = np.mean(mse)
        mse_std = np.std(mse, ddof=1)
        filename = 'metrics.txt'
        filepath = './checkpoints/04_22_macgan/'

        with open(filepath + filename, 'w') as f:
            f.write("MAE: {:.4f} +/- {:.4f}\n".format(mae_mean, mae_std))
            f.write("MSE: {:.4f} +/- {:.4f}\n".format(mse_mean, mse_std))
            f.write("PSNR: {:.4f} +/- {:.4f}\n".format(psnr_mean, psnr_std))
            f.write("MMD: {:.4f} +/- {:.4f}\n".format(mmd_mean, mmd_std))
            f.write("SSIM: {:.4f} +/- {:.4f}\n".format(ssim_mean, ssim_std))

        f.close()
