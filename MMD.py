import numpy as np
import os
import torch
from torch.autograd import Variable
from options.test_options import TestOptions
from data_functions import create_dataset
from itertools import islice
import SimpleITK as sitk


def guassian_kernel(source, target, kernel_mul=3.0, kernel_num=5, fix_sigma=None):

    # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    n_samples = int(source.size()[0]) + int(target.size()[0])

    # 将source, target按列方向合并
    total = torch.cat([source, target], dim=0)

    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    # 求任意两个数据之间的和，
    # 得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)

    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值
    # （比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

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
        MMD = 0.0
        for i, (data, name) in enumerate(zip(islice(dataset, opt.num_test), vector)):
            real_pet = data['PET']
            real_pet_data = real_pet.squeeze().cpu().numpy()
            fake_pet_name = name + '_fake.nii.gz'
            fake_pet_path = os.path.join(fake_path_dir + fake_pet_name)
            fake_pet = sitk.ReadImage(fake_pet_path)
            fake_pet_data = sitk.GetArrayFromImage(fake_pet)
            for slice in range(real_pet_data.shape[1]):
                MMD += mmd_rbf(Variable(torch.Tensor(real_pet_data[:, :, slice])),Variable(torch.Tensor(fake_pet_data[:, :, slice])))
            MMD = MMD.item() / opt.num_test

            sd.append(MMD)

        result_sd = np.array(sd)
        arr_mean = np.mean(result_sd)
        arr_std = np.std(result_sd, ddof=1)
        print("Mean Value:%f" % arr_mean)
        print("Standard Deviation:%f" % arr_std)
