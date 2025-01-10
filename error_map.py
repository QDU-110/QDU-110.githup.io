# 这个文件是SSIM.py MAE.py等文件的汇总
import os
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data_functions import create_dataset
from itertools import islice
import SimpleITK as sitk
if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = TestOptions().parse()



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
    for i, (data,name) in enumerate(zip(islice(dataset, opt.num_test), vector)):
        real_pet = data['PET']
        real_pet_data = real_pet.squeeze().cpu().numpy()
        fake_pet_name  = name + '_fake.nii.gz'
        fake_pet_path = os.path.join(fake_path_dir + fake_pet_name)
        fake_pet = sitk.ReadImage(fake_pet_path)
        fake_pet_data = sitk.GetArrayFromImage(fake_pet)

        diff = abs(real_pet_data - fake_pet_data)
        mid_slice = diff.shape[2] // 2  # 取中间切片
        diff_slice = diff[:, :, mid_slice]  # 提取中间切片的差值图
        plt.imshow(diff_slice, cmap='jet')
        plt.colorbar(label='Difference')
        plt.title('Difference Image (Slice {})'.format(mid_slice))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        break
