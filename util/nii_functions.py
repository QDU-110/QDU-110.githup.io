import os

import numpy as np
import SimpleITK as sitk

def read_image(path):
        """
        读取给定路径下的影像
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

def resample_image(input_path, output_path, target_size=(128, 128, 128)):
        """
        重采样影像
        Parameters:
                input_path:输入影像路径
                output_path:输出影像路径
                target_size:目标影像尺寸
        """
        image = sitk.ReadImage(input_path)
        target_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                      zip(image.GetSize(), image.GetSpacing(), target_size)]
        target_direction = image.GetDirection()
        target_origin = image.GetOrigin()
        target_pixel_type = image.GetPixelID()
        target_image = sitk.Image(list(target_size), target_pixel_type)
        target_image.SetSpacing(target_spacing)
        target_image.SetDirection(target_direction)
        target_image.SetOrigin(target_origin)

        interpolator = sitk.sitkLinear
        default_value = 0
        resampled_image = sitk.Resample(image, target_image, sitk.Transform(), interpolator, default_value)

        sitk.WriteImage(resampled_image, output_path)

def inverseImage(fake_B,name,path):
        """
        转换影像到目标格式
        """
        fake_data = np.transpose(np.squeeze(fake_B.data.cpu().numpy()), (2, 1, 0))
        reference_datapath = './datasets/PET_128/002_S_5018_PET_128.nii.gz'
        reference_pet = sitk.ReadImage(reference_datapath)
        target_spacing = reference_pet.GetSpacing()
        target_direction = reference_pet.GetDirection()
        target_origin = reference_pet.GetOrigin()
        fake_B_NII = sitk.GetImageFromArray(fake_data)
        fake_B_NII.SetSpacing(target_spacing)
        fake_B_NII.SetDirection(target_direction)
        fake_B_NII.SetOrigin(target_origin)
        save_path = os.path.join(path,'{}_fake.nii.gz'.format(name))
        sitk.WriteImage(fake_B_NII,save_path)