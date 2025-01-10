import os
import numpy as np
import torch
from data_functions.base_dataset import BaseDataset
from data_functions.image_folder import make_dataset
import SimpleITK as sitk
from util import nii_functions

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data_functions/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data_functions/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_MRI = os.path.join(opt.dataroot, opt.phase + 'MRI')  # create a path '/path/to/data_functions/trainA'
        self.dir_PET = os.path.join(opt.dataroot, opt.phase + 'PET')  # create a path '/path/to/data_functions/trainB'
        self.MRI_paths = sorted(make_dataset(self.dir_MRI, opt.max_dataset_size))  # load images from '/path/to/data_functions/trainA'
        self.PET_paths = sorted(make_dataset(self.dir_PET, opt.max_dataset_size))  # load images from '/path/to/data_functions/trainB'
        self.MRI_size = len(self.MRI_paths)  # get the size of dataset A
        self.PET_size = len(self.MRI_paths)  # get the size of dataset B
        assert self.MRI_size == self.PET_size
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """
        Return a data_functions point and its metadata information.

        Parameters:
            index:一个随机索引

        Return:
            返回一个字典
            MRI (tensor) - - MRI域的影像tensor
            PET (tensor) - - 目标PET域影像tensor
            MRI_paths (str) - - MRI影像路径
            PET_paths (str) - - PET影像路径
        """
        # read a image given a random integer index
        MRI_path = self.MRI_paths[index % self.MRI_size]
        PET_path = self.PET_paths[index % self.PET_size]
        MRI_image = nii_functions.read_image(MRI_path)
        PET_image = nii_functions.read_image(PET_path)
        MRI = abs(sitk.GetArrayFromImage(MRI_image))
        PET = abs(sitk.GetArrayFromImage(PET_image))
        MRI = np.transpose(MRI, (2, 1, 0))
        PET = np.transpose(PET, (2, 1, 0))

        MRI = self.intensity_normalize(MRI) # [-1,1]
        PET = self.intensity_normalize(PET)

        MRI = torch.unsqueeze(torch.FloatTensor(MRI), dim=0)
        PET = torch.unsqueeze(torch.FloatTensor(PET), dim=0)
        # print(A.size(), B.size())
        assert MRI.size() == PET.size()
        # apply the same transform to both A and B

        return {'MRI': MRI, 'PET': PET, 'MRI_paths': MRI_path, 'PET_paths': PET_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.MRI_paths)

    def intensity_normalize(self, input: np.array):
        output =  2 * (input - np.min(input)) / (np.max(input) - np.min(input)) - 1
        return output


if __name__=='__main__':
    A = sitk.GetArrayFromImage(sitk.ReadImage('../datasets/MRI_128/002_S_5018_MRI_128.nii.gz'))
    A = A[:64,:64,:64]
    print(A.shape)
