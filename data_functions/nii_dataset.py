import SimpleITK as sitk
import os
import re
import numpy as np
import random
import torch
import torch.utils.data
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def lstFiles(Path):
    images_list = []  # create an empty list, the raw image data_functions files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
    images_list = sorted(images_list, key=numericalSort)
    return images_list
class NifitDataSet(torch.utils.data.Dataset):

    def __init__(self, data_path,
                 which_direction='AtoB',
                 shuffle_labels=False,):
        # Init membership variables
        self.data_path = data_path
        self.images_list = lstFiles(os.path.join(data_path, 'images'))
        self.labels_list = lstFiles(os.path.join(data_path, 'labels'))
        self.images_size = len(self.images_list)
        self.labels_size = len(self.labels_list)

        self.which_direction = which_direction

        self.shuffle_labels = shuffle_labels

        self.bit = sitk.sitkFloat32

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def __getitem__(self, index):

        data_path = self.images_list[index]

        if self.shuffle_labels is True:

            index_B = random.randint(0, self.labels_size - 1)
            label_path = self.labels_list[index_B]

        else:

            label_path = self.labels_list[index]

        if self.which_direction == 'AtoB':

            data_path = data_path
            label_path = label_path

        # read image and label
        image = self.read_image(data_path)
        label = self.read_image(label_path)
        sample = {'image': image, 'label': label}

        # convert sample to tf tensors
        image_np = abs(sitk.GetArrayFromImage(sample['image']))
        label_np = abs(sitk.GetArrayFromImage(sample['label']))

        # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])  (actually it´s the contrary)
        image_np = np.transpose(image_np, (2, 1, 0))
        label_np = np.transpose(label_np, (2, 1, 0))

        # label_np = (label_np - 127.5) / 127.5
        # image_np = (image_np - 127.5) / 127.5

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        return torch.from_numpy(image_np), torch.from_numpy(label_np)

    def __len__(self):
        return len(self.images_list)


