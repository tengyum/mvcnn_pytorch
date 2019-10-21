import numpy as np
import glob
import torch.utils.data
import os
import math
import random
import pickle
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision as vision

import data_statics
from torchvision import transforms, datasets
from inlearn.utils import data_utils


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False,
                 num_models=0, num_views=12, shuffle=True, cam_settings='4_1_100_100_0.01'):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.root_dir = root_dir

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        self.cam_settings = cam_settings
        self.view_settings = [int(i) for i in self.cam_settings.split('_')[:2]]
        self.pixel_col = self.pixel_row = int(self.cam_settings.split('_')[2])
        self.num_views = np.prod(self.view_settings)

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/%s' % self.classnames[i] + '/%s' % set_
                                         + '/%s_*' % self.classnames[i] + '/%s.pickle' % self.cam_settings))
            # Select subset for different number of views
            # stride = int(12 / self.num_views)  # 12 6 4 3 2 1
            # all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models, len(all_files))])

        # self.filepaths = [p for p in self.filepaths for _ in range(self.num_views)]
        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.filepaths = filepaths_new

        c = np.prod([int(i) for i in self.cam_settings.split('_')[:4]])
        self.cam_settings_norm = data_statics.cam_settings_norms['c%s' % c]

        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[self.cam_settings_norm[cam_settings]['mean']] * 3,
                                 std=[self.cam_settings_norm[cam_settings]['std']] * 3)
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-4]
        class_id = self.classnames.index(class_name)
        # Use PIL instead for Original Data
        # imgs = []
        # for i in range(self.num_views):
        #     im = Image.open(self.filepaths[idx * self.num_views + i]).convert('RGB')
        #     if self.transform:
        #         im = self.transform(im)
        #     imgs.append(im)

        # Use PICKLE for Tengyu's Data
        with open(path, 'rb') as f:
            im = pickle.load(f)
        # plt.imshow(im, vmin=0, vmax=1, cmap='gray')
        # plt.show()
        if self.transform:
            im = self.transform(im)

        im = torch.reshape(im, (3, self.view_settings[1], self.pixel_row, self.view_settings[0], self.pixel_col))\
            .permute((1, 3, 0, 2, 4)).reshape((-1, 3, self.pixel_row, self.pixel_col))

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(2, 2)
        # for j in range(4):
        #     ax = axes[j % 2][j // 2]
        #     ax.imshow(im[j].cpu().numpy().transpose((1, 2, 0)))
        # plt.show()

        return (class_id, im, path)
        # return (class_id, torch.stack(imgs), self.filepaths[idx * self.num_views:(idx + 1) * self.num_views])


class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False,
                 num_models=0, num_views=12, cam_settings='4_1_100_100_0.01'):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.root_dir = root_dir
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        self.cam_settings = cam_settings
        self.view_settings = [int(i) for i in self.cam_settings.split('_')[:2]]
        self.pixel_col = self.pixel_row = int(self.cam_settings.split('_')[2])
        self.num_views = np.prod(self.view_settings)
        for i in range(len(self.classnames)):
            # Original Data
            # all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*shaded*.png'))
            # Tengyu's Data
            all_files = sorted(glob.glob(parent_dir + '/%s' % self.classnames[i] + '/%s' % set_
                                         + '/%s_*' % self.classnames[i] + '/%s.pickle' % self.cam_settings))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models, len(all_files))])

        c = np.prod([int(i) for i in self.cam_settings.split('_')[:4]])
        self.cam_settings_norm = data_statics.cam_settings_norms['c%s' % c]

        # self.filepaths = [p for p in self.filepaths for _ in range(self.num_views)]
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[self.cam_settings_norm[cam_settings]['mean']] * 3,
                                 std=[self.cam_settings_norm[cam_settings]['std']] * 3)
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        # view_i = idx % self.num_views  # Since I repeat the same file num_views times,
        # in Tengyu's representation, all views are stored in one pickle file

        # Since for large number of views, e.g., 400x100=40000 views, the repeating method is so slow cause each object
        # will have 40000 pictures for different views. Use random view now to leverage this time cost.
        view_i = random.randint(0, self.num_views - 1)
        # Orginal class_name position
        # class_name = path.split('/')[-3]

        # Tengyu's data position
        class_name = path.split('/')[-4]
        class_id = self.classnames.index(class_name)

        # Use PIL instead for Original Data
        # im = Image.open(self.filepaths[idx]).convert('RGB')
        # if self.transform:
        #     im = self.transform(im)

        # Use PICKLE for Tengyu's Data
        with open(path, 'rb') as f:
            im = pickle.load(f)
            # plt.imshow(im, vmin=0, vmax=1, cmap='gray')
            # plt.show()
        if self.transform:
            im = self.transform(im)
            # plt.imshow(im[view_i], vmin=0, vmax=1, cmap='gray')
            # plt.show()

        im = torch.reshape(im, (3, self.view_settings[1], self.pixel_row, self.view_settings[0], self.pixel_col)) \
            .permute((1, 3, 0, 2, 4)).reshape((-1, 3, self.pixel_row, self.pixel_col))
        im = im[view_i]
        # return im, class_id
        return (class_id, im, path)


if __name__ == '__main__':
    hyper_p = {
        'name': 'MVCNN',
        'batchSize': 8,  # it will be *num_views images in each batch for mvcnn
        'num_models': 1000,
        'lr': 5e-5,
        'weight_decay': 0.0,
        'no_pretraining': True,
        'cnn_name': 'vgg1',
        'num_views': 12,
        'cam_settings': '4_1_100_100_0.01',
        'train_path': '/media/tengyu/DataU/Data/ModelNet/ModelNet40_c40000/*/train',
        'val_path': '/media/tengyu/DataU/Data/ModelNet/ModelNet40_c40000/*/test',
        'train': True
    }
    n_models_train = hyper_p['num_models'] * hyper_p['num_views']
    train_dataset = SingleImgDataset(hyper_p['train_path'], scale_aug=False, rot_aug=False, num_models=n_models_train,
                                     num_views=hyper_p['num_views'], cam_settings=hyper_p['cam_settings'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    data_utils.get_mean_std(train_loader)
