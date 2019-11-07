import numpy as np
import itertools
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse

from tools.Trainer import ModelNetTrainer
from models.MVCNN import MVCNN, SVCNN

from inlearn.utils import data_utils
from tools.ImgDataset import pickle_train_test_loader
from hyper_p_pk import ModelNet40_Hyper


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


def run_rxp(hyper_p, exp_settings, Data_Hyper, run_times):
    for exp_i in range(run_times):
        for cam, pre in exp_settings:
            hyper_p['pretraining'] = pre
            pretraining = hyper_p['pretraining']

            # STAGE 1
            log_dir = hyper_p['name'] + '_stage_1'
            create_folder(log_dir)

            data_hyper = Data_Hyper(cam)
            train_loader, val_loader = pickle_train_test_loader(*data_hyper.get_hyper_train(),
                                                                *data_hyper.get_hyper_test())

            train_dataset, val_dataset = train_loader.dataset, val_loader.dataset

            print('num_train_files: ' + str(len(train_dataset.pickle_dirs)))
            print('num_val_files: ' + str(len(val_dataset.pickle_dirs)))

            cnet = SVCNN(hyper_p['name'], nclasses=40, pretraining=pretraining, cnn_name=hyper_p['cnn_name'])

            optimizer = optim.Adam(cnet.parameters(), lr=hyper_p['lr'], weight_decay=hyper_p['weight_decay'])

            # n_models_train = hyper_p['num_models'] * hyper_p['num_views']

            # train_dataset = SingleImgDataset(hyper_p['train_path'], scale_aug=False, rot_aug=False, num_models=n_models_train,
            #                                  num_views=hyper_p['num_views'])
            # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
            #
            # val_dataset = SingleImgDataset(hyper_p['val_path'], scale_aug=False, rot_aug=False, test_mode=True)
            # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

            trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn',
                                      log_dir, num_views=1)
            trainer.train(30)

            # STAGE 2
            log_dir = hyper_p['name'] + '_stage_2'
            create_folder(log_dir)

            data_hyper.single = False

            hyper_train_data, hyper_train_loader = data_hyper.get_hyper_train()
            hyper_train_loader['batch_size'] = hyper_p['batch_size']

            train_loader, val_loader = pickle_train_test_loader(hyper_train_data, hyper_train_loader,
                                                                *data_hyper.get_hyper_test())

            train_dataset, val_dataset = train_loader.dataset, val_loader.dataset

            print('num_train_files: ' + str(len(train_dataset.pickle_dirs)))
            print('num_val_files: ' + str(len(val_dataset.pickle_dirs)))

            cnet_2 = MVCNN(hyper_p['name'], cnet, nclasses=40, cnn_name=hyper_p['cnn_name'], num_views=train_dataset.view_num)
            del cnet

            optimizer = optim.Adam(cnet_2.parameters(), lr=hyper_p['lr'], weight_decay=hyper_p['weight_decay'], betas=(0.9, 0.999))


            trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn',
                                      log_dir, num_views=train_dataset.view_num)
            trainer.train(30)

            rst_dir = data_hyper.get_hyper_rst()['rst_dir']
            with open(rst_dir, 'a+') as f:
                line = '%s\t%s\t%s\t%f\n' % (cam,
                                             'mvcnn_resnet18',
                                             hyper_p['pretraining'],
                                             trainer.max_acc)
                f.write(line)


def main():
    hyper_p = {
        'name': 'MVCNN',
        'batch_size': 8,  # it will be *num_views images in each batch for mvcnn
        'num_models': 1000,
        'lr': 5e-5,
        'weight_decay': 0.0,
        'pretraining': True,
        'cnn_name': 'resnet18',
        'train': True
    }

    # c10000
    # cam_settings = ['4_1_50_50_0.02', '8_2_25_25_0.02', '20_5_10_10_0.02',
    #                 '40_10_5_5_0.02', '100_25_2_2_0.02', '200_50_1_1_0.02']

    # c22500
    cam_settings = ['4_1_75_75_0.01', '12_3_25_25_0.01', '20_5_15_15_0.01']
                    # '60_15_5_5_0.01', '100_25_3_3_0.01', '300_75_1_1_0.01']

    # c40000
    # cam_settings = ['4_1_100_100_0.01', '8_2_50_50_0.01', '16_4_25_25_0.01', '20_5_20_20_0.01']
                    # '40_10_10_10_0.01', '80_20_5_5_0.01', '100_25_4_4_0.01', '200_50_2_2_0.01', '400_100_1_1_0.01']
    pretraineds = [True, False]
    exp_settings = list(itertools.product(*[cam_settings, pretraineds]))
    Data_Hyper = ModelNet40_Hyper
    run_times = 5

    run_rxp(hyper_p, exp_settings, Data_Hyper, run_times)


if __name__ == '__main__':
    main()
