"""Script to split the Kather-19 dataset. 
Code from https://github.com/agaldran/t3po
"""

import shutil
import random, os, os.path as osp
import argparse


def create_val_img_folder(path_data_in='data/NCT-CRC-HE-100K'):
    '''
    This method is responsible for separating kather-19 images into separate sub folders
    path_data_in points to a folder containing the 9 subfolders resulting from downloading and uncompressing:
    https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
    '''
    path_data = path_data_in.replace('NCT-CRC-HE-100K', 'kather19')
    path_train_data = osp.join(path_data, 'train')
    path_val_data = osp.join(path_data, 'val')
    path_test_data = osp.join(path_data, 'test')

    os.makedirs(osp.join(path_train_data), exist_ok=True)
    os.makedirs(osp.join(path_val_data), exist_ok=True)
    os.makedirs(osp.join(path_test_data), exist_ok=True)

    subfs = os.listdir(path_data_in)
    # loop over subfolders in train data folder
    for f in subfs:
        os.makedirs(osp.join(path_train_data, f), exist_ok=True)
        os.makedirs(osp.join(path_val_data, f), exist_ok=True)
        os.makedirs(osp.join(path_test_data, f), exist_ok=True)

        path_this_f = osp.join(path_data_in, f)
        print(path_this_f)
        im_list = os.listdir(path_this_f)

        train_len = int(0.70 * len(im_list))  # random 70% for train
        val_len = int(0.15 * len(im_list))  # random 15% for val
        test_len = int(0.15 * len(im_list))  # random 15% for test

        # take 15% out of im_list for test
        im_list_test = random.sample(im_list, test_len)
        im_list_train = list(set(im_list) - set(im_list_test))

        # take 15% out of im_list for val, the rest is train
        im_list_val = random.sample(im_list_train, val_len)
        im_list_train = list(set(im_list_train) - set(im_list_val))

        # loop over subfolders and move train/val/test images over to corresponding subfolders
        for im in im_list_train:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_train_data, f, im))
        for im in im_list_val:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_val_data, f, im))
        for im in im_list_test:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_test_data, f, im))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--path_data_in', type=str, default='data/NCT-CRC-HE-100K', help="")

    args = parser.parse_args()
    create_val_img_folder(args.path_data_in)
