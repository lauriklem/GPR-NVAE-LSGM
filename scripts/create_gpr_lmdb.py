import argparse
import torch
import numpy as np
import lmdb
import os
from PIL import Image
import sys
sys.path.append('../')
from gpr_dataset import GPRDataset

# run:
# python create_gpr_lmdb.py
# python create_gpr_lmdb.py --split validation


def list_data(folders, images_between, datadir, labels, namelength=2, first=1):
    """
    Lists GPR data as list of following lists: [input1, input2, ground_truth, label],
    where inputs and ground truth are paths to corresponding images. Since real and simulated images
    are in different folders and follow slightly different naming conventions, this should be used
    only for folders where images have been named similarly. Data is listed in order.

    :param folders: Iterable of used folders.
    :param images_between: Number of images between the inputs.
    :param datadir: Directory of the image folders.
    :param labels: Used labels.
    :param namelength: Length of the names of the image files.
    :param first: Integer corresponding to name of the first image (0.jpg or 1.jpg etc.).
    :return: List of lists (paths) [input1, input2, ground_truth, label]
    """

    data = []
    format_str = "{0:0" + str(namelength) + "d}.jpg"  # Determines the length of the image name (0.jpg vs 00.jpg etc.)

    # Image number indicates the survey line:
    dir1 = datadir + "/" + str(folders[0])
    n_images = len(os.listdir(dir1))  # number of images in a folder

    for survey_line in range(first, n_images - images_between - 1 + first):
        for folder in folders:
            image_dir = datadir + "/" + str(folder) + "/"
            input1 = image_dir + str(format_str.format(survey_line))
            input2 = image_dir + str(format_str.format(survey_line + images_between + 1))

            for label_idx in range(images_between):
                label = labels[label_idx]
                gt = image_dir + str(format_str.format(survey_line + label_idx + 1))  # Path of ground truth
                data.append([input1, input2, gt, label])

    return data


def gpr_lmdb(split, gpr_data_path, lmdb_path):
    # Validation means testing here
    assert split in {'train', 'validation'}
    # real_folders = list(range(4)) + list(range(5, 18))
    sim_folders = range(18, 34)

    real_train = []
    sim_train = []
    real_test = []
    sim_test = []

    """
    for folder in real_folders:
        folder_path = gpr_data_path + "/{}".format(folder)
        images = os.listdir(folder_path)
        for im in images:
            if folder in [2, 9]:
                real_test.append(folder_path + "/" + im)
            else:
                real_train.append(folder_path + "/" + im)
    """

    for folder in sim_folders:
        folder_path = gpr_data_path + "/{}".format(folder)
        images = os.listdir(folder_path)
        for im in images:
            if folder in [27, 30]:
                sim_test.append(folder_path + "/" + im)
            else:
                sim_train.append(folder_path + "/" + im)

    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    ind_path = lmdb_path + 'train_test_ind.pt'
    if os.path.exists(ind_path):
        ind_dat = torch.load(ind_path)
        train_ind = ind_dat['train']
        test_ind = ind_dat['test']
    else:
        train_combined = real_train + sim_train
        test_combined = real_test + sim_test
        train_ind = train_combined
        test_ind = test_combined
        torch.save({'train': train_ind, 'test': test_ind}, ind_path)

    file_ind = train_ind if split == 'train' else test_ind
    lmdb_path = lmdb_path + '%s.lmdb' % split

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    count = 0
    with env.begin(write=True) as txn:
        for row in file_ind:
            im = Image.open(row)
            im = np.array(im.getdata(), dtype="uint8").reshape((im.size[1], im.size[0], 3))
            if count == 0:
                print("Shape of the images is {}.".format(im.shape))

            txn.put(str(count).encode(), im)
            count += 1
            if count % 100 == 0:
                print(count)

        print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPR LMDB creator')
    parser.add_argument('--gpr_data_path', type=str, default='../datasets/gpr_pics/',
                        help='location of images from GPR dataset')
    parser.add_argument('--lmdb_path', type=str, default='../datasets/gpr-lmdb/',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()

    gpr_lmdb(args.split, args.gpr_data_path, args.lmdb_path)

