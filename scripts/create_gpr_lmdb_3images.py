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
# python create_gpr_lmdb_3images.py
# python create_gpr_lmdb_3images.py --split validation

"""
DON'T USE, CANNOT STORE FLOAT32 TO LMDB CORRECTLY
"""


def gpr_lmdb_3images(split, gpr_data_path, lmdb_path):
    # Validation means testing here
    assert split in {'train', 'validation'}
    ds = GPRDataset(stdev=0, datadir=gpr_data_path, images_between=[4])

    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    ind_path = lmdb_path + 'train_test_ind.pt'
    if os.path.exists(ind_path):
        ind_dat = torch.load(ind_path)
        train_ind = ind_dat['train']
        test_ind = ind_dat['test']
    else:
        # train_combined = real_train + sim_train
        # test_combined = real_test + sim_test
        train_ind = ds.training_sim
        test_ind = ds.testing_sim
        torch.save({'train': train_ind, 'test': test_ind}, ind_path)

    file_ind = train_ind if split == 'train' else test_ind
    lmdb_path = lmdb_path + '%s.lmdb' % split

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    count = 0
    with env.begin(write=True) as txn:
        for row in file_ind:
            input1, input2, gt, label = row
            im1 = Image.open(input1)
            lblimg = np.full(fill_value=label, shape=(im1.size[1], im1.size[0], 1), dtype="float32")
            im1 = np.array(im1.getdata(), dtype="float32").reshape((im1.size[1], im1.size[0], 3))
            im2 = Image.open(input2)
            im2 = np.array(im2.getdata(), dtype="float32").reshape((im2.size[1], im2.size[0], 3))

            gt = Image.open(gt)
            gt = np.array(gt.getdata(), dtype="float32").reshape((gt.size[1], gt.size[0], 3))

            im = np.concatenate((im1, im2, lblimg, gt), axis=-1)
            if count == 0:
                print(im.shape)
            txn.put(str(count).encode(), im)
            count += 1
            if count % 100 == 0:
                print(count)

        print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPR LMDB creator')
    parser.add_argument('--gpr_data_path', type=str, default='../datasets/gpr_pics/',
                        help='location of images from GPR dataset')
    parser.add_argument('--lmdb_path', type=str, default='../datasets/gpr-lmdb_3images/',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()

    gpr_lmdb_3images(args.split, args.gpr_data_path, args.lmdb_path)

