import os, os.path
from PIL import Image
import numpy as np
import pickle
from chainer.dataset import dataset_mixin
from chainer import links, Variable

"""
Dataset for GPR images
"""

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


def list_extrapolation(datadir):
    labels = np.array(range(1, 5)) / float(5)
    data_extra = []
    data_normal = []

    # Image number indicates the survey line:
    dir1 = datadir + "/" + str(27)
    n_images = len(os.listdir(dir1))  # number of images in a folder
    for test_folder in [27, 30]:
        if test_folder == 27:
            left_ind = [2, 3, 11, 12, 13]
        else:
            left_ind = [1, 2, 3, 12, 13, 14]
        for survey_line in range(0, n_images - 4 - 1):
            image_dir = datadir + "/{}/".format(test_folder)
            input1 = image_dir + str(survey_line) + ".jpg"
            input2 = image_dir + str(survey_line + 5) + ".jpg"
            for label_idx in range(4):
                label = labels[label_idx]
                gt = image_dir + str(survey_line + label_idx + 1) + ".jpg"  # Path of ground truth
                if survey_line in left_ind:
                    data_extra.append([input1, input2, gt, label])
                else:
                    data_normal.append([input1, input2, gt, label])

    return data_extra, data_normal


class GPRDataset(dataset_mixin.DatasetMixin):
    def __init__(self,
                 stdev,
                 datadir='',
                 seed=101,
                 images_between=None,
                 real_folders_train=None,
                 real_folders_test=None,
                 simulated_folders_train=None,
                 simulated_folders_test=None,
                 verbose=True,
                 rgb=True,
                 onlyreal=False,
                 onlysim=False,
                 ):
        self.rgb = rgb
        self.datadir = datadir
        rng = np.random.default_rng(seed=seed)
        n_testing = 2  # number of folders used for testing
        if real_folders_train is None:
            real_folders = list(range(18))
            perm = rng.permutation(real_folders)
            real_folders_test = perm[:n_testing]
            real_folders_train = perm[n_testing:]

            # Folders 3 and 4 have the same images for some reason
            real_folders_train = np.delete(real_folders_train, np.argwhere(real_folders_train == 4))

        if simulated_folders_train is None:
            simulated_folders = list(range(18, 34))
            perm2 = rng.permutation(simulated_folders)
            simulated_folders_test = perm2[:n_testing]
            simulated_folders_train = perm2[n_testing:]

        if verbose:
            print("Loading dataset from " + datadir)
            print("No folder input, using random folders defined by seed.")
            print("Testing folders: {} for real, {} for sim.".format(real_folders_test, simulated_folders_test))
            print("Training folders: {} for real, {} for sim.".format(real_folders_train, simulated_folders_train))

        self.stdev = stdev

        if images_between is None:
            images_between = [4]

        labels_list = []
        name_str = ""  # naming string used for pickle files
        for ib in images_between:
            name_str += "_{}".format(ib)
            labels = np.array(range(1, ib + 1)) / float(ib + 1)
            labels_list.append(labels)

        # List image paths as a list of lists [input1, input2, ground_truth, label]
        data_real_train = []
        data_real_test = []
        data_sim_train = []
        data_sim_test = []

        for i in range(len(images_between)):
            ib = images_between[i]
            labels = labels_list[i]
            for folder in real_folders_train:
                nl = 1
                if folder < 18 or 34 <= folder <= 36:
                    nl = 2
                elif 37 <= folder <= 55:
                    nl = 3
                data_real_train += list_data([folder], ib, datadir, labels, namelength=nl, first=1)

            for folder in real_folders_test:
                nl = 1
                if folder < 18 or 34 <= folder <= 36:
                    nl = 2
                elif 37 <= folder <= 55:
                    nl = 3
                data_real_test += list_data([folder], ib, datadir, labels, namelength=nl, first=1)

            data_sim_train += list_data(simulated_folders_train, ib, datadir, labels, namelength=1, first=0)
            data_sim_test += list_data(simulated_folders_test, ib, datadir, labels, namelength=1, first=0)

        if onlyreal:
            training_data = data_real_train
        elif onlysim:
            training_data = data_sim_train
        else:
            training_data = data_real_train + data_sim_train
        self.training_data = training_data
        self.training_real = data_real_train
        self.training_sim = data_sim_train

        # Separate real and sim for testing to calculate metrics separately for them
        self.testing_real = data_real_test
        self.testing_sim = data_sim_test

        self.extrapolation_part, self.normal_interp_part = list_extrapolation(datadir)

        # Write to files
        f = open(datadir + "/training_data{}.npy".format(name_str), "wb")
        np.save(f, self.training_data)
        f.close()

        f = open(datadir + "/testing_real{}.npy".format(name_str), "wb")
        np.save(f, self.testing_real)
        f.close()

        f = open(datadir + "/testing_sim{}.npy".format(name_str), "wb")
        np.save(f, self.testing_sim)
        f.close()

        if verbose:
            print("Dataset loading complete")
            print("Number of training combinations: {} real, {} sim, {} total".format(len(data_real_train),
                                                                                      len(data_sim_train),
                                                                                      len(self.training_data)))
            print("Number of testing combinations: {} real, {} sim, {} total".format(len(self.testing_real),
                                                                                     len(self.testing_sim),
                                                                                     len(self.testing_real) +
                                                                                     len(self.testing_sim)))
            print()

    def __len__(self):
        return len(self.training_data)

    def len_training_real(self):
        return len(self.training_real)

    def len_training_sim(self):
        return len(self.training_sim)

    def len_testing_real(self):
        return len(self.testing_real)

    def len_testing_sim(self):
        return len(self.testing_sim)

    def image_from_label(self, label_float):
        """
        Generate label image using given label (float between 0-1)
        """
        label_float = label_float * 2.0 - 1.0  # scale to [-1, 1]

        if self.stdev > 0:
            # Gaussian distribution
            rng = np.random.default_rng()
            img_label = rng.normal(label_float, self.stdev, size=(1, 256, 256)).astype("float32")
        else:
            # Constant
            img_label = np.full(shape=(1, 256, 256), fill_value=label_float, dtype="float32")  # float32 required for forward pass
        return img_label

    def get_example(self, i, test=False, real_or_sim="real", fixedpos=False):
        """
        Returns one training or testing example from the respective data. Example is returned as
        input1, input2, and label image concatenated channel-wise, ground-truth is returned separately.

        :param i: Index of the example in data (random number between 0 - len(data)).
        :param test: Return example from testing or training data.
        :param real_or_sim: Return example from real or sim testing data.
        :return: input1, input2, and label image concatenated into one image, ground-truth image.
        """
        i = int(i)  # Sometimes using sampler causes i to be numpy float64
        if test:
            if real_or_sim == "real":
                row = self.testing_real[i]
            else:
                row = self.testing_sim[i]
        else:
            row = self.training_data[i]

        in1_dir = row[0]
        in2_dir = row[1]
        gt_dir = row[2]
        label = row[3]

        img1 = Image.open(in1_dir)
        img2 = Image.open(in2_dir)
        gt = Image.open(gt_dir)

        if self.rgb:
            img1 = img1.convert("RGB")
            img1 = np.asarray(img1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

            img2 = img2.convert("RGB")
            img2 = np.asarray(img2).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

            if not test:
                gt = gt.convert("RGB")
                gt = np.asarray(gt).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        else:
            img1 = np.asarray(img1).astype("f") / 128.0 - 1.0
            if len(img1.shape) > 2:
                img1 = img1[:, :, 0]
            img1 = img1[np.newaxis]

            img2 = np.asarray(img2).astype("f") / 128.0 - 1.0
            if len(img2.shape) > 2:
                img2 = img2[:, :, 0]
            img2 = img2[np.newaxis]

            if not test:
                gt = np.asarray(gt).astype("f") / 128.0 - 1.0
                if len(gt.shape) > 2:
                    gt = gt[:, :, 0]
                gt = gt[np.newaxis]

        lbl_img = self.image_from_label(label)

        if not fixedpos:
            img3 = np.concatenate((img1, img2, lbl_img), axis=0)
            return img3, gt
        else:
            img3 = np.concatenate((img1, img2), axis=0)
            return img3, gt, label

    def get_image(self, in1_dir, in2_dir, label):
        img1 = Image.open(in1_dir)
        img1 = img1.convert("RGB")
        img1 = np.asarray(img1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

        img2 = Image.open(in2_dir)
        img2 = img2.convert("RGB")
        img2 = np.asarray(img2).astype("f").transpose(2, 0, 1) / 128.0 - 1.0

        lbl_img = self.image_from_label(label)

        img3 = np.concatenate((img1, img2, lbl_img), axis=0)

        return img3

    def get_example_interp(self, i, real_or_sim="real"):
        """
        Return example for linear interpolation.
        """

        if real_or_sim == "real":
            row = self.testing_real[i]
        else:
            row = self.testing_sim[i]

        in1_dir = row[0]
        in2_dir = row[1]
        gt_dir = row[2]
        label = row[3]

        img1 = Image.open(in1_dir)
        img2 = Image.open(in2_dir)
        gt = Image.open(gt_dir)

        return img1, img2, gt, label




