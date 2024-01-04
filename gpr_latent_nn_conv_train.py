from gpr_latent_nn_network_conv import InterpNetConv
import pickle
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.interp_utils import linear_interpolation
import matplotlib.pyplot as plt

"""
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, gens, gts):
        losses = []
        for generated, gt in zip(gens, gts):
            losses.append(1 - gpr_evaluation.struc_sim(generated, gt))
        return torch.mean(torch.tensor(losses, requires_grad=True))

"""


def list_latents(latent_list):
    x_list, gt_list = [], []
    for row in latent_list:
        in1, in2, gt, label = row
        dims = np.array(in1).shape
        label_img = np.full((1, dims[1], dims[2]), label)
        x = np.concatenate((in1, in2, label_img))
        x_list.append(x)
        gt_list.append(gt)

    return x_list, gt_list


def list_input_data(latents, latent_index):
    x_list = []
    for row in latents:
        l1, l2, lbl_img = row
        x_temp = []
        for i in range(l1[latent_index].shape[0]):
            x_temp.extend(np.stack((l1[latent_index][i], l2[latent_index][i], lbl_img[latent_index][0])))

        x_list.append(np.array(x_temp))

    return x_list


def list_gt_data(latents, latent_index):
    gt_list = []
    for row in latents:
        gt = row[latent_index]
        gt_list.append(gt)

    return gt_list


def main(args):
    print("Loading data...")
    fname = args.latents + "latents_train.pkl"
    f = open(fname, "rb")
    latents_train = pickle.load(f)
    f.close()

    fname = args.latents + "latents_test.pkl"
    f = open(fname, "rb")
    latents_test = pickle.load(f)
    f.close()

    fname = args.latents + "gt_train.pkl"
    f = open(fname, "rb")
    latents_gt_train = pickle.load(f)
    f.close()

    fname = args.latents + "gt_test.pkl"
    f = open(fname, "rb")
    latents_gt_test = pickle.load(f)
    f.close()

    dst_folder = "./latent_nn/checkpoints"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    n_latents = len(latents_train[0][0])
    for latent_index in range(n_latents):
        x_train = list_input_data(latents_train, latent_index)
        x_test = np.array(list_input_data(latents_test, latent_index))
        gt_train = list_gt_data(latents_gt_train, latent_index)
        gt_test = np.array(list_gt_data(latents_gt_test, latent_index))

        model = InterpNetConv(x_train[0].shape)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        n_epochs = args.epoch
        batch_size = args.batch_size
        loss_fn = nn.L1Loss()

        print("Index {}, training...".format(latent_index))
        for epoch in range(n_epochs):
            loss_train = train_interp_nn(x_train, gt_train, model, optimizer, batch_size, loss_fn, epoch)
            loss_test = test_interp_nn(x_test, gt_test, model, batch_size, loss_fn)
            if (epoch + 1) % 10 == 0:
                print("Epoch {}: train {:.3f}, test {:.3f}".format(epoch + 1, loss_train, loss_test))
        print()
        torch.save(model.state_dict(), dst_folder + "/checkpoint_conv{}.pt".format(latent_index))


def train_interp_nn(x_train, gt_train, model, optimizer, batch_size, loss_fn, ep):
    # trans = transforms.Compose([transforms.ToTensor()])
    train_ind = np.array(range(len(x_train)))
    rng = np.random.default_rng()
    rng.shuffle(train_ind)

    x_train_shuffled, gt_train_shuffled = [], []
    for ind in train_ind:
        x_train_shuffled.append(x_train[ind])
        gt_train_shuffled.append(gt_train[ind])

    x_train_shuffled = np.array(x_train_shuffled)
    gt_train_shuffled = np.array(gt_train_shuffled)

    model.train()
    losses = []
    for i in range(0, len(train_ind), batch_size):
        optimizer.zero_grad()
        x_batch = torch.tensor(x_train_shuffled[i:i + batch_size], dtype=torch.float).cuda()
        gt_batch = torch.tensor(gt_train_shuffled[i:i + batch_size], dtype=torch.float).cuda()

        pred = model(x_batch)
        loss = loss_fn(pred, gt_batch)
        losses.append(float(loss.item()))
        loss.backward()
        # print(torch.mean(torch.abs(model.seq[1].weight.grad)))
        optimizer.step()

        """
        if ep > 50:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(pred[0, 0].detach().cpu().numpy())
            ax[1].imshow(gt_batch[0, 0].cpu().numpy())
            ax[2].imshow(np.abs(pred[0, 0].detach().cpu().numpy() - gt_batch[0, 0].cpu().numpy()))
            plt.show()
        """

    return np.sum(losses)


def test_interp_nn(x_test, gt_test, model, batch_size, loss_fn):
    # trans = transforms.Compose([transforms.ToTensor()])
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = torch.tensor(x_test[i:i + batch_size], dtype=torch.float).cuda()
            gt_batch = torch.tensor(gt_test[i:i + batch_size], dtype=torch.float).cuda()

            pred = model(x_batch)
            loss = loss_fn(pred, gt_batch)
            losses.append(float(loss.item()))

    return np.sum(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('--latents', type=str, default='/path/to/latents.pkl',
                        help='location of the latents')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='size of a batch')

    args = parser.parse_args()

    main(args)