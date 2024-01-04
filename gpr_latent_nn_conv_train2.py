from gpr_latent_nn_network_conv import InterpNetConv
import pickle
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from gpr_vae_3to1_latent_analysis import list_data_3to1
from nvae import NVAE
from util import utils
from util.interp_utils import linear_interpolation
from PIL import Image
import gpr_evaluation
import matplotlib.pyplot as plt


class CustomL1(nn.Module):
    def __init__(self):
        super(CustomL1, self).__init__()

    def forward(self, gens, gts):
        losses = []
        for generated, gt in zip(gens, gts):
            losses.append(1 - gpr_evaluation.struc_sim(generated, gt))
        return torch.mean(torch.tensor(losses, requires_grad=True))


def lerp_latents(latent_list):
    gt_list = []
    for row in latent_list:
        in1, in2, _, label = row
        in1 = torch.tensor(in1, dtype=torch.float).cuda()
        in2 = torch.tensor(in2, dtype=torch.float).cuda()
        gt = linear_interpolation(in1, in2, label)[0]
        gt_list.append(gt.cpu().numpy())

    return gt_list


def prepare_x(x):
    x_list = []
    l1, l2, label_img = x
    for i in range(len(l1)):
        z1 = l1[i]
        z2 = l2[i]
        x_list.append(np.concatenate((z1, z2, label_img)))
    return np.array(x_list)


def main(args):
    checkpoint_vae = torch.load(args.checkpoint_vae, map_location='cpu')
    vae_args = checkpoint_vae['args']
    arch_instance_nvae = utils.get_arch_cells(vae_args.arch_instance, vae_args.use_se)
    vae = NVAE(vae_args, arch_instance_nvae)
    vae.load_state_dict(checkpoint_vae['vae_state_dict'])
    vae = vae.cuda()
    vae.eval()

    latent_folder = args.latents

    print("Loading data...")
    fname = latent_folder + "latents_train.pkl"
    f = open(fname, "rb")
    x_train = pickle.load(f)
    f.close()

    fname = latent_folder + "latents_test.pkl"
    f = open(fname, "rb")
    x_test = pickle.load(f)
    f.close()

    fname = latent_folder + "gt_train.pkl"
    f = open(fname, "rb")
    gt_train = pickle.load(f)
    f.close()

    fname = latent_folder + "gt_test.pkl"
    f = open(fname, "rb")
    gt_test = pickle.load(f)
    f.close()

    x_list = prepare_x(x_train[0])
    model = InterpNetConv(x_list[0].shape, len(x_list))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    loss_fn = nn.L1Loss(reduction="sum")
    n_epochs = args.epoch
    batch_size = args.batch_size

    print("Training...")
    for epoch in range(n_epochs):
        loss_train = train_interp_nn(x_train, gt_train, model, vae, optimizer, batch_size, loss_fn)
        loss_test = test_interp_nn(x_test, gt_test, model, vae, batch_size, loss_fn)
        print("Epoch {}: train {:.4f}, test {:.4f}".format(epoch + 1, loss_train, loss_test))

    dst_folder = "./latent_nn/checkpoints"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    torch.save(model.state_dict(), dst_folder + "/checkpoint_conv.pt")


def train_interp_nn(x_train, gt_train, model, vae, optimizer, batch_size, loss_fn):
    model.train()
    vae.eval()
    perm_dims = (0, 2, 3, 1)
    train_ind = np.array(range(len(x_train)))
    rng = np.random.default_rng()
    rng.shuffle(train_ind)

    x_train_shuffled, gt_train_shuffled = [], []
    for ind in train_ind:
        x_train_shuffled.append(x_train[ind])
        gt_train_shuffled.append(gt_train[ind])

    losses = []
    for i in range(0, len(train_ind), batch_size):
        x_batch = None
        for x in x_train_shuffled[i:i + batch_size]:
            if x_batch is None:
                x_batch = torch.tensor(prepare_x(x), dtype=torch.float).cuda().unsqueeze(0)
            else:
                x_batch = torch.vstack((x_batch, torch.tensor(prepare_x(x), dtype=torch.float).cuda().unsqueeze(0)))

        y = model(x_batch)

        gens, gts = [], []
        for j in range(batch_size):
            all_z = []
            for k in range(len(y)):
                pred = y[k][j]
                all_z.append(pred.unsqueeze(0))
            generated = vae.sample_with_z(all_z, no_grad=False)
            generated = torch.permute(generated, perm_dims) * 255
            generated = generated[0, 100:, :, :]
            gens.append(generated)

            gt = Image.open(gt_train_shuffled[i + j])
            gt = np.array(gt.getdata(), dtype=np.uint8).reshape((gt.size[1], gt.size[0], 3))[100:, :, :]
            gt = torch.tensor(gt, dtype=torch.float, requires_grad=True).cuda()
            gts.append(gt)

        gens = torch.stack(gens)
        gts = torch.stack(gts)

        loss = loss_fn(gens, gts)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses)


def test_interp_nn(x_test, gt_test, model, vae, batch_size, loss_fn):
    model.eval()
    vae.eval()
    perm_dims = (0, 2, 3, 1)
    losses = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = None
            for x in x_test[i:i + batch_size]:
                if x_batch is None:
                    x_batch = torch.tensor(prepare_x(x), dtype=torch.float).cuda().unsqueeze(0)
                else:
                    x_batch = torch.vstack((x_batch, torch.tensor(prepare_x(x), dtype=torch.float).cuda().unsqueeze(0)))

            y = model(x_batch)

            gens, gts = [], []
            for j in range(batch_size):
                all_z = []
                for k in range(len(y)):
                    pred = y[k][j]
                    all_z.append(pred.unsqueeze(0))
                generated = vae.sample_with_z(all_z, no_grad=False)
                generated = torch.permute(generated, perm_dims) * 255
                generated = generated[0, 100:, :, :]
                gens.append(generated)

                gt = Image.open(gt_test[i + j])
                gt = np.array(gt.getdata(), dtype=np.uint8).reshape((gt.size[1], gt.size[0], 3))[100:, :, :]
                gt = torch.tensor(gt, dtype=torch.float, requires_grad=True).cuda()
                gts.append(gt)

            gens = torch.stack(gens)
            gts = torch.stack(gts)

            loss = loss_fn(gens, gts)
            losses.append(float(loss.item()))

    return np.mean(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('--latents', type=str, default='/path/to/latents.pkl',
                        help='location of the latents')
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='size of a batch')
    parser.add_argument('--checkpoint_vae', type=str, default="./checkpoints/checkpoint.pt",
                        help='the checkpoint of the VAE model')

    args = parser.parse_args()

    main(args)