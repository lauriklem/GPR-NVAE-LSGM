from gpr_latent_nn_network import InterpNet
import pickle
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.interp_utils import linear_interpolation


def list_vectors(latent_list):
    in1_vectors, in2_vectors, gt_vectors, labels = [], [], [], []
    for row in latent_list:
        in1, in2, gt, label = row
        dims = np.array(in1).shape
        for i in range(dims[1]):
            for j in range(dims[2]):
                in1_vectors.append(in1[:, i, j])
                # print("In1: {}".format(in1[:, i, j]))
                in2_vectors.append(in2[:, i, j])
                gt_vectors.append(gt[:, i, j])
                labels.append(label)

    return in1_vectors, in2_vectors, gt_vectors, labels


def test_lerp(in1_test, in2_test, gt_test, label_test, loss_fn):
    losses = []
    for i in range(len(in1_test)):
        in1 = torch.tensor(in1_test[i], dtype=torch.float)
        in2 = torch.tensor(in2_test[i], dtype=torch.float)
        label = label_test[i]
        gt = torch.tensor(gt_test[i], dtype=torch.float).unsqueeze(0)

        pred = linear_interpolation(in1, in2, label)
        loss = loss_fn(pred, gt)
        losses.append(float(loss.item()))

    return np.sum(losses)


def main(args):
    latent_folder = args.latents
    latent_files = os.listdir(latent_folder)
    n_latents = int(len(latent_files) / 2)

    print("Loading training data...")
    in1_train, in2_train, gt_train, label_train = [], [], [], []
    for latent_index in range(n_latents):
        latent_file = latent_folder + "latents{}.pkl".format(latent_index)
        f = open(latent_file, "rb")
        latents = pickle.load(f)
        f.close()

        in1_vectors, in2_vectors, gt_vectors, labels = list_vectors(latents)

        in1_train.extend(in1_vectors)
        in2_train.extend(in2_vectors)
        gt_train.extend(gt_vectors)
        label_train.extend(labels)

    print("Loading testing data...")
    in1_test, in2_test, gt_test, label_test = [], [], [], []
    for latent_index in range(n_latents):
        latent_file = latent_folder + "latents_test{}.pkl".format(latent_index)
        f = open(latent_file, "rb")
        latents = pickle.load(f)
        f.close()

        in1_vectors, in2_vectors, gt_vectors, labels = list_vectors(latents)

        in1_test.extend(in1_vectors)
        in2_test.extend(in2_vectors)
        gt_test.extend(gt_vectors)
        label_test.extend(labels)

    model = InterpNet(len(in1_train[0]))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = args.epoch
    batch_size = args.batch_size
    loss_fn = nn.L1Loss()

    lerp_loss = test_lerp(in1_test, in2_test, gt_test, label_test, loss_fn)
    print("Loss with linear interpolation: {:.3f}".format(lerp_loss / args.batch_size))

    print("Training...")
    for epoch in range(n_epochs):
        loss_train = train_interp_nn(in1_train, in2_train, gt_train, label_train, model, optimizer, batch_size, loss_fn)
        loss_test = test_interp_nn(in1_test, in2_test, gt_test, label_test, model, batch_size, loss_fn)
        print("Epoch {}: train {:.3f}, test {:.3f}".format(epoch + 1, loss_train, loss_test))

    dst_folder = "./latent_nn/checkpoints"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    torch.save(model.state_dict(), dst_folder + "/checkpoint.pt")


def train_interp_nn(in1_train, in2_train, gt_train, label_train, model, optimizer, batch_size, loss_fn):
    # trans = transforms.Compose([transforms.ToTensor()])
    train_ind = np.array(range(len(in1_train)))
    rng = np.random.default_rng()
    rng.shuffle(train_ind)

    in1_train_shuffled, in2_train_shuffled, gt_train_shuffled, label_train_shuffled = [], [], [], []
    for ind in train_ind:
        in1_train_shuffled.append(in1_train[ind])
        in2_train_shuffled.append(in2_train[ind])
        gt_train_shuffled.append(gt_train[ind])
        label_train_shuffled.append(label_train[ind])

    model.train()
    losses = []
    for i in range(0, len(train_ind), args.batch_size):
        optimizer.zero_grad()
        x_batch, gt_batch = [], []
        for j in range(batch_size):
            x = list(in1_train_shuffled[i + j]) + list(in2_train_shuffled[i + j]) + [label_train_shuffled[i + j]]
            x_batch.append(torch.tensor(x, dtype=torch.float))
            gt_batch.append(torch.tensor(gt_train_shuffled[i + j], dtype=torch.float))

        x_batch = torch.stack(x_batch).cuda()
        gt_batch = torch.stack(gt_batch).cuda()

        pred = model(x_batch)
        loss = loss_fn(pred, gt_batch)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()

    return np.sum(losses)


def test_interp_nn(in1_test, in2_test, gt_test, label_test, model, batch_size, loss_fn):
    # trans = transforms.Compose([transforms.ToTensor()])
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(in1_test), args.batch_size):
            x_batch, gt_batch = [], []
            for j in range(batch_size):
                x = list(in1_test[i + j]) + list(in2_test[i + j]) + [label_test[i + j]]
                x_batch.append(torch.tensor(x, dtype=torch.float))
                gt_batch.append(torch.tensor(gt_test[i + j], dtype=torch.float))

            x_batch = torch.stack(x_batch).cuda()
            gt_batch = torch.stack(gt_batch).cuda()
            pred = model(x_batch)
            loss = loss_fn(pred, gt_batch)
            losses.append(float(loss.item()))

    return np.sum(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('--latents', type=str, default='/path/to/latents.pkl',
                        help='location of the latents')
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of a batch')

    args = parser.parse_args()

    main(args)