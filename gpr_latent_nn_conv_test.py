import os
import argparse
import pickle
import torch

import gpr_evaluation
from gpr_dataset import GPRDataset
from util import utils, interp_utils
from nvae import NVAE
from torchvision import transforms
from PIL import Image
import numpy as np
from gpr_latent_nn_network_conv import InterpNetConv
import matplotlib.pyplot as plt


def interp_nn(vae, interp_net_list, input1, input2, label, args):
    """
    Generate interpolated image between given inputs using latent space network.
    """
    trans = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        input1 = Image.open(input1)
        im1 = np.array(input1.getdata(), dtype=np.uint8).reshape((input1.size[1], input1.size[0], 3))
        im1 = trans(Image.fromarray(im1, mode='RGB'))
        im1 = utils.common_x_operations([im1, im1], args.num_x_bits).unsqueeze(0)

        input2 = Image.open(input2)
        im2 = np.array(input2.getdata(), dtype=np.uint8).reshape((input2.size[1], input2.size[0], 3))
        im2 = trans(Image.fromarray(im2, mode='RGB'))
        im2 = utils.common_x_operations([im2, im2], args.num_x_bits).unsqueeze(0)

        all_eps1, all_z1 = vae.calculate_eps_and_z(im1)
        all_eps2, all_z2 = vae.calculate_eps_and_z(im2)
        all_z = []

        # fig, ax = plt.subplots(len(all_eps1), 3)

        for i in range(len(all_eps1)):
            z1 = all_z1[i].cpu().numpy()[0, :, :, :]
            z2 = all_z2[i].cpu().numpy()[0, :, :, :]

            dims = z1.shape
            label_img = np.full((dims[1], dims[2]), label)

            x_temp = []
            for m in range(dims[0]):
                x_temp.extend(np.stack((z1[m], z2[m], label_img)))

            x = torch.tensor(np.array(x_temp), dtype=torch.float).cuda().unsqueeze(0)
            z_out = interp_net_list[i](x)
            all_z.append(z_out)
            """
            ax[i, 0].imshow(z1[0, :, :])
            ax[i, 0].axis("off")
            ax[i, 1].imshow(z2[0, :, :])
            ax[i, 1].axis("off")
            ax[i, 2].imshow(z_out.cpu().numpy()[0, :, :, :][0, :, :])
            ax[i, 2].axis("off")
            """
        # plt.show()
        # print(all_z[0][0][0][0][0].item())
        # print(all_z[1][0][0][0][0].item())

        generated = vae.sample_with_z(all_z)
        perm_dims = (0, 2, 3, 1)
        generated = torch.permute(generated, perm_dims).cpu().numpy() * 255
        generated = np.round(generated[0], 0).astype("uint8")
        return generated


def latent_nn_test(vae, interp_net_list, test_data, vae_args):
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for row in test_data:
            input1, input2, gt, label = row

            generated = interp_nn(vae, interp_net_list, input1, input2, label, vae_args)

            generated = generated[100:, :, :]
            generated = Image.fromarray(generated)

            gt = Image.open(gt)
            gt = np.array(gt.getdata(), dtype=np.uint8).reshape((gt.size[1], gt.size[0], 3))[100:, :, :]
            gt = Image.fromarray(gt)

            """
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(generated)
            ax[1].imshow(gt)
            plt.show()
            """

            mae, ssim, psnr = gpr_evaluation.calculate_all(generated, gt)
            mae_list.append(mae)
            ssim_list.append(ssim)
            psnr_list.append(psnr)

        return mae_list, ssim_list, psnr_list


def main(args):
    trans = transforms.Compose([transforms.ToTensor()])

    checkpoint_vae = torch.load(args.checkpoint_vae, map_location='cpu')
    vae_args = checkpoint_vae['args']
    arch_instance_nvae = utils.get_arch_cells(vae_args.arch_instance, vae_args.use_se)
    vae = NVAE(vae_args, arch_instance_nvae)
    vae.load_state_dict(checkpoint_vae['vae_state_dict'])
    vae = vae.cuda()
    vae.eval()

    ds = GPRDataset(stdev=0, datadir="./datasets/gpr_pics", images_between=[4],
                    verbose=True)

    with torch.no_grad():
        input1, _, _, _ = ds.normal_interp_part[0]
        input1 = Image.open(input1)
        im1 = np.array(input1.getdata(), dtype=np.uint8).reshape((input1.size[1], input1.size[0], 3))
        im1 = trans(Image.fromarray(im1, mode='RGB'))
        im1 = utils.common_x_operations([im1, im1], vae_args.num_x_bits).unsqueeze(0)
        all_eps1, all_z1 = vae.calculate_eps_and_z(im1)
        dims = list(all_z1[0][0, :, :, :].size())

    interp_net0 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net0.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(0)))
    interp_net0.eval()
    interp_net0.cuda()

    interp_net1 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net1.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(1)))
    interp_net1.eval()
    interp_net1.cuda()

    interp_net2 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net2.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(2)))
    interp_net2.eval()
    interp_net2.cuda()

    interp_net3 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net3.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(3)))
    interp_net3.eval()
    interp_net3.cuda()

    interp_net4 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net4.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(4)))
    interp_net4.eval()
    interp_net4.cuda()

    interp_net5 = InterpNetConv((int(3 * dims[0]), dims[1], dims[2]))
    interp_net5.load_state_dict(torch.load(args.checkpoint_nn + "checkpoint_conv{}.pt".format(5)))
    interp_net5.eval()
    interp_net5.cuda()

    interp_net_list = torch.nn.ModuleList((
        interp_net0,
        interp_net1,
        interp_net2,
        interp_net3,
        interp_net4,
        interp_net5,
    ))

    mae_normal, ssim_normal, psnr_normal = latent_nn_test(vae, interp_net_list, ds.normal_interp_part, vae_args)
    mae_extr, ssim_extr, psnr_extr = latent_nn_test(vae, interp_net_list, ds.extrapolation_part, vae_args)

    gpr_evaluation.print_extrapolation(mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('--checkpoint_nn', type=str, default="./latent_nn/checkpoints/checkpoint.pt",
                        help='the checkpoint of the interpolating network')
    parser.add_argument('--checkpoint_vae', type=str, default="./checkpoints/checkpoint.pt",
                        help='the checkpoint of the VAE model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of a batch')
    parser.add_argument('--dataset', type=str, default='gpr', help='which dataset to use')

    args = parser.parse_args()

    main(args)