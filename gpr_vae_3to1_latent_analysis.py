import argparse
import os

import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.multiprocessing import Process

from gpr_generate_utils import generate_paths
from nvae_3to1 import NVAE
from plot_settings import plotting_fonts
from util import utils
from PIL import Image
import torchvision.transforms as transforms
import umap
import matplotlib.pyplot as plt
import umap.plot
import pickle
from gpr_vae_1to1_latent_analysis import list_images_signals, plot_interp


def list_data_3to1(datadir, folders):
    data = []
    labels = np.array(range(1, 5)) / float(5)
    dir1 = datadir + "/" + str(folders[0])
    n_images = len(os.listdir(dir1))

    for folder in folders:
        for survey_line in range(0, n_images - 4 - 1):
            image_dir = datadir + "/{}/".format(folder)
            input1 = image_dir + str(survey_line) + ".jpg"
            input2 = image_dir + str(survey_line + 5) + ".jpg"

            for label_idx in range(4):
                label = labels[label_idx]
                gt = image_dir + str(survey_line + label_idx + 1) + ".jpg"
                data.append([input1, input2, gt, label])

    return data


def gpr_vae_3to1_get_latents(vae, data, latent_index=0):
    trans = transforms.Compose([transforms.ToTensor()])

    latents = []
    with torch.no_grad():
        for row in data:
            input1, input2, _, label = row
            im1 = Image.open(input1)
            lblimg = np.full(fill_value=label * 2 - 1, shape=(im1.size[1], im1.size[0], 1), dtype="float32")
            im1 = np.array(im1.getdata(), dtype="float32").reshape((im1.size[1], im1.size[0], 3)) / 128.0 - 1

            im2 = Image.open(input2)
            im2 = np.array(im2.getdata(), dtype="float32").reshape((im2.size[1], im2.size[0], 3)) / 128.0 - 1
            x = np.concatenate((im1, im2, lblimg), axis=-1)
            x = trans(x).unsqueeze(0)
            x = utils.common_x_operations_cond(x)

            all_eps1, all_z1 = vae.calculate_eps_and_z(x)
            latents.append(all_z1[latent_index].cpu().numpy().flatten())

    return latents


def main(eval_args):
    # common initialization
    logging, writer = utils.common_init(eval_args.global_rank, eval_args.seed, eval_args.save)

    # load a checkpoint
    logging.info('#' * 80)
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    logging.info('loaded the model at epoch %d.', checkpoint['epoch'])

    if not hasattr(args, 'num_x_bits'):
        # logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(args, 'num_x_bits', 8)

    if not hasattr(args, 'channel_mult'):
        # logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(args, 'channel_mult', [1, 2])

    # logging.info('loaded the model at epoch %d', checkpoint['epoch'])

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    # logging.info('args = %s', args)

    # load VAE
    vae = NVAE(args, arch_instance_nvae)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    # logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))

    # replace a few fields in args based on eval_args
    # this will allow train/evaluate on different systems
    args.num_proc_node = eval_args.num_proc_node
    args.num_process_per_node = eval_args.num_process_per_node
    args.data = eval_args.data
    if eval_args.batch_size > 0:
        args.batch_size = eval_args.batch_size

    vae.eval()
    sim_train, sim_test = [], [27, 30]
    for folder in range(18, 34):
        if folder not in sim_test:
            sim_train.append(folder)

    train_ind = list_data_3to1(datadir="./datasets/gpr_pics", folders=sim_train)
    valid_ind = list_data_3to1(datadir="./datasets/gpr_pics", folders=sim_test)

    _, temp_signal, _, temp_signal_test = list_images_signals()

    dir1 = "./datasets/gpr_pics/18"
    n_images = len(os.listdir(dir1))

    signal = []
    for i in range(len(sim_train)):
        offset = i * n_images
        for j in range(1, n_images - 4):
            signal.extend(temp_signal[offset + j:offset + j + 4])

    signal_test = []
    for i in range(len(sim_test)):
        offset = i * n_images
        for j in range(1, n_images - 4):
            signal_test.extend(temp_signal_test[offset + j:offset + j + 4])

    dst_folder = "./results/vae_3to1/"
    dst_folder += eval_args.checkpoint.split("./checkpoints/")[-1].split("/vae")[0] + "/"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    latent_index = eval_args.latent_index
    latent_file = dst_folder + "latents{}.pkl".format(latent_index)
    latent_file_test = dst_folder + "latents_test{}.pkl".format(latent_index)

    if os.path.isfile(latent_file):
        print("Fetching latents from files...")
        f = open(latent_file, "rb")
        latents = pickle.load(f)
        f.close()
        f = open(latent_file_test, "rb")
        latents_test = pickle.load(f)
        f.close()
    else:
        print("Calculating latents...")
        latents = np.array(gpr_vae_3to1_get_latents(vae, train_ind, latent_index))
        latents_test = np.array(gpr_vae_3to1_get_latents(vae, valid_ind, latent_index))
        print("Writing to files...")
        f = open(latent_file, "wb")
        pickle.dump(latents, f)
        f.close()
        f = open(latent_file_test, "wb")
        pickle.dump(latents_test, f)
        f.close()

    print("Latents have a shape of {}".format(latents.shape))
    plotting_fonts()  # Set plotting fonts
    colors = ListedColormap(['C0', 'C1', 'C2'])

    if eval_args.use_labels:
        print("Using label information")
        y = signal
    else:
        print("Not using label information")
        y = None

    # Random seeds to use
    # rseed = [103, 102, 101]
    rseed = [101]

    # Several parameters
    for seed in rseed:
        n_neighbors_list = [10, 15, 20]
        min_dist_ist = [0.05, 0.1, 0.2]

        fig, axs = plt.subplots(len(n_neighbors_list), len(min_dist_ist), figsize=(12.8, 9.6))
        i = 0
        print("Finding UMAP transform...")
        for neigh in n_neighbors_list:
            j = 0
            for md in min_dist_ist:
                # Find umap transformation
                umap_trans = umap.UMAP(n_neighbors=neigh, min_dist=md, metric="manhattan", random_state=seed,
                                       transform_seed=seed).fit(latents, y)
                # Transform test data
                test_data_transformed = umap_trans.transform(latents_test)

                # Plot results
                scatter = axs[i, j].scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1], c=signal, cmap=colors,
                                            marker=".")
                axs[i, j].scatter(test_data_transformed[:, 0], test_data_transformed[:, 1], c=signal_test, cmap=colors,
                                  marker="^", edgecolor="k", linewidths=0.5)
                axs[i, j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                if i == 0:
                    axs[i, j].title.set_text("min-dist {}".format(md))
                if j == 0:
                    axs[i, j].set_ylabel("n_neighbors {}".format(neigh))
                j += 1
            i += 1
        plt.tight_layout()
        if y is None:
            plt.savefig(dst_folder + "umap_grid_{}.pdf".format(latent_index))
        else:
            plt.savefig(dst_folder + "umap_grid_target_{}.pdf".format(latent_index))
        plt.show()

    # One set of parameters
    for seed in rseed:
        plt.figure()
        if y is None:
            umap_trans = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=101,
                                   transform_seed=seed).fit(
                latents)
            f = open(dst_folder + "umap.pkl", "wb")
        else:
            umap_trans = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=101,
                                   transform_seed=seed).fit(
                latents, y=signal)
            f = open(dst_folder + "umap_target.pkl", "wb")

        pickle.dump(umap_trans, f)
        f.close()
        test_data_transformed = umap_trans.transform(latents_test)
        scatter = plt.scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1], c=signal, cmap=colors,
                              marker=".")
        plt.scatter(test_data_transformed[:, 0], test_data_transformed[:, 1], c=signal_test, cmap=colors, marker="^",
                    edgecolor="k", linewidths=0.5)
        plt.legend(handles=scatter.legend_elements()[0], labels=["No signal", "Weak signal", "Strong signal"])
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.tight_layout()
        if y is None:
            plt.savefig(dst_folder + "umap_result_{}.pdf".format(latent_index))
        else:
            plt.savefig(dst_folder + "umap_result_target_{}.pdf".format(latent_index))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    # directories for experiment results and checkpoint
    parser.add_argument('--checkpoint', type=str, default='/path/to/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--root', type=str, default='/tmp/nvae-diff/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug_ode',
                        help='id used for storing intermediate results')
    parser.add_argument('--eval_mode', type=str, default='evaluate', choices=['sample', 'evaluate'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--nll_eval', action='store_true', default=False,
                        help='if True, we perform NLL evaluation.')
    parser.add_argument('--fid_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='A dir to store fid related files')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='The temperature used for sampling.')
    parser.add_argument('--num_iw_samples', type=int, default=1,
                        help='The number of samples from latent space used in IW evaluation.')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    # DDP.
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')

    # Lauri's additions
    parser.add_argument('--interp_type', type=str, default='linear', choices=['linear', 'spherical'],
                        help='Interpolation type for interpolating latents: Linear interpolation '
                             'or spherical linear interpolation')
    parser.add_argument('--latent_index', type=int, default=-1,
                        help='The index of latent variables that will be used for the analysis')
    parser.add_argument('--use_labels', action='store_true', default=False,
                        help='if given, use target information in UMAP.')

    args = parser.parse_args()
    args.save = args.root + '/' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        utils.init_processes(0, size, main, args)
