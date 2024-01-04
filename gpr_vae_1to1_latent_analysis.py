import argparse
import os
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.multiprocessing import Process
from nvae import NVAE
from plot_settings import plotting_fonts
from util import utils
from PIL import Image
import torchvision.transforms as transforms
from util.interp_utils import linear_interpolation, slerp
import umap
import matplotlib.pyplot as plt
import umap.plot
from gpr_vae_1to1_generate import generate_paths
import pickle


def plot_interp(umap_trans, latents_gt, latents_lerp, latents_slerp, gt_signal, signal, dst_folder, name):
    """
    Plot interpolation "paths"
    """
    colors = ListedColormap(['C0', 'C1', 'C2'])
    text_offset = 0.3

    plt.figure()
    plt.plot(latents_gt[:, 0], latents_gt[:, 1], c="C6", marker="s", mfc='k', mec='k')  # Ground truth path
    plt.plot(latents_lerp[1:-1, 0], latents_lerp[1:-1, 1], c="C6", marker="+", markersize=7, linestyle="dashed",
             mfc='k', mec='k')  # Linear interpolation path
    plt.plot(latents_slerp[1:-1, 0], latents_slerp[1:-1, 1], c="C6", marker="x", linestyle="dotted", mfc='k',
             mec='k')  # Spherical linear interpolation path

    # Connect lerp and slerp paths to left and right images
    plt.plot(latents_lerp[:2, 0], latents_lerp[:2, 1], c="C6", linestyle="dashed")  # Left to 0.2
    plt.plot(latents_lerp[-2:, 0], latents_lerp[-2:, 1], c="C6", linestyle="dashed")  # 0.8 to right
    plt.plot(latents_slerp[:2, 0], latents_slerp[:2, 1], c="C6", linestyle="dotted")
    plt.plot(latents_slerp[-2:, 0], latents_slerp[-2:, 1], c="C6", linestyle="dotted")

    plt.text(latents_gt[0, 0] + text_offset, latents_gt[0, 1], "L", fontsize=14)
    plt.text(latents_gt[-1, 0] + text_offset, latents_gt[-1, 1], "R", fontsize=14)

    xmin, xmax, ymin, ymax = plt.axis()

    # Plot other points
    plt.scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1], c=signal, cmap=colors, alpha=0.3)

    # Plot gt points as squares
    for j in range(len(gt_signal)):
        plt.scatter(latents_gt[j, 0], latents_gt[j, 1], c="C{}".format(gt_signal[j]), marker="s", edgecolor="k",
                    zorder=10)
    plt.legend(["Ground truth", "LERP", "SLERP"])
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.xlim([xmin - text_offset, xmax + text_offset])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.savefig(dst_folder + name)


def get_all_z_interp(vae, input1, input2, label, interp_type, args, latent_index=0):
    """
    Get latent variables z when interpolating between two images.
    """
    trans = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        input1 = Image.open(input1)
        im1 = np.array(input1.getdata(), dtype=np.uint8).reshape((input1.size[1], input1.size[0], 3))
        im1 = trans(Image.fromarray(im1, mode='RGB'))

        input2 = Image.open(input2)
        im2 = np.array(input2.getdata(), dtype=np.uint8).reshape((input2.size[1], input2.size[0], 3))
        im2 = trans(Image.fromarray(im2, mode='RGB'))

        im1 = utils.common_x_operations([im1, im1], args.num_x_bits).unsqueeze(0)
        im2 = utils.common_x_operations([im2, im2], args.num_x_bits).unsqueeze(0)

        all_eps1, all_z1 = vae.calculate_eps_and_z(im1)
        all_eps2, all_z2 = vae.calculate_eps_and_z(im2)

        z1 = all_z1[latent_index]
        z2 = all_z2[latent_index]
        if interp_type == "spherical":
            return slerp(z1, z2, label).cpu().numpy().flatten()
        else:
            return linear_interpolation(z1, z2, label).cpu().numpy().flatten()


def gpr_vae_1to1_get_latents(vae, images_list, args, datadir="", latent_index=0):
    """
    Calculates all latents for given image list.
    """
    trans = transforms.Compose([transforms.ToTensor()])

    latents = []
    with torch.no_grad():
        for im_path in images_list:
            im = Image.open(datadir + im_path)
            im = np.array(im.getdata(), dtype=np.uint8).reshape((im.size[1], im.size[0], 3))
            im = trans(Image.fromarray(im, mode='RGB'))
            im = utils.common_x_operations([im, im], args.num_x_bits).unsqueeze(0)
            all_eps1, all_z1 = vae.calculate_eps_and_z(im)
            latents.append(all_eps1[latent_index].cpu().numpy().flatten())

    return latents


def list_images_signals():
    """
    Lists images in simulated data (folders 18-33) and the corresponding signal strengths.
    0 = no signal, 1 = weak signal, 2 = strong signal.
    Test folders are 27 and 30
    """
    images_list, signal, images_list_test, signal_test = [], [], [], []
    for folder in range(18, 34):  # [27, 30]:
        folder_path = "./datasets/gpr_pics/{}/".format(folder)
        n_images = len(os.listdir(folder_path))
        for i in range(n_images):
            if folder in [27, 30]:
                images_list_test.append(folder_path + "{}.jpg".format(i))

            else:
                images_list.append(folder_path + "{}.jpg".format(i))

            if folder in [18, 19]:
                signal.append(0)
            elif folder == 20:
                if 5 <= i <= 15:
                    signal.append(2)
                elif 4 <= i <= 17:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 21:
                if 11 <= i <= 14:
                    signal.append(2)
                elif 3 <= i <= 10 or 15 <= i <= 17:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 22:
                if i == 0 or i == 20:
                    signal.append(1)
                else:
                    signal.append(2)
            elif folder == 23:
                if i == 0 or i == 20:
                    signal.append(1)
                else:
                    signal.append(2)
            elif folder == 24:
                if 2 <= i <= 18:
                    signal.append(2)
                elif 1 <= i <= 19:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 25:
                if 3 <= i <= 17:
                    signal.append(2)
                elif 1 <= i <= 19:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 26:
                if 9 <= i <= 11:
                    signal.append(2)
                elif 7 <= i <= 13:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 27:
                if 9 <= i <= 11:
                    signal_test.append(2)
                elif 7 <= i <= 13:
                    signal_test.append(1)
                else:
                    signal_test.append(0)
            elif folder == 28:
                if 6 <= i <= 14:
                    signal.append(2)
                elif 4 <= i <= 16:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 29:
                if 6 <= i <= 14:
                    signal.append(2)
                elif 3 <= i <= 17:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 30:
                if 8 <= i <= 12:
                    signal_test.append(2)
                elif 6 <= i <= 14:
                    signal_test.append(1)
                else:
                    signal_test.append(0)
            elif folder == 31:
                if 9 <= i <= 11:
                    signal.append(2)
                elif 5 <= i <= 14:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 32:
                if 9 <= i <= 11:
                    signal.append(1)
                else:
                    signal.append(0)
            elif folder == 33:
                if 9 <= i <= 11:
                    signal.append(1)
                else:
                    signal.append(0)

    return images_list, signal, images_list_test, signal_test


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

    # List all images and the signal labels
    images_list, signal, images_list_test, signal_test = list_images_signals()

    dst_folder = "./results/vae_1to1/"
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
        latents = np.array(gpr_vae_1to1_get_latents(vae, images_list, args, latent_index=latent_index))
        latents_test = np.array(gpr_vae_1to1_get_latents(vae, images_list_test, args, latent_index=latent_index))
        print("Writing to files...")
        f = open(latent_file, "wb")
        pickle.dump(latents, f)
        f.close()
        f = open(latent_file_test, "wb")
        pickle.dump(latents_test, f)
        f.close()

    print("Latents have a shape of {}".format(latents.shape))
    print("Finding UMAP transform...")
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
        for neigh in n_neighbors_list:
            j = 0
            for md in min_dist_ist:
                # Find umap transformation
                umap_trans = umap.UMAP(n_neighbors=neigh, min_dist=md, metric="manhattan", random_state=seed,
                                       transform_seed=seed).fit(latents, y)
                # Transform test data
                test_data_transformed = umap_trans.transform(latents_test)

                # Plot results
                scatter = axs[i, j].scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1], c=signal,
                                            cmap=colors, marker=".")
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
    """
    # One set of parameters
    for seed in rseed:
        if y is None:
            umap_trans = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=seed,
                                   transform_seed=seed, n_components=2).fit(latents)
            f = open(dst_folder + "umap.pkl", "wb")
        else:
            umap_trans = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="manhattan", random_state=seed,
                                   transform_seed=seed, n_components=2).fit(latents, y=signal)
            f = open(dst_folder + "umap_target.pkl", "wb")

        pickle.dump(umap_trans, f)
        f.close()
        test_data_transformed = umap_trans.transform(latents_test)

        # 3-D plot
        if len(umap_trans.embedding_[0, :]) >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1], umap_trans.embedding_[:, 1],
                                 c=signal, cmap=colors, marker=".")
            ax.scatter(test_data_transformed[:, 0], test_data_transformed[:, 1], test_data_transformed[:, 2],
                       c=signal_test, cmap=colors, marker="^", edgecolor="k", linewidths=0.5)
            plt.legend(handles=scatter.legend_elements()[0], labels=["No signal", "Weak signal", "Strong signal"])
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            plt.tight_layout()
            if y is None:
                plt.savefig(dst_folder + "umap_result_3d_{}.pdf".format(latent_index))
            else:
                plt.savefig(dst_folder + "umap_result_target_3d_{}.pdf".format(latent_index))
        else:
            # 2-D plot
            plt.figure()

            scatter = plt.scatter(umap_trans.embedding_[:, 0], umap_trans.embedding_[:, 1],
                                  c=signal, cmap=colors, marker=".")
            plt.scatter(test_data_transformed[:, 0], test_data_transformed[:, 1],
                        c=signal_test, cmap=colors, marker="^", edgecolor="k", linewidths=0.5)
            plt.legend(handles=scatter.legend_elements()[0], labels=["No signal", "Weak signal", "Strong signal"])
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            plt.tight_layout()
            if y is None:
                plt.savefig(dst_folder + "umap_result_{}.pdf".format(latent_index))
            else:
                plt.savefig(dst_folder + "umap_result_target_{}.pdf".format(latent_index))
        plt.show()

    # Plot interpolation paths
    if latent_index == 0:
        if y is None:
            f = open(dst_folder + "umap.pkl", "rb")
        else:
            f = open(dst_folder + "umap_target.pkl", "rb")
        umap_trans = pickle.load(f)
        f.close()

        left_images = [0, 5, 7, 9] + [5, 7, 9, 15] + [3, 11, 12] + [2, 3, 12, 13]
        folders = [27, 27, 27, 27] + [30, 30, 30, 30] + [27, 27, 27] + [30, 30, 30, 30]

        datadir = "./datasets/gpr_pics"

        in1_all, in2_all, gt_all = generate_paths(left_images, folders, [4])
        in1_paths = in1_all[0]
        in2_paths = in2_all[0]
        gt_paths = gt_all[0]

        if y is None:
            dst_folder += "umap_notarget/"
        else:
            dst_folder += "umap_target/"
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Manually labeled gt signal strengths
        gt_signals = [
            [0, 0, 1, 1, 2, 2],
            [1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 1, 0],
            [1, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [2, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 2],
            [2, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]
        ]
        # Index of images
        case_ind = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]
        for i in range(len(case_ind)):
            gt_signal = gt_signals[i]
            c_ind = case_ind[i]

            input1 = datadir + in1_paths[c_ind]  # List of left images
            input2 = datadir + in2_paths[c_ind]  # List of right images
            temp_gt = gt_paths[c_ind]  # List of ground truths
            labels = np.array(range(1, 4 + 1)) / float(4 + 1)  # List of labels

            # Calculate interpolated latents
            latents_lerp, latents_slerp = [], []
            for label in labels:
                latents_slerp.append(
                    get_all_z_interp(vae, input1, input2, label, "spherical", args, latent_index=latent_index))
                latents_lerp.append(
                    get_all_z_interp(vae, input1, input2, label, "linear", args, latent_index=latent_index))

            # Transform latents
            latents_lerp = umap_trans.transform(np.array(latents_lerp))
            latents_slerp = umap_trans.transform(np.array(latents_slerp))
            gt_interp = umap_trans.transform(
                np.array(gpr_vae_1to1_get_latents(vae, temp_gt, args, datadir=datadir, latent_index=latent_index)))
            left = umap_trans.transform(
                np.array(gpr_vae_1to1_get_latents(vae, [input1], args, latent_index=latent_index)))
            right = umap_trans.transform(
                np.array(gpr_vae_1to1_get_latents(vae, [input2], args, latent_index=latent_index)))

            gt_interp = np.concatenate((left, gt_interp, right))
            latents_lerp = np.concatenate((left, latents_lerp, right))
            latents_slerp = np.concatenate((left, latents_slerp, right))

            plot_interp(umap_trans, gt_interp, latents_lerp, latents_slerp, gt_signal, signal, dst_folder,
                        name=str(c_ind) + ".pdf")
    """

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
