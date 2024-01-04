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
import umap
import matplotlib.pyplot as plt
import umap.plot
import pickle
from gpr_vae_1to1_latent_analysis import list_images_signals
from diffusion_continuous import make_diffusion
from torch.optim import Adam as FusedAdam
from util.ema import EMA

"""
Analyze the latent space of the LSGM model
"""


def gpr_lsgm_1to1_get_latents(vae, dae, images_list, args, eval_args, diffusion_cont, datadir=""):
    """
    Calculate latents for the given image list
    """
    trans = transforms.Compose([transforms.ToTensor()])

    latents = []
    with torch.no_grad():
        for im_path in images_list:
            im = Image.open(datadir + im_path)
            im = np.array(im.getdata(), dtype=np.uint8).reshape((im.size[1], im.size[0], 3))
            im = trans(Image.fromarray(im, mode='RGB'))
            im = utils.common_x_operations([im, im], args.num_x_bits).unsqueeze(0)

            # Encode with NVAE encoder
            logits, all_log_q, all_eps = vae(im)
            eps = vae.concat_eps_per_scale(all_eps)[0]

            # Forward diffusion
            noise = diffusion_cont.reverse_generative_ode(dae=dae,
                                                          eps=eps,
                                                          ode_eps=eval_args.ode_eps,
                                                          ode_solver_tol=eval_args.ode_solver_tol,
                                                          enable_autocast=args.autocast_eval,
                                                          no_autograd=args.no_autograd_jvp)

            latents.append(noise.cpu().numpy().flatten())
            print("{}/{}".format(len(latents), len(images_list)))
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

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)

    # load VAE
    vae = NVAE(args, arch_instance_nvae)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))

    # load DAE
    num_input_channels = vae.latent_structure()[0]
    dae = utils.get_dae_model(args, num_input_channels)
    dae.load_state_dict(checkpoint['dae_state_dict'])
    diffusion_cont = make_diffusion(args)

    logging.info('DAE: param size = %fM ', utils.count_parameters_in_M(dae))
    checkpoint_name = os.path.basename(eval_args.checkpoint)
    if checkpoint_name == 'checkpoint.pt':
        logging.info('Swapping the parameters of DAE with EMA parameters')
        # checkpoint.pt models require swapping EMA parameters
        dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae,
                                  weight_decay=args.weight_decay, eps=1e-4)
        # add EMA functionality to the optimizer
        dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])

        # replace DAE parameters with EMA values
        dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    elif checkpoint_name in {'checkpoint_fid.pt', 'checkpoint_nll.pt', 'checkpoint_finetuned.pt', 'checkpoint_ssim.pt'}:
        logging.info('swapping the parameters of DAE with EMA parameters is ** not ** required.')
    else:
        raise ValueError('Cannot recognize checkpoint name %s' % checkpoint_name)
    dae = dae.cuda()

    # set the model to eval() model.
    dae.eval()
    # set vae to train mode if the arg says
    vae.eval()

    # List images and the respective labels
    images_list, signal, images_list_test, signal_test = list_images_signals()

    dst_folder = "./results/lsgm_1to1/"
    dst_folder += eval_args.checkpoint.split("./checkpoints/")[-1].split("/lsgm")[0] + "/"
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
        latents = np.array(gpr_lsgm_1to1_get_latents(vae=vae,
                                                     dae=dae,
                                                     images_list=images_list,
                                                     args=args,
                                                     eval_args=eval_args,
                                                     diffusion_cont=diffusion_cont))
        latents_test = np.array(gpr_lsgm_1to1_get_latents(vae=vae,
                                                          dae=dae,
                                                          images_list=images_list_test,
                                                          args=args,
                                                          eval_args=eval_args,
                                                          diffusion_cont=diffusion_cont))
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

    # UMAP with several parameters
    for seed in rseed:
        n_neighbors_list = [10, 15, 20]
        min_dist_ist = [0.05, 0.1, 0.2]

        fig, axs = plt.subplots(len(n_neighbors_list), len(min_dist_ist), figsize=(12.8, 9.6))
        i = 0
        for neigh in n_neighbors_list:
            j = 0
            for md in min_dist_ist:
                # Find transformation and transform test data with it
                umap_trans = umap.UMAP(n_neighbors=neigh, min_dist=md, metric="manhattan", random_state=seed,
                                       transform_seed=seed).fit(latents, y)
                test_data_transformed = umap_trans.transform(latents_test)

                # Plot data
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

    # UMAP with one set of parameters
    for seed in rseed:
        if y is None:
            umap_trans = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="manhattan", random_state=seed,
                                   transform_seed=seed, n_components=2).fit(latents)
            f = open(dst_folder + "umap.pkl", "wb")
        else:
            umap_trans = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="manhattan", random_state=seed,
                                   transform_seed=seed, n_components=2).fit(latents, y=signal)
            f = open(dst_folder + "umap_target.pkl", "wb")

        pickle.dump(umap_trans, f)
        f.close()
        test_data_transformed = umap_trans.transform(latents_test)

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
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='The temperature used for sampling.')
    parser.add_argument('--vae_temp', type=float, default=1.0,
                        help='The temperature used for sampling in vae.')
    parser.add_argument('--vae_train_mode', action='store_true', default=False,
                        help='evaluate vae in train mode, suitable for BN experiments.')
    parser.add_argument('--num_iw_samples', type=int, default=1,
                        help='The number of samples from latent space used in IW evaluation.')
    parser.add_argument('--num_iw_inner_samples', type=int, default=1,
                        help='How often we solve the ODE and average when calculating prior probability.')
    parser.add_argument('--num_fid_samples', type=int, default=50000,
                        help='The number of samples used for FID computation.')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    parser.add_argument('--elbo_eval', action='store_true', default=False,
                        help='if True, we perform discrete ELBO evaluation.')
    parser.add_argument('--fid_disc_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation.')
    parser.add_argument('--fid_ode_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation using ODE-based model samples.')
    parser.add_argument('--nll_ode_eval', action='store_true', default=False,
                        help='if True, we perform ODE-based NLL evaluation.')
    parser.add_argument('--nfe_eval', action='store_true', default=False,
                        help='if True, we sample 50 batches of images and average NFEs.')
    parser.add_argument('--ode_sampling', action='store_true', default=False,
                        help='if True, do ODE-based sampling, otherwise regular sampling. Only relevant when sampling.')
    parser.add_argument('--ode_eps', type=float, default=0.00001,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--diffusion_steps', type=int, default=0,
                        help='number of diffusion steps')
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
    parser.add_argument('--latent_index', type=int, default=0,
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
