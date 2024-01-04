
import argparse
import torch
import numpy as np
import os
from torch.multiprocessing import Process
from nvae import NVAE
from diffusion_discretized import DiffusionDiscretized
from diffusion_continuous import make_diffusion
import torchvision.transforms as transforms
from util.interp_utils import linear_interpolation, slerp

try:
    from apex.optimizers import FusedAdam
except ImportError:
    print("No Apex Available. Using PyTorch's native Adam. Install Apex for faster training.")
    from torch.optim import Adam as FusedAdam
from util import utils, datasets
from util.ema import EMA
from PIL import Image
from gpr_generate_utils import generate_paths
from gpr_diff_images import difference_images

"""
Generate images with LSGM model
"""


def main(eval_args):
    # common initialization
    logging, writer = utils.common_init(eval_args.global_rank, eval_args.seed, eval_args.save)

    # load a checkpoint
    logging.info('#' * 80)
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    # adding some arguments for backward compatibility.
    if not hasattr(args, 'num_x_bits'):
        logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(args, 'num_x_bits', 8)

    if not hasattr(args, 'channel_mult'):
        logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(args, 'channel_mult', [1, 2])

    if not hasattr(args, 'mixing_logit_init'):
        logging.info('*** Setting %s manually ****', 'mixing_logit_init')
        setattr(args, 'mixing_logit_init', -3.0)

    if eval_args.diffusion_steps > 0:
        args.diffusion_steps = eval_args.diffusion_steps

    logging.info('loaded the model at epoch %d', checkpoint['epoch'])

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

    shape = [dae.num_input_channels, dae.input_size, dae.input_size]

    trans = transforms.Compose([transforms.ToTensor()])
    batchsize = args.batch_size

    images_between = [4]
    # Left images and folders that will be generated
    left_images = [0, 5, 7, 9] + [5, 7, 9, 15] + [3, 11, 12] + [2, 3, 12, 13]
    folders = [27, 27, 27, 27] + [30, 30, 30, 30] + [27, 27, 27] + [30, 30, 30, 30]

    datadir = "./datasets/gpr_pics"

    dst_folder = "./results/lsgm_1to1/"
    dst_folder += eval_args.checkpoint.split("./checkpoints/")[-1].split("/lsgm")[0] + "/"
    interp_type = eval_args.interp_type
    dst_folder += interp_type + "/"

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    in1_all, in2_all, gt_all = generate_paths(left_images, folders, images_between)
    print("Generating images...")
    with torch.no_grad():
        for ib in images_between:
            labels = np.array(range(1, ib + 1)) / float(ib + 1)
            for i in range(len(in1_all)):
                in1_paths = in1_all[i]
                in2_paths = in2_all[i]
                gt_paths = gt_all[i]

                for j in range(len(in1_paths)):
                    print(j)
                    im1_path = datadir + in1_paths[j]
                    im2_path = datadir + in2_paths[j]
                    temp_gt = gt_paths[j]

                    for label_ind in range(len(temp_gt)):
                        gt = Image.open(datadir + temp_gt[label_ind])
                        gt.save(dst_folder + str(j) + "_gt_{:.2f}_ib_{}.png".format(labels[label_ind], ib))

                    input1 = Image.open(im1_path)
                    im1 = np.array(input1.getdata(), dtype=np.uint8).reshape((input1.size[1], input1.size[0], 3))
                    im1 = trans(Image.fromarray(im1, mode='RGB'))

                    input2 = Image.open(im2_path)
                    im2 = np.array(input2.getdata(), dtype=np.uint8).reshape((input2.size[1], input2.size[0], 3))
                    im2 = trans(Image.fromarray(im2, mode='RGB'))

                    left_path = dst_folder + str(j) + "_left_ib_{}.png".format(ib)
                    right_path = dst_folder + str(j) + "_right_ib_{}.png".format(ib)
                    input1.save(left_path)
                    input2.save(right_path)

                    im1 = utils.common_x_operations([im1, im1], args.num_x_bits).unsqueeze(0)
                    im2 = utils.common_x_operations([im2, im2], args.num_x_bits).unsqueeze(0)

                    # Feed inputs through the NVAE encoder
                    logits1, all_log_q1, all_eps1 = vae(im1)
                    logits2, all_log_q2, all_eps2 = vae(im2)

                    # Get top latents
                    eps1 = vae.concat_eps_per_scale(all_eps1)[0]
                    eps2 = vae.concat_eps_per_scale(all_eps2)[0]

                    # Forward diffusion for both inputs
                    noise1 = diffusion_cont.reverse_generative_ode(dae=dae,
                                                                   eps=eps1,
                                                                   ode_eps=eval_args.ode_eps,
                                                                   ode_solver_tol=eval_args.ode_solver_tol,
                                                                   enable_autocast=args.autocast_eval,
                                                                   no_autograd=args.no_autograd_jvp)

                    noise2 = diffusion_cont.reverse_generative_ode(dae=dae,
                                                                   eps=eps2,
                                                                   ode_eps=eval_args.ode_eps,
                                                                   ode_solver_tol=eval_args.ode_solver_tol,
                                                                   enable_autocast=args.autocast_eval,
                                                                   no_autograd=args.no_autograd_jvp)

                    # Interpolate in the latent space of the SGM
                    for label in labels:
                        if interp_type == "spherical":
                            noise = slerp(noise1[0], noise2[0], label)
                        else:
                            noise = linear_interpolation(noise1[0], noise2[0], label)

                        # Reverse diffusion
                        eps, nfe, time_ode_solve = diffusion_cont.sample_model_ode(dae=dae,
                                                                                   num_samples=1,
                                                                                   shape=shape,
                                                                                   ode_eps=eval_args.ode_eps,
                                                                                   ode_solver_tol=eval_args.ode_solver_tol,
                                                                                   enable_autocast=args.autocast_eval,
                                                                                   temp=1.0,
                                                                                   noise=noise)

                        decomposed_eps = vae.decompose_eps(eps)
                        # Use NVAE decoder to get the final image
                        generated = vae.sample(1, 1., decomposed_eps, args.autocast_eval)

                        perm_dims = (0, 2, 3, 1)
                        generated = torch.permute(generated, perm_dims).cpu().numpy() * 255

                        generated = np.round(generated[0], 0).astype("uint8")
                        generated = Image.fromarray(generated)
                        im_path = dst_folder + str(j) + "_gen_{:.2f}_ib_{}".format(label, ib) + ".png"
                        generated.save(im_path)

    print("Calculating difference images...")
    difference_images(len(left_images), images_between, dst_folder)


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
