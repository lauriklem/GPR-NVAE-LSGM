import argparse
import numpy as np
import torch
from torch.multiprocessing import Process
from nvae import NVAE
from util import utils
import gpr_evaluation
from PIL import Image
import torchvision.transforms as transforms
from gpr_dataset import GPRDataset
from util.interp_utils import linear_interpolation, slerp
import matplotlib.pyplot as plt


"""
Test 1-to-1 NVAE model
"""


def vae_recon_test(vae, valid_queue, args):
    trans = transforms.ToPILImage()
    vae.eval()
    mae_list, ssim_list, psnr_list = [], [], []
    for step, x in enumerate(valid_queue):
        x_batch = utils.common_x_operations(x, args.num_x_bits)
        with torch.no_grad():
            if len(x_batch.size()) > 3:
                for i in range(x_batch.size(dim=0)):
                    x = x_batch[i, :, :, :].unsqueeze(0)
                    all_eps, all_z = vae.calculate_eps_and_z(x)
                    generated = vae.sample_with_z(all_z)
                    perm_dims = (0, 2, 3, 1)
                    generated = torch.permute(generated, perm_dims).cpu().numpy() * 255
                    generated = np.round(generated[0], 0).astype("uint8")
                    generated = Image.fromarray(generated)

                    gt = trans(utils.unsymmetrize_image_data(x_batch[i, :, :, :]))
                    """
                    plt.figure()
                    plt.imshow(generated)
                    plt.figure()
                    plt.imshow(gt)
                    plt.show()
                    """
                    mae, ssim, psnr = gpr_evaluation.calculate_all(generated, gt)
                    mae_list.append(mae)
                    ssim_list.append(ssim)
                    psnr_list.append(psnr)
            else:
                all_eps, all_z = vae.calculate_eps_and_z(x_batch)
                generated = vae.sample_with_z(all_z)
                perm_dims = (0, 2, 3, 1)
                generated = torch.permute(generated, perm_dims).cpu().numpy() * 255
                generated = np.round(generated[0], 0).astype("uint8")
                generated = Image.fromarray(generated)

                gt = trans(x_batch)

                mae, ssim, psnr = gpr_evaluation.calculate_all(generated, gt)
                mae_list.append(mae)
                ssim_list.append(ssim)
                psnr_list.append(psnr)

    return mae_list, ssim_list, psnr_list


def interp_1to1(vae, input1, input2, label, interp_type, args):
    """
    Generate interpolated image between given inputs with 1-to-1 NVAE.
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
        all_z = []
        for i in range(len(all_eps1)):
            z1 = all_eps1[i]
            z2 = all_eps2[i]
            if interp_type == "spherical":
                all_z.append(slerp(z1[0], z2[0], label))
            else:
                all_z.append(linear_interpolation(z1[0], z2[0], label))

        generated = vae.sample(1, 1.0, eps_z=all_z)

        perm_dims = (0, 2, 3, 1)
        generated = torch.permute(generated, perm_dims).cpu().numpy() * 255

        generated = np.round(generated[0], 0).astype("uint8")

    return generated


def gpr_vae_1to1_test(vae, test_data, interp_type, args):
    """
    Calculate metrics on the given test data for given 1-to-1 NVAE model.
    """
    mae_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for row in test_data:
            input1, input2, gt, label = row
            gt = Image.open(gt)

            generated = interp_1to1(vae, input1, input2, label, interp_type, args)

            generated = generated[100:, :, :]
            generated = Image.fromarray(generated)

            gt = np.array(gt.getdata(), dtype=np.uint8).reshape((gt.size[1], gt.size[0], 3))[100:, :, :]
            gt = Image.fromarray(gt)

            mae, ssim, psnr = gpr_evaluation.calculate_all(generated, gt)
            mae_list.append(mae)
            ssim_list.append(ssim)
            psnr_list.append(psnr)

    return mae_list, ssim_list, psnr_list


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

    ds = GPRDataset(stdev=0, datadir="./datasets/gpr_pics", images_between=[4],
                    verbose=True)

    interp_type = eval_args.interp_type

    # Normal interpolation cases
    mae_normal, ssim_normal, psnr_normal = gpr_vae_1to1_test(vae=vae,
                                                             test_data=ds.normal_interp_part,
                                                             interp_type=interp_type,
                                                             args=args)

    # Extrapolation cases
    mae_extr, ssim_extr, psnr_extr = gpr_vae_1to1_test(vae=vae,
                                                       test_data=ds.extrapolation_part,
                                                       interp_type=interp_type,
                                                       args=args)

    # Print all metrics
    gpr_evaluation.print_extrapolation(mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr, True)


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
