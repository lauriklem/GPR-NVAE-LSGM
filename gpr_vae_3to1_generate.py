import argparse
import numpy as np
import torch
from torch.multiprocessing import Process
from nvae_3to1 import NVAE
from util import utils
from PIL import Image
import torchvision.transforms as transforms
from gpr_generate_utils import generate_paths
import os
from gpr_diff_images import difference_images


"""
Generate images with 3-to-1 NVAE
"""


def main(eval_args):
    # common initialization
    logging, writer = utils.common_init(eval_args.global_rank, eval_args.seed, eval_args.save)

    # load a checkpoint
    # logging.info('#' * 80)
    # logging.info('loading the model at:')
    # logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

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

    trans = transforms.Compose([transforms.ToTensor()])
    batchsize = args.batch_size

    images_between = [4]
    left_images = [0, 5, 7, 9] + [5, 7, 9, 15] + [3, 11, 12] + [2, 3, 12, 13]
    folders = [27, 27, 27, 27] + [30, 30, 30, 30] + [27, 27, 27] + [30, 30, 30, 30]

    dst_folder = "./results/vae_3to1/"
    dst_folder += eval_args.checkpoint.split("./checkpoints/")[-1].split("/vae")[0] + "/"
    datadir = "./datasets/gpr_pics"

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
                    input1 = datadir + in1_paths[j]
                    input2 = datadir + in2_paths[j]
                    temp_gt = gt_paths[j]

                    for label_ind in range(len(temp_gt)):
                        gt = Image.open(datadir + temp_gt[label_ind])
                        gt.save(dst_folder + str(j) + "_gt_{:.2f}_ib_{}.png".format(labels[label_ind], ib))

                    input1 = Image.open(input1)
                    im1 = np.array(input1.getdata(), dtype="float32").reshape(
                        (input1.size[1], input1.size[0], 3)) / 128.0 - 1

                    input2 = Image.open(input2)
                    im2 = np.array(input2.getdata(), dtype="float32").reshape(
                        (input2.size[1], input2.size[0], 3)) / 128.0 - 1

                    left_path = dst_folder + str(j) + "_left_ib_{}.png".format(ib)
                    right_path = dst_folder + str(j) + "_right_ib_{}.png".format(ib)
                    input1.save(left_path)
                    input2.save(right_path)

                    for label in labels:
                        lblimg = np.full(fill_value=label * 2 - 1, shape=(im1.shape[1], im1.shape[0], 1),
                                         dtype="float32")
                        x = np.concatenate((im1, im2, lblimg), axis=-1)
                        x = trans(x).unsqueeze(0)

                        x = utils.common_x_operations_cond(x)

                        logits, all_log_q, all_eps = vae(x)

                        output = vae.decoder_output(logits)
                        generated = output.mean()
                        generated = generated.clamp(min=-1., max=1.)
                        generated = utils.unsymmetrize_image_data(generated)

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
