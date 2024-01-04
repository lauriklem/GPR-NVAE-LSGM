import argparse
import os
import numpy as np
import torch
from torch.multiprocessing import Process
from nvae import NVAE
from util import utils
from PIL import Image
import torchvision.transforms as transforms
from gpr_vae_3to1_latent_analysis import list_data_3to1
import pickle
import matplotlib.pyplot as plt


def gpr_latent_nn_get_latents(vae, data, args, datadir=""):
    """
    Calculates all latents for given image list.
    """
    trans = transforms.Compose([transforms.ToTensor()])

    latents, gt_list = [], []
    with torch.no_grad():
        for row in data:
            input1, input2, gt, label = row
            # Encode left image
            in1 = Image.open(datadir + input1)
            in1 = np.array(in1.getdata(), dtype=np.uint8).reshape((in1.size[1], in1.size[0], 3))
            in1 = trans(Image.fromarray(in1, mode='RGB'))
            in1 = utils.common_x_operations([in1, in1], args.num_x_bits).unsqueeze(0)
            all_eps1, all_z1 = vae.calculate_eps_and_z(in1)

            l1 = []
            for z in all_z1:
                l1.append(z.cpu().numpy()[0, :, :, :].astype("float32"))  # Channels, spatial dim, spatial dim

            # Encode right image
            in2 = Image.open(datadir + input2)
            in2 = np.array(in2.getdata(), dtype=np.uint8).reshape((in2.size[1], in2.size[0], 3))
            in2 = trans(Image.fromarray(in2, mode='RGB'))
            in2 = utils.common_x_operations([in2, in2], args.num_x_bits).unsqueeze(0)
            all_eps1, all_z1 = vae.calculate_eps_and_z(in2)

            l2 = []
            for z in all_z1:
                l2.append(z.cpu().numpy()[0, :, :, :].astype("float32"))

            lbls = []
            for z in all_z1:
                dims = z.cpu().numpy()[0, :, :, :].shape
                label_img = np.full((1, dims[1], dims[2]), label)
                lbls.append(label_img)

            latents.append([l1, l2, lbls])

            gt = Image.open(datadir + gt)
            gt = np.array(gt.getdata(), dtype=np.uint8).reshape((gt.size[1], gt.size[0], 3))
            gt = trans(Image.fromarray(gt, mode='RGB'))
            gt = utils.common_x_operations([gt, gt], args.num_x_bits).unsqueeze(0)
            all_eps1, all_z1 = vae.calculate_eps_and_z(gt)

            gt2 = []
            for z in all_z1:
                gt2.append(z.cpu().numpy()[0, :, :, :].astype("float32"))
            gt_list.append(gt2)

            """
            fig, ax = plt.subplots(len(gt2), 4)
            for m in range(len(gt2)):
                lerped = (1 - label) * l1[m][0, :, :] + label * l2[m][0, :, :]
                diff = 1 - np.abs(lerped - gt2[m][0, :, :])
                ax[m, 0].imshow(l1[m][0, :, :], cmap="gray")
                ax[m, 0].axis("off")
                ax[m, 1].imshow(l2[m][0, :, :], cmap="gray")
                ax[m, 1].axis("off")
                ax[m, 2].imshow(gt2[m][0, :, :], cmap="gray")
                ax[m, 2].axis("off")
                ax[m, 3].imshow(diff, cmap="gray")
                ax[m, 3].axis("off")
                print(np.round(np.sum(np.abs(lerped - gt2[m][0, :, :])), 1), end="  ")
            print()
            plt.tight_layout()
            plt.show()
            """

    return latents, gt_list


def list_images():
    """
    Lists images in simulated data (folders 18-33).
    Test folders are 27 and 30.
    """
    images_list, images_list_test = [], []
    for folder in range(18, 34):
        folder_path = "./datasets/gpr_pics/{}/".format(folder)
        n_images = len(os.listdir(folder_path))
        for i in range(n_images):
            if folder in [27, 30]:
                images_list_test.append(folder_path + "{}.jpg".format(i))
            else:
                images_list.append(folder_path + "{}.jpg".format(i))

    return images_list, images_list_test


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

    # List all images
    sim_train, sim_test = [], [27, 30]
    for folder in range(18, 34):
        if folder not in sim_test:
            sim_train.append(folder)
    train_ind = list_data_3to1(datadir="./datasets/gpr_pics", folders=sim_train)
    valid_ind = list_data_3to1(datadir="./datasets/gpr_pics", folders=sim_test)

    dst_folder = "./latent_nn/latents/"
    dst_folder += eval_args.checkpoint.split("./checkpoints/")[-1].split("/vae")[0] + "/"
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    """
    # Determine the number of latent groups
    trans = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        im = Image.open(train_ind[0][0])
        im = np.array(im.getdata(), dtype=np.uint8).reshape((im.size[1], im.size[0], 3))
        im = trans(Image.fromarray(im, mode='RGB'))
        im = utils.common_x_operations([im, im], args.num_x_bits).unsqueeze(0)
        all_eps1, all_z1 = vae.calculate_eps_and_z(im)

    n_latents = len(all_z1)
    print("Latents have dimensionality {}".format(list(all_z1[0].size())[1:]))
    """

    latent_file = dst_folder + "latents_train.pkl"
    latent_file_test = dst_folder + "latents_test.pkl"
    gt_file = dst_folder + "gt_train.pkl"
    gt_file_test = dst_folder + "gt_test.pkl"

    print("Calculating latents...")
    latents, gt_list = gpr_latent_nn_get_latents(vae, train_ind, args)
    latents_test, gt_list_test = gpr_latent_nn_get_latents(vae, valid_ind, args)

    print("Writing to files...")
    f = open(latent_file, "wb")
    pickle.dump(latents, f)
    f.close()
    f = open(latent_file_test, "wb")
    pickle.dump(latents_test, f)
    f.close()

    f = open(gt_file, "wb")
    pickle.dump(gt_list, f)
    f.close()
    f = open(gt_file_test, "wb")
    pickle.dump(gt_list_test, f)
    f.close()


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
