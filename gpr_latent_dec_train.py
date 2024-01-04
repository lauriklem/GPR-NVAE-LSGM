import argparse

import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from util import utils
from nvae_dec_only import NVAE_dec
import torch
import gpr_evaluation
from gpr_dataset import GPRDataset
from util.sr_utils import SpectralNormCalculator
from thirdparty.adamax import Adamax
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.multiprocessing import Process
import matplotlib.pyplot as plt


def list_data(latents, gt_list):
    data_list = []
    trans = transforms.Compose([transforms.ToTensor()])
    for i in range(len(latents)):
        z_all = []
        for z in latents[i]:
            z = torch.tensor(z, dtype=torch.float).cuda().unsqueeze(0)
            z_all.append(z)
        gt = trans(gt_list[i]).unsqueeze(0) * 2.0 - 1
        data_list.append([z_all, gt])

    return data_list


def main(args):
    print("Loading data...")
    fname = args.latents + "latents_train.pkl"
    f = open(fname, "rb")
    latents_train = pickle.load(f)
    f.close()

    fname = args.latents + "latents_test.pkl"
    f = open(fname, "rb")
    latents_test = pickle.load(f)
    f.close()

    fname = args.latents + "gt_train.pkl"
    f = open(fname, "rb")
    latents_gt_train = pickle.load(f)
    f.close()

    fname = args.latents + "gt_test.pkl"
    f = open(fname, "rb")
    latents_gt_test = pickle.load(f)
    f.close()

    train_queue = list_data(latents_train, latents_gt_train)
    valid_queue = list_data(latents_test, latents_gt_test)

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    model = NVAE_dec(args, arch_instance_nvae)
    model = model.cuda()

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    vae_optimizer = Adamax(model.parameters(), args.learning_rate_vae,
                           weight_decay=args.weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)

    # create SN calculator
    sn_calculator = SpectralNormCalculator()
    sn_calculator.add_conv_layers(model)
    sn_calculator.add_bn_layers(model)

    grad_scalar = GradScaler(2 ** 10)
    bpd_coeff = utils.get_bpd_coeff(args.dataset)

    milestones_epochs = range(0, args.epochs, 10)
    global_step, epoch, init_epoch, best_score = 0, 0, 0, 1e10
    rng = np.random.default_rng()

    print("Training...")
    for epoch in range(init_epoch, args.epochs):
        rng.shuffle(train_queue)
        rng.shuffle(valid_queue)
        if epoch > args.warmup_epochs:
            vae_scheduler.step()

        train_obj, global_step = train(model, train_queue, vae_optimizer, grad_scalar, global_step, warmup_iters, sn_calculator, args)
        print("Epoch {}, {:.2f}".format(epoch, train_obj))

    for j in range(len(valid_queue)):
        generated = model.sample_secondary_decoder(valid_queue[j][0])
        perm_dims = (0, 2, 3, 1)
        generated = torch.permute(generated, perm_dims).cpu().numpy() * 255
        generated = np.round(generated[0], 0).astype("uint8")

        trans = transforms.Compose([transforms.ToPILImage()])
        gt = valid_queue[j][1][0]
        gt = trans((gt + 1) / 2)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(generated)
        ax[1].imshow(gt)
        plt.show()


def train(model, train_queue, optimizer, grad_scalar, global_step, warmup_iters, sn_calculator, args):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for i in range(len(train_queue)):
        x = train_queue[i][0]
        gt = train_queue[i][1].cuda()

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            logits = model.forward_secondary_dec(x)
            output = model.decoder_output(logits)
            recon_loss = utils.reconstruction_loss(output, gt, crop=model.crop_output)

            nelbo_batch = recon_loss
            loss = torch.mean(nelbo_batch)

            norm_loss = sn_calculator.spectral_norm_parallel()
            bn_loss = sn_calculator.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(
                    args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff


        grad_scalar.scale(loss).backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)
        global_step += 1
    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    parser.add_argument('--latents', type=str, default='/path/to/latents.pkl',
                        help='location of the latents')
    parser.add_argument('--dataset', type=str, default='gpr', help='which dataset to use')
    # optimization
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate_vae', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff', type=float, default=1.,
                        help='The constant value used for max KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--log_sig_q_scale', type=float, default=5.,  # we used to use [-5, 5]
                        help='log sigma q is clamped into [-log_sig_q_scale, log_sig_q_scale].')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--channel_mult', nargs='+', type=int,
                        help='channel multiplier per scale', )
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--decoder_dist', type=str, default='dml', choices=['normal', 'dml', 'dl', 'bin'],
                        help='Distribution used in VAE decoder: Normal, Discretized Mix of Logistic,'
                             'Bernoulli, or discretized logistic.')
    parser.add_argument('--progressive_input_vae', type=str, default='none', choices=['none', 'input_skip'],
                        help='progressive type for input')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--autocast_train', action='store_true', default=True,
                        help='This flag enables FP16 in training.')
    parser.add_argument('--autocast_eval', action='store_true', default=True,
                        help='This flag enables FP16 in evaluation.')
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
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')

    args = parser.parse_args()

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