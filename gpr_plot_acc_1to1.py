import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plot_settings import plotting_fonts
import os
import numpy as np


"""
Plot metrics during training of a 1-to-1 NVAE model 
"""


def get_accs_extr(path):
    """
    List all accuracies from the given checkpoint path
    """
    common, linear, spherical, epochs = [], [], [], []
    for acc_type in ["common", "linear", "spherical"]:
        f = open(path + "/acc_" + acc_type, 'r')
        list1 = json.load(f)

        if acc_type == "common":
            bpd_log_p, nat_log_p, bpd_elbo, nat_elbo, train_loss = [], [], [], [], []
            for elem in list1:
                bpd_log_p.append(elem["bpd_log_p"])
                nat_log_p.append(elem["nat_log_p"])
                bpd_elbo.append(elem["bpd_elbo"])
                nat_elbo.append(elem["nat_elbo"])
                train_loss.append(elem["train_loss"])
            common = [bpd_log_p, nat_log_p, bpd_elbo, nat_elbo, train_loss]

        else:
            mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr, epochs = [], [], [], [], [], [], []
            for elem in list1:
                mae_normal.append(elem['mae_normal'])
                mae_extr.append(elem['mae_extr'])
                ssim_normal.append(elem['ssim_normal'])
                ssim_extr.append(elem['ssim_extr'])
                psnr_normal.append(elem['psnr_normal'])
                psnr_extr.append(elem['psnr_extr'])
                epochs.append(elem['epoch'])
            if acc_type == "linear":
                linear = [mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr]
            else:
                spherical = [mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr]
        f.close()
    return common, linear, spherical, epochs


if __name__ == "__main__":
    plotting_fonts()  # Set plotting fonts
    fig_format = "pdf"
    save_fig = False

    model = "latent_nn"
    dst_folder = "./accs/" + model

    path = "./checkpoints/{}/vae".format(model)
    common, linear, spherical, epochs = get_accs_extr(path)
    bpd_log_p, nat_log_p, bpd_elbo, nat_elbo, train_loss = common
    mae_normal_l, mae_extr_l, ssim_normal_l, ssim_extr_l, psnr_normal_l, psnr_extr_l = linear
    mae_normal_s, mae_extr_s, ssim_normal_s, ssim_extr_s, psnr_normal_s, psnr_extr_s = spherical

    mae_sum_l = np.array(mae_normal_l) + np.array(mae_extr_l)
    ssim_sum_l = np.array(ssim_normal_l) + np.array(ssim_extr_l)
    psnr_sum_l = np.array(psnr_normal_l) + np.array(psnr_extr_l)

    mae_sum_s = np.array(mae_normal_s) + np.array(mae_extr_s)
    ssim_sum_s = np.array(ssim_normal_s) + np.array(ssim_extr_s)
    psnr_sum_s = np.array(psnr_normal_s) + np.array(psnr_extr_s)

    # Plot bpd and train loss
    fig, axs, = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axs[0].plot(epochs, bpd_elbo)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("bpd ELBO")

    # Train loss
    axs[1].plot(epochs, train_loss)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Training loss")
    plt.tight_layout()
    if save_fig:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        plt.savefig(dst_folder + "/loss.{}".format(fig_format), format=fig_format)

    # Normal interpolation
    plt.figure(figsize=(12.8, 9.6))
    gs = gridspec.GridSpec(4, 4)

    # MAE
    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(epochs, mae_normal_l, c="C0")
    ax1.plot(epochs, mae_normal_s, c="C0", linestyle="dotted")
    ax1.plot(epochs, mae_extr_l, c="C1")
    ax1.plot(epochs, mae_extr_s, c="C1", linestyle="dotted")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE")
    ax1.legend(["Normal LERP", "Normal SLERP", "Edge LERP", "Edge SLERP"])

    # SSIM
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(epochs, ssim_normal_l, c="C0")
    ax2.plot(epochs, ssim_normal_s, c="C0", linestyle="dotted")
    ax2.plot(epochs, ssim_extr_l, c="C1")
    ax2.plot(epochs, ssim_extr_s, c="C1", linestyle="dotted")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend(["Normal LERP", "Normal SLERP", "Edge LERP", "Edge SLERP"])

    # PSNR
    ax3 = plt.subplot(gs[2:, 1:3])
    ax3.plot(epochs, psnr_normal_l, c="C0")
    ax3.plot(epochs, psnr_normal_s, c="C0", linestyle="dotted")
    ax3.plot(epochs, psnr_extr_l, c="C1")
    ax3.plot(epochs, psnr_extr_s, c="C1", linestyle="dotted")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR")
    ax3.legend(["Normal LERP", "Normal SLERP", "Edge LERP", "Edge SLERP"])
    plt.tight_layout()
    if save_fig:
        plt.savefig(dst_folder + "/metrics.{}".format(fig_format), format=fig_format)

    fig = plt.figure(figsize=(12.8, 9.6))
    gs = gridspec.GridSpec(4, 4)

    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(epochs, mae_sum_l, c="C0")
    ax1.plot(epochs, mae_sum_s, c="C1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE")
    ax1.legend(["LERP", "SLERP"])

    # SSIM
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(epochs, ssim_sum_l, c="C0")
    ax2.plot(epochs, ssim_sum_s, c="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend(["LERP", "SLERP"])

    # PSNR
    ax3 = plt.subplot(gs[2:, 1:3])
    ax3.plot(epochs, psnr_sum_l, c="C0")
    ax3.plot(epochs, psnr_sum_s, c="C1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR")
    ax3.legend(["LERP", "SLERP"])
    plt.tight_layout()
    if save_fig:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        plt.savefig(dst_folder + "/metrics_sums.{}".format(fig_format), format=fig_format)
    else:
        plt.show()