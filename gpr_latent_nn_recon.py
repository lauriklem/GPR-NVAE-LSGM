import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plot_settings import plotting_fonts
import os
import numpy as np


"""
Plot metrics during training of a 1-to-1 NVAE model 
"""


def get_recon(path):
    """
    List reconstuction metrics from the given checkpoint path
    """
    f = open(path + "/acc_recon", 'r')
    list1 = json.load(f)
    epochs, mae_train, ssim_train, psnr_train, mae_test, ssim_test, psnr_test = [], [], [], [], [], [], []
    for elem in list1:
        epochs.append(elem["epoch"])
        mae_train.append(elem["mae_train"])
        ssim_train.append(elem["ssim_train"])
        psnr_train.append(elem["psnr_train"])
        mae_test.append(elem["mae_test"])
        ssim_test.append(elem["ssim_test"])
        psnr_test.append(elem["psnr_test"])

    f.close()
    return epochs, mae_train, ssim_train, psnr_train, mae_test, ssim_test, psnr_test


if __name__ == "__main__":
    plotting_fonts()  # Set plotting fonts
    fig_format = "pdf"
    save_fig = False

    model = "latent_nn"
    dst_folder = "./accs/" + model

    path = "./checkpoints/{}/vae".format(model)
    epochs, mae_train, ssim_train, psnr_train, mae_test, ssim_test, psnr_test = get_recon(path)

    plt.figure(figsize=(12.8, 9.6))
    gs = gridspec.GridSpec(4, 4)

    # MAE
    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(epochs, mae_train, c="C0")
    ax1.plot(epochs, mae_test, c="C1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE")
    ax1.legend(["Train", "Test"])

    # SSIM
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(epochs, ssim_train, c="C0")
    ax2.plot(epochs, ssim_test, c="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend(["Train", "Test"])

    # PSNR
    ax3 = plt.subplot(gs[2:, 1:3])
    ax3.plot(epochs, psnr_train, c="C0")
    ax3.plot(epochs, psnr_test, c="C1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR")
    ax3.legend(["Train", "Test"])

    plt.tight_layout()
    if save_fig:
        plt.savefig(dst_folder + "/recon.{}".format(fig_format), format=fig_format)
    else:
        plt.show()