import numpy as np
from PIL import Image
import gpr_evaluation
import os
import json
import argparse


def difference_images(n, images_between, img_folder):
    """
    Calculates MAE and SSIM images between generated and ground truth images.
    """
    diff_folder = img_folder + "diff/"
    if not os.path.exists(diff_folder):
        os.makedirs(diff_folder)

    metrics_list = []

    for i in range(n):
        for ib in images_between:
            labels = np.array(range(1, ib + 1)) / float(ib + 1)
            for l in labels:
                name = "{}_gen_{:.2f}_ib_{}.png".format(i, l, ib)
                gen = Image.open(img_folder + name)
                gt = Image.open(img_folder + "{}_gt_{:.2f}_ib_{}.png".format(i, l, ib))
                mae, mae_img = gpr_evaluation.mean_absolute_error(gen, gt, get_image=True)
                ssim, ssim_img = gpr_evaluation.struc_sim(gen, gt, get_image=True)
                metrics_list.append({
                    "name": name,
                    "mae": float(mae),
                    "ssim": float(ssim)
                })
                mae_img = np.mean(mae_img, axis=-1)
                mae_img = 255.0 - (mae_img - np.min(mae_img)) / (np.max(mae_img) - np.min(mae_img)) * 255.0  # minmax scaling to range 0-255
                # mae_img = 255.0 - mae_img
                mae_img = np.round(mae_img, 0).astype("uint8")
                mae_img = Image.fromarray(mae_img)
                mae_img.save(diff_folder + "{}_mae_{:.2f}_ib_{}.png".format(i, l, ib))

                ssim_img = np.mean(ssim_img, axis=-1)
                # ssim_img = (ssim_img - np.min(ssim_img)) / (np.max(ssim_img) - np.min(ssim_img)) * 255.0  # minmax scaling to range 0-255
                ssim_img = (1 + ssim_img) * 255.0 / 2
                ssim_img = np.round(ssim_img, 0).astype("uint8")
                ssim_img = Image.fromarray(ssim_img)
                ssim_img.save(diff_folder + "{}_ssim_{:.2f}_ib_{}.png".format(i, l, ib))

    f = open(diff_folder + "mae_ssim", "w")
    json.dump(metrics_list, f, indent=4)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate difference images between ground truth and generated images')
    parser.add_argument('--dir', '-d',
                        help='Directory of the generated images, e.g. ./results/ib4_epoch300_batch16/generated/')

    args = parser.parse_args()

