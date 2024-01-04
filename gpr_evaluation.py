import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def mean_absolute_error(image1, image2, get_image=False):
    """
    Calculates the mean absolute error between two images.
    :param image1: Image 1
    :param image2: Image 2
    :param get_image: Return absolute error image or not
    :return: Mean absolute error
    """
    e = np.asarray(image1, dtype="float32") - np.asarray(image2, dtype="float32")
    ae = np.abs(e)
    mae = np.mean(ae)
    if get_image:
        return mae, ae
    return mae


def struc_sim(image1, image2, get_image=False):
    """
    Calculates the structural similarity between two images.
    :param image1: Image 1
    :param image2: Image 2
    :param get_image: Return ssim image or not
    :return: Structural similarity
    """
    array1 = np.asarray(image1, dtype="uint8")
    array2 = np.asarray(image2, dtype="uint8")
    if get_image:
        return ssim(array1, array2, data_range=255, channel_axis=2, full=True)  # , gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim(array1, array2, data_range=255, channel_axis=2)


def peak_snr(image1, image2):
    """
    Calculates the peak signal-to-noise ratio between two images.
    :param image1: Image 1
    :param image2: Image 2
    :return: Peak signal-to-noise ratio
    """
    array1 = np.asarray(image1, dtype="uint8")
    array2 = np.asarray(image2, dtype="uint8")
    return psnr(array1, array2, data_range=255)


def calculate_all(image1, image2):
    """
    Calculates MAE, SSIM, and PSNR between two images.
    """
    mae = mean_absolute_error(image1, image2)
    ss = struc_sim(image1, image2)
    pr = peak_snr(image1, image2)
    return mae, ss, pr


def print_metrics(mae_real, mae_sim, ssim_real, ssim_sim, psnr_real, psnr_sim):
    """
    Print means of MAE, SSIM, and PSNR for real and simulated data.
    """
    print("\tReal\tSim")
    print("MAE:\t{:.2f}\t{:.2f}".format(np.mean(mae_real), np.mean(mae_sim)))
    print("SSIM:\t{:.2f}\t{:.2f}".format(np.mean(ssim_real), np.mean(ssim_sim)))
    print("PSNR:\t{:.2f}\t{:.2f}".format(np.mean(psnr_real), np.mean(psnr_sim)))


def print_extrapolation(mae_normal, mae_extr, ssim_normal, ssim_extr, psnr_normal, psnr_extr, print_std=False):
    """
    Print means (and standard deviations) of MAE, SSIM, and PSNR for normal and extrapolation cases.
    """
    print("\tNormal\tExtrapolation")
    print("MAE:\t{:.2f}\t{:.2f}".format(np.mean(mae_normal), np.mean(mae_extr)))
    print("SSIM:\t{:.2f}\t{:.2f}".format(np.mean(ssim_normal), np.mean(ssim_extr)))
    print("PSNR:\t{:.2f}\t{:.2f}".format(np.mean(psnr_normal), np.mean(psnr_extr)))
    if print_std:
        print("Standard deviations:")
        print("\tNormal\tExtrapolation")
        print("MAE:\t{:.2f}\t{:.2f}".format(np.std(mae_normal), np.std(mae_extr)))
        print("SSIM:\t{:.2f}\t{:.2f}".format(np.std(ssim_normal), np.std(ssim_extr)))
        print("PSNR:\t{:.2f}\t{:.2f}".format(np.std(psnr_normal), np.std(psnr_extr)))
