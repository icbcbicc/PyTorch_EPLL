import argparse
import os
import time
from distutils.util import strtobool

import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.io as sio
import torch
import torchvision.transforms as transforms

from epll import EPLL


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--im_file", type=str, default="160068.jpg", help="path to the clean image")
    parser.add_argument("-std", "--noise_std", type=float, default=0.1, help="standard deviation of random gaussian noise")
    parser.add_argument("--use_cuda", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("-p", "--prior_file", type=str, default="GSModel_8x8_200_2M_noDC_zeromean.mat", help="path to the GMM prior")
    parser.add_argument("-n", "--noise_file", type=str, default="noise_160068.mat", help="path to the gaussian noise with std=1, debug only")
    return parser.parse_args()


if __name__ == "__main__":
    DEBUG = False

    cfg = parse_config()

    # load the image
    to_tensor = transforms.ToTensor()
    clean_im = to_tensor(Image.open(cfg.im_file)).float()     # [c, w, h]

    # make a batch
    clean_im = torch.stack([clean_im, clean_im], dim=0)        # [2, c, w, h]

    # add noise
    noise_std = cfg.noise_std
    if DEBUG is True:
        # fixed noise
        print(f"[*] Adding fixed noise: {cfg.noise_file}")
        mat_contents = sio.loadmat(file_name=cfg.noise_file)
        noise = torch.tensor(mat_contents['noise']).float()
        noise_im = clean_im + noise_std * noise
    else:
        # random noise
        print(f"[*] Adding random gaussian noise with std {noise_std:.3f}")
        noise_im = clean_im + noise_std * torch.randn(size=clean_im.shape)
        torch.clamp_(noise_im, min=0, max=1)

    # params
    lamb = 8**2 / noise_std**2
    betas = torch.tensor([1, 4, 8, 16, 32]) / noise_std**2
    num_iters = 1

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    print(f"[*] Using device {str(device).upper()}")
    noise_im = noise_im.to(device)
    clean_im = clean_im.to(device)

    epll = EPLL(clean_im, lamb, betas, num_iters, device)

    # load the GMM prior
    epll.load_GMM(cfg.prior_file)

    # denoising
    start = time.time()
    restored_im = epll.denoise(noise_im)
    print(f"[*] Eelapsed time is: {time.time() - start:.1f} s")

    # display
    batch_size = clean_im.shape[0]
    fig = plt.figure(figsize=(12, 4 * batch_size), dpi=200)

    if clean_im.shape[1] == 3:
        # [3, h, w] -> [h, w, 3]
        clean_im = clean_im.cpu().permute(0, 2, 3, 1)
        noise_im = noise_im.cpu().permute(0, 2, 3, 1)
        restored_im = restored_im.cpu().permute(0, 2, 3, 1)
    elif clean_im.shape[1] == 1:
        # [1, h, w] -> [h, w]
        clean_im = clean_im.cpu().squeeze(1)
        noise_im = noise_im.cpu().squeeze(1)
        restored_im = restored_im.cpu().squeeze(1)
    else:
        raise Exception(f"Invalid image shape: {clean_im.shape}")

    for i in range(batch_size):
        ax1 = fig.add_subplot(batch_size, 3, i * 3 + 1)
        plt.imshow(clean_im[i], cmap='gray')
        plt.axis('off')
        ax1.set_title('clean')

        ax2 = fig.add_subplot(batch_size, 3, i * 3 + 2)
        plt.imshow(noise_im[i], cmap='gray')
        plt.axis('off')
        ax2.set_title(f"noisy std={cfg.noise_std:.3f} PSNR={10 * torch.log10(1 / torch.mean((noise_im - clean_im) ** 2)):.3f}")

        ax3 = fig.add_subplot(batch_size, 3, i * 3 + 3)
        plt.imshow(restored_im[i], cmap='gray')
        plt.axis('off')
        ax3.set_title(f"restored PSNR={10 * torch.log10(1 / torch.mean((restored_im - clean_im) ** 2)):.3f}")

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(cfg.im_file)[0] + '_demo_batch' +  os.path.splitext(cfg.im_file)[1]}", dpi=200)
