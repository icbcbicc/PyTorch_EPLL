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
    parser.add_argument("-n", "--noise_file", type=str, default="noise.mat", help="path to the gaussian noise with std=1, debug only")
    return parser.parse_args()


def load_GMM(prior_file, device):
    mat_contents = sio.loadmat(prior_file)
    GMM = {}
    GMM["dim"] = torch.tensor(mat_contents['GS']['dim'][0, 0].item()).to(device)
    GMM["nmodels"] = torch.tensor(mat_contents['GS']['nmodels'][0, 0].item()).to(device)
    GMM["means"] = torch.tensor(mat_contents['GS']['means'][0, 0], dtype=torch.float32).to(device)
    GMM["covs"] = torch.tensor(mat_contents['GS']['covs'][0, 0], dtype=torch.float64).to(device)
    GMM["invcovs"] = torch.tensor(mat_contents['GS']['invcovs'][0, 0], dtype=torch.float64).to(device)
    GMM["mixweights"] = torch.tensor(mat_contents['GS']['mixweights'][0, 0], dtype=torch.float64).to(device)

    return GMM


if __name__ == "__main__":
    DEBUG = False

    cfg = parse_config()

    # load the image
    to_tensor = transforms.ToTensor()
    clean_im = to_tensor(Image.open(cfg.im_file).convert('L')).double()     # [1, w, h]

    # add noise
    noise_std = cfg.noise_std
    if DEBUG is True:
        # fixed noise
        print(f"[*] Adding fixed noise: {cfg.noise_file}")
        mat_contents = sio.loadmat(file_name=cfg.noise_file)
        noise = torch.tensor(mat_contents['noise'])
        noise_im = clean_im + noise_std * noise
    else:
        # random noise
        print(f"[*] Adding random gaussian noise with std {noise_std:.3f}")
        noise_im = clean_im + noise_std * torch.randn(size=clean_im.shape)

    # params
    lamb = 8**2 / noise_std**2
    betas = torch.tensor([1, 4, 8, 16, 32]) / noise_std**2

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    print(f"[*] Using device {device}")
    noise_im = noise_im.to(device)
    clean_im = clean_im.to(device)

    # load the prior
    GMM = load_GMM(cfg.prior_file, device)

    # denoise
    start = time.time()
    epll = EPLL(device)
    restored_im = epll.denoise(noise_im, clean_im, GMM, lamb, betas, 1)
    print(f"[*] Eelapsed time is: {time.time() - start:.1f} s")

    # display
    fig = plt.figure(figsize=(12, 4), dpi=200)

    clean_im = torch.squeeze(clean_im.cpu())
    noise_im = torch.squeeze(noise_im.cpu())
    restored_im = torch.squeeze(restored_im.cpu())

    ax1 = fig.add_subplot(1, 3, 1)
    plt.imshow(clean_im, cmap='gray')
    plt.axis('off')
    ax1.set_title('clean')

    ax2 = fig.add_subplot(1, 3, 2)
    plt.imshow(noise_im, cmap='gray')
    plt.axis('off')
    ax2.set_title(f"noise std={cfg.noise_std:.3f}")

    ax3 = fig.add_subplot(1, 3, 3)
    plt.imshow(restored_im, cmap='gray')
    plt.axis('off')
    ax3.set_title(f"restored PSNR={10 * torch.log10(1 / torch.mean((restored_im - clean_im) ** 2)):.3f}")

    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(cfg.im_file)[0] + '_demo' +  os.path.splitext(cfg.im_file)[1]}", dpi=200)
