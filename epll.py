import numpy as np
import scipy.io as sio
import torch

from utils import avg_col2im, im2col


class EPLL():
    def __init__(self, clean_im, lamb, betas, num_iters, stride, device):
        self.clean_im = clean_im
        self.lamb = lamb
        self.betas = betas
        self.num_iters = num_iters
        self.stride = stride
        self.device = device

    def load_GMM(self, prior_file):
        mat_contents = sio.loadmat(prior_file)
        self.GMM = {}
        self.GMM["nmodels"] = torch.tensor(mat_contents['GS']['nmodels'][0, 0].item()).to(self.device)
        self.GMM["means"] = torch.tensor(mat_contents['GS']['means'][0, 0], dtype=torch.float32).to(self.device).permute(1, 0)        # shape: [200, 64]
        self.GMM["mixweights"] = torch.tensor(mat_contents['GS']['mixweights'][0, 0], dtype=torch.float32).to(self.device)            # shape: [200, 1]
        self.GMM["covs"] = torch.tensor(mat_contents['GS']['covs'][0, 0], dtype=torch.float32).to(self.device).permute(2, 0, 1)       # shape: [200, 64, 64]

    def prior(self, noise_imcol, noise_sd) -> torch.Tensor:

        def log_gaussian_pdf(X, sigma):
            R = torch.cholesky(sigma)
            q = torch.zeros([X.shape[0], X.shape[1], R.shape[0], X.shape[3]]).to(self.device)
            for i, item in enumerate(X):
                for c, ctem in enumerate(item):
                    solution = torch.matmul(R.inverse(), ctem)
                    q[i, c] = torch.sum(solution**2, dim=1)
            c = X.shape[2] * torch.log(torch.tensor(2 * np.pi)).to(self.device) + 2 * torch.sum(torch.log(torch.diagonal(R, dim1=1, dim2=2)), dim=1, keepdim=True)
            c = c.unsqueeze(0).unsqueeze(0)
            y = -(c + q) / 2
            y = torch.mean(y, dim=1)
            return y

        # remove DC component
        mean_noise_imcol = torch.mean(noise_imcol, dim=2, keepdim=True)
        noise_imcol -= mean_noise_imcol

        GMM_noisy_covs = torch.zeros_like(self.GMM["covs"])
        sigma_noise = (noise_sd**2 * torch.eye(8**2)).to(self.device)      # shape: [64, 64]
        GMM_noisy_covs = self.GMM["covs"] + sigma_noise

        p_y_z = torch.log(self.GMM["mixweights"]) + log_gaussian_pdf(noise_imcol, GMM_noisy_covs)   # shape: [batch_size, 200, noise_imcol.shape[-1]]

        return p_y_z, mean_noise_imcol, GMM_noisy_covs

    def denoise(self, noise_im) -> torch.Tensor:
        """
        input:
            noise_im: [b, c, h, w]
            clean_im: [b, c, h, w]
        return:
            restored_im: [b, c, h, w]
        """
        # half quadratic split
        restored_im = noise_im.clone()
        for beta in self.betas:
            for t in range(self.num_iters):
                restored_imcol = im2col(restored_im, 8, 8, self.stride)      # matlab style im2col, output shape = [batch, c, patch_size**2, num_patches],

                p_y_z, mean_noise_imcol, GMM_noisy_covs = self.prior(noise_imcol=restored_imcol, noise_sd=beta**(-0.5))

                max_index = torch.argmax(p_y_z, dim=1)
                # weiner filtering
                Xhat = torch.zeros_like(restored_imcol)
                for b in range(Xhat.shape[0]):
                    for i in range(self.GMM["nmodels"]):
                        index = torch.nonzero((max_index[b] - i) == 0, as_tuple=False)[:, 0]
                        B = torch.matmul(self.GMM["covs"][i], restored_imcol[b, :, :, index])
                        solution = torch.matmul(GMM_noisy_covs[i].inverse(), B)
                        Xhat[b, :, :, index] = solution

                Xhat += mean_noise_imcol
                restored_imcol = Xhat

                I1 = torch.zeros_like(restored_im)
                for b in range(I1.shape[0]):
                    I1[b] = avg_col2im(restored_imcol[b], noise_im.shape[2], noise_im.shape[3], self.stride)
                restored_im = noise_im * self.lamb / (self.lamb + beta * 8**2) + (beta * 8**2 / (self.lamb + beta * 8**2)) * I1

                psnr1 = 10 * torch.log10(1 / torch.mean((restored_im - self.clean_im) ** 2))
                # psnr2 = 10 * torch.log10(1 / torch.mean((I1 - clean_im) ** 2))
                print(f"[beta={beta:.3f}, iter={t}] PSNR={psnr1.item():.3f}")

        torch.clamp_(restored_im, min=0, max=1)

        return restored_im
