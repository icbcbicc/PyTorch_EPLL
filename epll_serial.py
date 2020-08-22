import numpy as np
import scipy.io as sio
import torch

from utils import avg_col2im_serial, im2col_serial


class EPLL_serial():
    """
    Execuate in a fully serial manner with minimium RAM usage
    for im in batch:
        for channel in im:
            process this channel
    """

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
        self.GMM["dim"] = torch.tensor(mat_contents['GS']['dim'][0, 0].item()).to(self.device)
        self.GMM["nmodels"] = torch.tensor(mat_contents['GS']['nmodels'][0, 0].item()).to(self.device)
        self.GMM["means"] = torch.tensor(mat_contents['GS']['means'][0, 0], dtype=torch.float32).to(self.device)
        self.GMM["covs"] = torch.tensor(mat_contents['GS']['covs'][0, 0], dtype=torch.float32).to(self.device)
        self.GMM["invcovs"] = torch.tensor(mat_contents['GS']['invcovs'][0, 0], dtype=torch.float32).to(self.device)
        self.GMM["mixweights"] = torch.tensor(mat_contents['GS']['mixweights'][0, 0], dtype=torch.float32).to(self.device)

    def prior(self, noise_imcol, noise_sd) -> torch.Tensor:
        # noise_imcol.shape: [64, -1]

        def log_gaussian_pdf(X, sigma):
            R = torch.cholesky(sigma)
            solutoin, _ = torch.solve(X, R)
            q = torch.sum(solutoin**2, dim=0)
            c = X.shape[0] * torch.log(torch.tensor(2 * np.pi)) + 2 * torch.sum(torch.log(torch.diagonal(R)))
            y = -(c + q) / 2
            return y

        # remove DC component
        mean_noise_imcol = torch.mean(noise_imcol, dim=0)
        noise_imcol -= mean_noise_imcol

        GMM_noisy_convs = torch.zeros_like(self.GMM["covs"])
        p_y_z = torch.zeros([self.GMM["nmodels"], noise_imcol.shape[-1]])
        sigma_noise = (noise_sd**2 * torch.eye(8**2)).to(self.device)
        for i in range(self.GMM["nmodels"]):
            GMM_noisy_convs[:, :, i] = self.GMM["covs"][:, :, i] + sigma_noise
            p_y_z[i] = torch.log(self.GMM["mixweights"][i]) + log_gaussian_pdf(noise_imcol, GMM_noisy_convs[:, :, i])

        max_index = torch.argmax(p_y_z, dim=0)

        # weiner filtering
        Xhat = torch.zeros_like(noise_imcol)
        for i in range(self.GMM["nmodels"]):
            index = torch.nonzero((max_index - i) == 0, as_tuple=False)[:, 0]
            A = self.GMM["covs"][:, :, i] + sigma_noise
            B = torch.matmul(self.GMM["covs"][:, :, i], noise_imcol[:, index]) + torch.matmul(sigma_noise, self.GMM["means"][:, i].unsqueeze(dim=1).repeat(1, len(index)))
            solutoin, _ = torch.solve(B, A)
            Xhat[:, index] = solutoin

        Xhat += mean_noise_imcol

        return Xhat

    def denoise(self, noise_im_batch) -> torch.Tensor:
        """
        input:
            noise_im: [b, c, h, w]
            clean_im: [b, c, h, w]
        return:
            restored_im: [b, c, h, w]
        """

        restored_im_batch = torch.zeros_like(noise_im_batch)
        for im_count, noise_im in enumerate(noise_im_batch):
            print(f"\nImage {im_count}")
            # half quadratic split
            restored_im = noise_im.clone()
            for beta in self.betas:
                for t in range(self.num_iters):
                    for c in range(noise_im.shape[0]):
                        restored_imcol = im2col_serial(restored_im[c], 8, 8, self.stride)      # matlab style im2col, output shape = [batch, path_size**2, num_patches],
                        restored_imcol = self.prior(noise_imcol=restored_imcol, noise_sd=beta**(-0.5))
                        I1 = avg_col2im_serial(restored_imcol, noise_im.shape[1], noise_im.shape[2], self.stride)
                        restored_im[c] = noise_im[c] * self.lamb / (self.lamb + beta * 8**2) + (beta * 8**2 / (self.lamb + beta * 8**2)) * I1

                    psnr1 = 10 * torch.log10(1 / torch.mean((restored_im - self.clean_im) ** 2))
                    # psnr2 = 10 * torch.log10(1 / torch.mean((I1 - clean_im) ** 2))
                    print(f"    [beta={beta:.3f}, iter={t}] PSNR={psnr1.item():.3f}")

            restored_im_batch[im_count] = restored_im

        torch.clamp_(restored_im_batch, min=0, max=1)
        return restored_im_batch
