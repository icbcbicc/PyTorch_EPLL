import os
import platform
import time

import joblib
import numpy as np
import PIL.Image as Image
from sklearn import feature_extraction, mixture


def read_data(data_path, num_samples):

    seed = 2020

    img_list = sorted(os.listdir(data_path))
    patch = np.zeros((num_samples * len(img_list), 8, 8))

    for i, im_name in enumerate(img_list):
        im = np.asarray(Image.open(os.path.join(data_path, im_name)).convert("L")) / 255
        patch[i * num_samples: (i + 1) * num_samples] = feature_extraction.image.extract_patches_2d(im, patch_size=(8, 8), max_patches=num_samples, random_state=seed)

    patch -= np.mean(patch, axis=(1, 2), keepdims=True)
    patch = patch.reshape((patch.shape[0], patch.shape[1] * patch.shape[2]))

    return patch


if __name__ == '__main__':

    num_samples = 10000
    n_components = 200
    is_train = True

    pc_name = platform.node()
    if pc_name == "vision-pc26-Ubuntu":
        data_path = "/home/z2228wan/data/BSDS300/images"
    else:
        data_path = "BSDS300/images"

    # load data
    print("[*] Loading data ...\t", end="")
    start = time.time()
    train_data = read_data(os.path.join(data_path, "train"), num_samples)
    test_data = read_data(os.path.join(data_path, "test"), num_samples)
    np.save(f"train_gmm/train_data_{num_samples}.npy", train_data)
    np.save(f"train_gmm/test_data_{num_samples}.npy", test_data)
    print(f"{time.time() - start:.3f} s")

    # fit a GMM model with EM
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=500, tol=1e-6, verbose=2, verbose_interval=1)
    if is_train:
        print("[*] Fitting ...")
        start = time.time()
        gmm.fit(train_data)
        print(f"Fitting takes {time.time() - start:.3f} s")
        joblib.dump(gmm, f"train_gmm/gmm_{n_components}.joblib")
    else:
        gmm = joblib.load(f"train_gmm/gmm_{n_components}.joblib")

    # testing this model
    log_prob = gmm.score_samples(test_data)
    neg_log_prob = - np.mean(log_prob)
    print(f"[*] Negative log likeliohood on testing set: {neg_log_prob:.3f}")
