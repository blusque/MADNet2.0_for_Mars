import torch.nn
import numpy as np
# from skimage import io
# import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from model.generator import Generator
from dataset import DEMDataset
from torch.utils.data import DataLoader
# from hill_shade import hill_shade


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_rse(img0, img1, L=255.):
    mse = ((np.abs(img0 - img1) / L) ** 2).mean()
    return np.sqrt(mse)


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel, now you have {} channels.".format(len(im1.shape)))

    M, N = im1.shape
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    window = matlab_style_gauss2d(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigma_l2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma_l2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return np.mean(np.mean(ssim_map))


class Validator:
    def __init__(self, gt, predicted):
        self.gt = gt
        self.predicted = predicted

    def validate(self):
        channels = self.gt.shape[0]
        rse = 0.
        ssim = 0.
        for i in range(channels):
            gt = self.gt[i, 0, :, :]
            # io.imshow(gt)
            # plt.show()
            predicted = self.predicted[i, 0, :, :]
            # io.imshow(predicted)
            # plt.show()
            rse += compute_rse(gt, predicted)
            ssim += compute_ssim(gt, predicted)
            # gt_relief = hill_shade(gt, z_factor=1.0)
            # predicted_relief = hill_shade(predicted, z_factor=1.0)
            # io.imshow(gt_relief)
            # plt.show()
            # io.imshow(predicted_relief)
            # plt.show()
        rse /= channels
        # print(rse)
        ssim /= channels
        # print(ssim)
        return rse, ssim


if __name__ == "__main__":
    model_path = '../checkpoint/gen_model_epoch_100.pth'
    model = Generator().to(device)
    state_dict = torch.load(model_path)['model'].state_dict()
    model.load_state_dict(state_dict, strict=False)

    dataset = DEMDataset('/media/mei/Elements/validating_dataset.hdf5')
    batch_size = 8
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    err = 0.
    for iteration, batch in enumerate(data_loader, 1):
        dtm, ori = batch
        print('dtm max: {}, min: {}'.format(dtm.max(), dtm.min()))
        dtm = torch.unsqueeze(dtm, 1)
        ori = torch.unsqueeze(ori, 1)
        dtm = dtm.numpy()
        dtm = dtm[:, :, 36: dtm.shape[2] - 36, 36: dtm.shape[3] - 36]
        ori = ori.to(device)

        gen_dtm = model(ori).cpu().detach().numpy() + 1
        gen_dtm = gen_dtm[:, :, 36: gen_dtm.shape[2] - 36, 36: gen_dtm.shape[3] - 36]
        gen_dtm = 100 * np.log10(10 * gen_dtm)
        # gen_dtm /= gen_dtm.max()
        # gen_dtm = np.abs(gen_dtm)
        # gen_dtm *= 254.
        for i in range(gen_dtm.shape[0]):
            print('gen_dtm {} max: {}, min: {}'.format(i, gen_dtm[i].max(), gen_dtm[i].min()))
            print('gen_dtm {} mean: {}, var: {}'.format(i, gen_dtm[i].mean(), gen_dtm[i].var()))

        val = Validator(dtm, gen_dtm)
        rse, ssim = val.validate()
        err += rse + ssim;
        print("total: 100; now: {}".format(iteration * batch_size))
        break
    # err /= batch_size
    # print(err.shape)
    print("mean err: ", err)
