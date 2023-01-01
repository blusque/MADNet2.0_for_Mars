import torch.nn
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from model.generator import Generator
from dataset import DEMDataset
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from hill_shade import hill_shade
from line_drawer import LineDrawer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_rse(img0, img1):
    mse = (np.abs(img0 - img1) ** 2)
    return np.sqrt(mse).mean()


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


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.):
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
            predicted = self.predicted[i, 0, :, :]
            rse += compute_rse(gt, predicted)
            ssim += compute_ssim(gt, predicted)
        rse /= channels
        ssim /= channels
        print('rse: ', rse)
        print('ssim: ', ssim)
        return rse, ssim


def show_result(ori, dtm, gen_dtm):
    for i in range(show_ori.shape[0]):
        gt = dtm[i, 0, :, :]
        ori = show_ori[i, 0, :, :]
        predicted = gen_dtm[i, 0, :, :]
        fig1, (ax11, ax12, ax13) = plt.subplots(1, 3)
        fig1.suptitle('origin')
        ax11.set_title('ground_truth')
        ax11.imshow(gt, cmap='gray')
        ax12.set_title('predicted')
        ax12.imshow(predicted, cmap='gray')
        ax13.set_title('ori')
        ax13.imshow(ori, cmap='gray')
        plt.show()
        gt_relief = hill_shade(gt, z_factor=1.0)
        predicted_relief = hill_shade(predicted, z_factor=1.0)
        fig2, (ax21, ax22) = plt.subplots(1, 2)
        fig2.suptitle('hill_shade_relief')
        ax21.set_title('gt_relief')
        ax21.imshow(gt_relief, cmap='gray')
        ax22.set_title('predicted_relief')
        ax22.imshow(predicted_relief, cmap='gray')
        plt.show()


def show_profile(drawer):
    if not drawer.line:
        pass
    x = drawer.line.get_xdata()
    y = drawer.line.get_ydata()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    epsilon = 1e-6
    theta = 0
    if dx < epsilon:
        theta = np.pi / 2.
    else:
        theta = np.arctan2(dy, dx)
    t = np.arange(0, 1, 0.01, dtype=np.float32)
    x_sample = np.ceil(x[0] + dx * t * np.cos(theta))
    y_sample = np.ceil(y[0] + dy * t * np.sin(theta))
    xy_sample = np.concatenate((np.reshape(x_sample, (-1, 1)), 
                                np.reshape(y_sample, (-1, 1))), 1)
    img = drawer.ax.get_images()[0].get_array()
    if len(img.shape) == 3:
        img = img[:, :, 0]
    height = img[xy_sample]
    plt.plot(t, height)
    plt.show()

if __name__ == "__main__":
    io.use_plugin('matplotlib', 'imshow')
    model_path = '../checkpoint/gen/gen_model_epoch_200.pth'
    model = Generator().to(device)
    state_dict = torch.load(model_path)['model'].state_dict()
    model.load_state_dict(state_dict, strict=False)

    dataset = DEMDataset('/media/mei/Elements/mini_dataset.hdf5')
    batch_size = 8
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    err = 0.
    for iteration, batch in enumerate(data_loader, 1):
        dtm, ori = batch
        dtm = dtm / 255.
        ori = ori / 255.
        print('dtm max: {}, min: {}'.format(dtm.max(), dtm.min()))
        dtm = torch.unsqueeze(dtm, 1)
        ori = torch.unsqueeze(ori, 1)
        dtm = dtm.numpy()
        show_ori = ori.numpy()
        ori = ori.to(device)

        gen_dtm = model(ori).cpu().detach().numpy()
        # gen_dtm = 100 * np.log10(10 * gen_dtm)
        # gen_dtm /= gen_dtm.max()
        # gen_dtm = np.abs(gen_dtm)
        # gen_dtm = np.where(gen_dtm > 1., 1., gen_dtm)
        # gen_dtm *= 254.
        for i in range(gen_dtm.shape[0]):
            print('gen_dtm {} max: {}, min: {}'.format(i, gen_dtm[i].max(), gen_dtm[i].min()))
            print('gen_dtm {} mean: {}, var: {}'.format(i, gen_dtm[i].mean(), gen_dtm[i].var()))

        val = Validator(dtm, gen_dtm)
        rse, ssim = val.validate()
        print('rse: ', rse)
        print('ssim: ', ssim)
        show_result(show_ori, dtm, gen_dtm)
        print("total: {}; now: {}".format(len(data_loader), iteration * batch_size))
        break
    # err /= batch_size
    # print(err.shape)
    print("mean err: ", err)
