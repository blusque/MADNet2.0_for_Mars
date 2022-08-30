import os.path
import argparse
import h5py
import numpy as np
from skimage import io
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size, default to be 64')
args = parser.parse_args()

f = h5py.File('training_dataset.hdf5', 'w')
cur_path = os.path.dirname(__file__)
ori_path = os.path.join(cur_path, 'dem_image/ori')
dtm_path = os.path.join(cur_path, 'dem_image/dtm')

ori_lists = os.listdir(ori_path)
ori_lists.sort()
dtm_lists = os.listdir(dtm_path)
dtm_lists.sort()
ori_len = len(ori_lists)
dtm_len = len(dtm_lists)

dtm_grp = f.create_group('dtm_grp')
ori_grp = f.create_group('ori_grp')

dtm_dataset_size = (dtm_len, 512, 512)
ori_dataset_size = (ori_len, 512, 512)
dtm_dataset = dtm_grp.create_dataset('dst1', dtm_dataset_size, np.float32)
ori_dataset = ori_grp.create_dataset('dst1', ori_dataset_size, np.float32)
# dtm_dataset = np.empty((batch_nums, batch_size, 512, 512), dtype=np.float32)
# ori_dataset = np.empty((batch_nums, batch_size, 512, 512), dtype=np.float32)
img_index = 0
for file in zip(dtm_lists, ori_lists):
    dtm_file = os.path.join(dtm_path, file[0])
    ori_file = os.path.join(ori_path, file[1])
    dtm = io.imread(dtm_file, plugin='pil')
    ori = io.imread(ori_file, plugin='pil')
    ori = cv.cvtColor(ori, cv.COLOR_RGB2GRAY)
    dtm_dataset[img_index, ...] = dtm
    ori_dataset[img_index, ...] = ori
    img_index += 1
    if img_index % 100 == 0:
        print('{}/{}'.format(img_index, ori_len))

print(dtm_grp['dst1'].name)
print(dtm_grp['dst1'].shape)
print(ori_grp['dst1'].name)
print(ori_grp['dst1'].shape)
