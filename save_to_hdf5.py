import os.path
import argparse
import h5py
import numpy as np
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='batch size, default to be 64')
parser.add_argument('--validate_dataset', '-v', type=bool, default=False, help='to make a hdf5 validate dataset or not\n'
                                                                         'if yes, input true;\n'
                                                                         'if you are to make a training set, input '
                                                                         'false or let it to be default')
parser.add_argument('--mini-set', '-m', type=bool, default=True)
args = parser.parse_args()

# cur_path = os.path.dirname(__file__)
# or
cur_path = '/media/mei/Elements/'

ori_path = os.path.join(cur_path, 'dem_image/ori')
dtm_path = os.path.join(cur_path, 'dem_image/dtm')

ori_lists = os.listdir(ori_path)
ori_lists.sort()
dtm_lists = os.listdir(dtm_path)
dtm_lists.sort()
f = None
ori_len = None
dtm_len = None

if not args.validate_dataset and not args.mini_set:
    # 制作训练集时
    f = h5py.File('/media/mei/Elements/training_dataset.hdf5', 'w')
    ori_len = len(ori_lists)
    dtm_len = len(dtm_lists)
elif args.validate_dataset:
    # 制作测试集时
    f = h5py.File('/media/mei/Elements/validating_dataset.hdf5', 'w')
    ori_len = 100
    dtm_len = 100
elif args.mini_set:
    # 制作小数据集时
    f = h5py.File('/media/mei/Elements/mini_dataset.hdf5', 'w')
    ori_len = 8
    dtm_len = 8

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
    if img_index >= ori_len:
        break
    dtm_file = os.path.join(dtm_path, file[0])
    ori_file = os.path.join(ori_path, file[1])
    dtm = io.imread(dtm_file, plugin='pil')
    ori = io.imread(ori_file, plugin='pil')[:, :, 0]
    dtm_dataset[img_index, ...] = dtm
    ori_dataset[img_index, ...] = ori
    img_index += 1
    if img_index % 10 == 0:
        print('{}/{}'.format(img_index, ori_len))

print(dtm_grp['dst1'].name)
print(dtm_grp['dst1'].shape)
print(ori_grp['dst1'].name)
print(ori_grp['dst1'].shape)
