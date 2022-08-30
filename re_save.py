import os.path
import re

cur_path = os.path.dirname(__file__)
img_path = os.path.join(cur_path, 'dem_image/training_dataset')
img_lists = os.listdir(img_path)
img_lists.sort()

dtm_path = os.path.join(cur_path, 'dem_image/dtm')
ori_path = os.path.join(cur_path, 'dem_image/ori')


def move_to_dir():
    dtm_lists = []
    ori_lists = []
    for file in img_lists:
        if re.search('.tif', file) is not None:
            if re.search('ORI', file) is not None:
                ori_name = file.split('/')[-1]
                ori_lists.append(ori_name)
            elif re.search('DTM', file) is not None:
                dtm_name = file.split('/')[-1]
                dtm_lists.append(dtm_name)

    print(dtm_lists)
    print(ori_lists)

    def mv(p1, p2):
        return "mv {} {}".format(p1, p2)

    for ori in ori_lists:
        ori_file = os.path.join(img_path, ori)
        ori_file_new = os.path.join(ori_path, ori)
        os.system(mv(ori_file, ori_file_new))
    for dtm in dtm_lists:
        dtm_file = os.path.join(img_path, dtm)
        dtm_file_new = os.path.join(dtm_path, dtm)
        os.system(mv(dtm_file, dtm_file_new))


if __name__ == '__main__':
    print(dtm_path)
    print(ori_path)
    if not os.path.exists(dtm_path):
        print('mkdir dtm')
        os.mkdir(dtm_path)

    if not os.path.exists(ori_path):
        print('mkdir ori')
        os.mkdir(ori_path)

    move_to_dir()
