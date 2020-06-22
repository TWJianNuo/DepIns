target_semidense_gt_address = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
source_semidense_gt_address = '/home/shengjie/Documents/Data/Kitti/kitti_dense_depth/data_depth_annotated'

from glob import glob
import os
from shutil import copyfile
folds = glob(os.path.join(source_semidense_gt_address, 'train', '*'))
dirs = ['image_02', 'image_03']
count = 0
for fold in folds:
    for dir in dirs:
        imgs = glob(os.path.join(fold, 'proj_depth', 'groundtruth', dir, "*.png"))
        for img in imgs:
            date = fold.split('/')[-1][0:10]
            seq = fold.split('/')[-1]
            imgname = img.split('/')[-1]
            target_folder = os.path.join(target_semidense_gt_address, date, seq, dir)
            os.makedirs(target_folder, exist_ok=True)
            source_path = img
            target_path = os.path.join(target_folder, imgname)
            copyfile(source_path, target_path)
            count = count + 1
            print(count)