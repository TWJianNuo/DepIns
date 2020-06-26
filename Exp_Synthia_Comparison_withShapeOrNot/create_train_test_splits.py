import os
import glob
from shutil import copyfile
synthia_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/Synthia_organized'
seqs = glob.glob(os.path.join(synthia_root, '*/'))
dirs = ['Stereo_Left', 'Stereo_Right']
dir_mapping = {'Stereo_Left' : 'l', 'Stereo_Right' : 'r'}

tot_files = list()
for seq in seqs:
    seq = seq.split('/')[-2]
    for dir in dirs:
        rgb_paths = glob.glob(os.path.join(synthia_root, seq, 'RGB', dir, 'Omni_F', '*.png'))
        depth_paths = glob.glob(os.path.join(synthia_root, seq, 'Depth', dir, 'Omni_F', '*.png'))
        for rgb_path in rgb_paths:
            comps = rgb_path.split('/')
            cur_file_name = os.path.join(comps[6], "***", comps[9]) + ' ' + comps[10].split('.')[0] + ' ' + dir_mapping[comps[8]]
            tot_files.append(cur_file_name)


import random
random.shuffle(tot_files)

train_split = tot_files[0 : int(len(tot_files) * 0.7)]
test_split = tot_files[int(len(tot_files) * 0.7)::]

split_root = '/home/shengjie/Documents/Project_SemanticDepth/splits/synthia_splits'
os.makedirs(split_root, exist_ok=True)
train_split_txt = open(os.path.join(split_root, 'train_files.txt'),"w")
eval_split_txt = open(os.path.join(split_root, 'val_files.txt'),"w")
test_split_txt = open(os.path.join(split_root, 'test_files.txt'),"w")
for train_file in train_split:
    train_split_txt.write(train_file + '\n')
train_split_txt.close()
for eval_file in train_split:
    eval_split_txt.write(eval_file + '\n')
eval_split_txt.close()
for test_file in test_split:
    test_split_txt.write(test_file + '\n')
test_split_txt.close()