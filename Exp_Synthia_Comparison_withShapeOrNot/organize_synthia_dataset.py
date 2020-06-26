import os
import glob
from shutil import copyfile
synthia_root = '/home/shengjie/Documents/Data/Synthia'
output_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/Synthia_organized'
seqs = glob.glob(os.path.join(synthia_root, '*/'))
dirs = ['Stereo_Left', 'Stereo_Right']
for seq in seqs:
    seq = seq.split('/')[-2]
    if seq == 'SYNTHIA-SF':
        continue
    for dir in dirs:
        rgb_paths = glob.glob(os.path.join(synthia_root, seq, 'RGB', dir, 'Omni_F', '*.png'))
        tocopy_dir_rgb = os.path.join(output_root, seq, 'RGB', dir, 'Omni_F')
        os.makedirs(tocopy_dir_rgb, exist_ok=True)
        for rgb_path in rgb_paths:
            src = rgb_path
            dst = os.path.join(tocopy_dir_rgb, rgb_path.split('/')[-1])
            copyfile(src=src, dst=dst)

        depth_paths = glob.glob(os.path.join(synthia_root, seq, 'Depth', dir, 'Omni_F', '*.png'))
        tocopy_dir_depth = os.path.join(output_root, seq, 'Depth', dir, 'Omni_F')
        os.makedirs(tocopy_dir_depth, exist_ok=True)
        for depth_path in depth_paths:
            src = depth_path
            dst = os.path.join(tocopy_dir_depth, depth_path.split('/')[-1])
            copyfile(src=src, dst=dst)

    print("%s Finished" % seq)

