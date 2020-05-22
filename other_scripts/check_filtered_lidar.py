import os
split_file1 = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/train_files.txt'
split_file2 = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/val_files.txt'
files = open(split_file1, 'r').readlines() + open(split_file2, 'r').readlines()
sideMapping = {'l':'image_02', 'r':'image_03'}
for file in files:
    comps = file.split(' ')
    lidarfile = os.path.join('/home/shengjie/Documents/Data/Kitti/filtered_lidar', comps[0], sideMapping[comps[2][0]], comps[1].zfill(10) + '.png')
    if not os.path.isfile(lidarfile):
        print("Missing!")