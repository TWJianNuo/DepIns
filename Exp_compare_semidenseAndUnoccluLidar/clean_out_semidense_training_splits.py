import os
splits_source = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full'
splits_target = '/home/shengjie/Documents/Project_SemanticDepth/splits/semidense_eigen_full'
splits = ['train_files.txt', 'val_files.txt']
semidense_gt_address = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
dir_mapping = {'l':'image_02', 'r':'image_03'}
for split in splits:
    valid_list = list()
    with open(os.path.join(splits_source, split)) as f:
        lines = f.readlines()
        for line in lines:
            seq, frame, dir = line.split(' ')
            dir = dir[0]
            ck_address = os.path.join(semidense_gt_address, seq, dir_mapping[dir], frame.zfill(10) + '.png')
            if os.path.isfile(ck_address):
                valid_list.append(line)
        wtfile = open(os.path.join(splits_target, split), "w")
        for valline in valid_list:
            wtfile.write(valline)
        wtfile.close()



