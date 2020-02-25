import glob
import os
import shutil
import numpy as np
import cv2
import random
import copy
def gen_split_quadruplets():
    root_dir = '/media/shengjie/other/Depins/Depins/splits/kitti_seman_mapped2depth'
    split_train_path = os.path.join(root_dir, 'train_files.txt')
    split_val_path = os.path.join(root_dir, 'val_files.txt')

    target_dir = '/media/shengjie/other/Depins/Depins/splits/quadruplets'
    split_train_write_path = os.path.join(target_dir, 'train_files.txt')
    split_val_write_path = os.path.join(target_dir, 'val_files.txt')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(split_train_path) as f:
        train_entries = f.readlines()
    with open(split_val_path) as f:
        val_entries = f.readlines()

    opMap = {'l\n': 'r\n', 'r\n': 'l\n'}
    to_write_list = list()
    for entry in train_entries:
        comps = entry.split(' ')

        entry_cur = comps[0] + ' ' + comps[1].zfill(10) + ' ' + comps[2]
        entry_prev = comps[0] + ' ' + str(int(comps[1]) - 1).zfill(10) + ' ' + comps[2]
        entry_next = comps[0] + ' ' + str(int(comps[1]) + 1).zfill(10) + ' ' + comps[2]
        entry_stereo = comps[0] + ' ' + comps[1].zfill(10) + ' ' + opMap[comps[2]]

        to_write_list.append(entry_cur)
        to_write_list.append(entry_prev)
        to_write_list.append(entry_next)
        to_write_list.append(entry_stereo)
        # if not entry in to_write_list:
        #     comps = entry.split(' ')
        #     if comps[2][0] == 'l':
        #         camid = 2
        #     else:
        #         camid = 3
        #     ind = int(comps[1])
        #     fpath0 = os.path.join(root_dir, comps[0], 'image_0' + str(camid), 'data', str(ind).zfill(10) + '.png')
        #     fpath1 = os.path.join(root_dir, comps[0], 'image_0' + str(camid), 'data', str(ind + 1).zfill(10) + '.png')
        #     fpath2 = os.path.join(root_dir, comps[0], 'image_0' + str(camid), 'data', str(ind + 2).zfill(10) + '.png')
        #
        #     tline1 = comps[0] + ' ' + str(ind) + ' ' + comps[2]
        #     tline2 = comps[0] + ' ' + str(ind + 1) + ' ' + comps[2]
        #     if os.path.exists(fpath0) and os.path.exists(fpath1) and os.path.exists(fpath2) and tline1 in train_entries and tline2 in train_entries:
        #         tline1 = comps[0] + ' ' + str(ind).zfill(10) + ' ' + comps[2]
        #         tline2 = comps[0] + ' ' + str(ind + 1).zfill(10) + ' ' + comps[2]
        #         to_write_list.append(tline1)
        #         to_write_list.append(tline2)

    f = open(split_val_write_path, "w")
    for entry in val_entries:
        f.writelines(entry)

    f = open(split_train_write_path, "w")
    for entry in to_write_list:
        f.writelines(entry)

def generate_visualization_split():
    data_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    sequenceList = {
        '2011_09_26/2011_09_26_drive_0005_sync',
        '2011_09_26/2011_09_26_drive_0011_sync',
        '2011_10_03/2011_10_03_drive_0047_sync'
    }
    save_root = '/media/shengjie/other/Depins/Depins/visualization/monodepth2_3dvisualization'

    entry_list = list()
    for seq in sequenceList:
        imgs = glob.glob(os.path.join(data_root, seq, 'image_02', 'data', '*.png'))
        for i in range(len(imgs)):
            if os.path.isfile(os.path.join(data_root, seq, 'image_02', 'data', str(i).zfill(10) + '.png')):
                entry_list.append(seq + ' ' + str(i).zfill(10) + ' ' + 'l\n')

    wf = open(os.path.join('/media/shengjie/other/Depins/Depins/splits/visualization3d', "train_files.txt"), "w")
    for entry in entry_list:
        wf.write(entry)
    wf.close()

def generate_cycleGan_split():
    data_root = '/media/shengjie/other/Depins/pytorch-CycleGAN-and-pix2pix/datasets/maps'
    sets = ['A', 'B']
    split_types = ['train', 'val', 'test']
    split_root = '../splits/cycleGan_maps'

    for split_type in split_types:
        for set in sets:
            entry_list = list()
            folder_root = os.path.join(data_root, split_type + set)
            imgs = glob.glob(os.path.join(folder_root, '*.jpg'))
            for img_path in imgs:
                comps = img_path.split('/')
                entry_list.append(comps[-2] + '/' + comps[-1])

            wf = open(os.path.join(split_root, split_type + set + '.txt'), "w")
            for entry in entry_list:
                wf.write(entry + '\n')
            wf.close()


def generate_cycleGan_split_visualize():
    data_root = '/media/shengjie/other/Depins/pytorch-CycleGAN-and-pix2pix/datasets/maps'
    sets = ['A']
    split_types = ['train', 'val', 'test']
    split_root = '../splits/cycleGan_maps_visualize'

    for split_type in split_types:
        for set in sets:
            entry_list = list()
            entry_list_B = list()
            folder_root = os.path.join(data_root, split_type + set)
            imgs = glob.glob(os.path.join(folder_root, '*.jpg'))
            for img_path in imgs:
                comps = img_path.split('/')
                entry_list.append(comps[-2] + '/' + comps[-1])
                entry_list_B.append(comps[-2][0:-1] + 'B' + '/' + comps[-1].split('.')[0][0:-1] + 'B.' + comps[-1].split('.')[1])

            wf = open(os.path.join(split_root, split_type + set + '.txt'), "w")
            for entry in entry_list:
                wf.write(entry + '\n')
            wf.close()

            wf = open(os.path.join(split_root, split_type + 'B' + '.txt'), "w")
            for entry in entry_list_B:
                wf.write(entry + '\n')
            wf.close()

def pair_frames():
    vrkitti_root = '/media/shengjie/other/Data/virtual_kitti/vkitti_1.3.1_rgb'
    rawkitti_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    count_list_rawkitti = list()
    count_list_vrkitti = list()
    # list all raw kitti entries
    for date in glob.glob(os.path.join(rawkitti_root, '*/')):
        for seq in glob.glob(os.path.join(date, '*/')):
            img_list = glob.glob(os.path.join(seq, 'image_02', 'data', '*.png'))
            count = 0
            for i in range(0, len(img_list)):
                if os.path.exists(os.path.join(seq, 'image_02', 'data', str(i).zfill(10) + '.png')):
                    count = count + 1
            count_list_rawkitti.append(seq + ": " + str(count))

    for seq in glob.glob(os.path.join(vrkitti_root, '*/')):
        img_list = glob.glob(os.path.join(seq, 'clone', '*.png'))
        count_list_vrkitti.append(seq + ": " + str(len(img_list)))
    print(count_list_vrkitti)
    print("=============================")
    print(count_list_rawkitti)

def generate_raw2vir_kitti():
    raw_seqs = ['/2011_09_26/2011_09_26_drive_0009_sync',
                '/2011_09_26/2011_09_26_drive_0011_sync',
                '/2011_09_26/2011_09_26_drive_0018_sync',
                '/2011_09_29/2011_09_29_drive_0004_sync',
                '/2011_10_03/2011_10_03_drive_0047_sync'
                ]
    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]
    raw_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    vir_root = '/media/shengjie/other/Data/virtual_kitti/vkitti_1.3.1_rgb'


def organize_virtual_kitti():
    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]
    cats = ['rgb', 'depthgt', 'extrinsicsgt', 'scenegt']
    prefix = 'vkitti_1.3.1'

    rootpath = '/media/shengjie/other/Data/virtual_kitti'
    dstpath = '/media/shengjie/other/Data/virtual_kitti_organized'
    for seq in vir_seqs:
        for cat in cats:
            if cat == 'rgb' or cat == 'depthgt' or 'scenegt':
                os.makedirs(os.path.join(dstpath, seq, cat), exist_ok=True)
                srcFiles = glob.glob(os.path.join(rootpath, prefix + '_' + cat, seq, 'clone', '*.png'))

                for srcf in srcFiles:
                    dstf = os.path.join(dstpath, seq, cat, srcf.split('/')[-1])
                    docs = shutil.copyfile(srcf, dstf)
                    print("%s finished" % docs)

                if os.path.exists(os.path.join(rootpath, prefix + '_' + cat, seq + '_clone_' + cat + '_rgb_encoding.txt')):
                    srcf = os.path.join(rootpath, prefix + '_' + cat, seq + '_clone_' + cat + '_rgb_encoding.txt')
                    dstf = os.path.join(dstpath, seq, 'scenegt_rgb_encoding.txt')
                    docs = shutil.copyfile(srcf, dstf)
                    print("%s finished" % docs)

            if cat == 'extrinsicsgt':
                if os.path.exists(os.path.join(rootpath, prefix + '_' + cat, seq + '_clone' + '.txt')):
                    srcf = os.path.join(rootpath, prefix + '_' + cat, seq + '_clone' + '.txt')
                    dstf = os.path.join(dstpath, seq, 'extrinsicsgt.txt')
                    docs = shutil.copyfile(srcf, dstf)
                    print("%s finished" % docs)


def create_kitti_style_real2virtual():
    real_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    vir_root = '/media/shengjie/other/Data/virtual_kitti_organized'
    dst_root = '/media/shengjie/other/Data/kitti_style_real2virtual'
    raw_seqs = ['2011_09_26/2011_09_26_drive_0009_sync',
                '2011_09_26/2011_09_26_drive_0011_sync',
                '2011_09_26/2011_09_26_drive_0018_sync',
                '2011_09_29/2011_09_29_drive_0004_sync',
                '2011_10_03/2011_10_03_drive_0047_sync'
                ]

    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]
    splitA = list()
    splitB = list()

    split_types = ['train', 'val', 'test']
    split_root = '../splits/kitti_real2syn'

    for seq in raw_seqs:
        img_list = glob.glob(os.path.join(real_root, seq, 'image_02', 'data', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(real_root, seq, 'image_02', 'data', str(i).zfill(10) + '.png')):
                splitA.append('trainA/' + seq.split('/')[1] + '_' + str(i).zfill(10) + '.png')
                dstf = os.path.join(dst_root, 'trainA', seq.split('/')[1] + '_' + str(i).zfill(10) + '.png')
                srcf = os.path.join(real_root, seq, 'image_02', 'data', str(i).zfill(10) + '.png')
                docs = shutil.copyfile(srcf, dstf)
                print("%s finished" % docs)

    for seq in vir_seqs:
        img_list = glob.glob(os.path.join(vir_root, seq, 'rgb', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(vir_root, seq, 'rgb', str(i).zfill(5) + '.png')):
                splitB.append('trainB/' + seq + '_' + str(i).zfill(5) + '.png')
                dstf = os.path.join(dst_root, 'trainB', seq + '_' + str(i).zfill(5) + '.png')
                srcf = os.path.join(vir_root, seq, 'rgb', str(i).zfill(5) + '.png')
                docs = shutil.copyfile(srcf, dstf)
                print("%s finished" % docs)


    for split_type in split_types:
        wf = open(os.path.join(split_root, split_type + 'A' + '.txt'), "w")
        for entry in splitA:
            wf.write(entry + '\n')
        wf.close()

        wf = open(os.path.join(split_root, split_type + 'B' + '.txt'), "w")
        for entry in splitB:
            wf.write(entry + '\n')
        wf.close()



def create_sfnorm_split():
    real_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    vir_root = '/media/shengjie/other/Data/virtual_kitti_organized'
    raw_seqs = ['2011_09_26/2011_09_26_drive_0009_sync',
                '2011_09_26/2011_09_26_drive_0011_sync',
                '2011_09_26/2011_09_26_drive_0018_sync',
                '2011_09_29/2011_09_29_drive_0004_sync',
                '2011_10_03/2011_10_03_drive_0047_sync'
                ]

    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]

    splitA = list()
    splitB = list()

    split_types = ['train']
    split_root = '../splits/sfnorm'

    mapping = {0 : 'l', 1 : 'r'}

    for seq in raw_seqs:
        img_list = glob.glob(os.path.join(real_root, seq, 'image_02', 'data', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(real_root, seq, 'image_02', 'data', str(i).zfill(10) + '.png')):
                splitA.append(seq + ' ' + str(i).zfill(10) + ' ' + mapping[np.random.randint(2, size=1)[0]])


    for seq in vir_seqs:
        img_list = glob.glob(os.path.join(vir_root, seq, 'rgb', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(vir_root, seq, 'rgb', str(i).zfill(5) + '.png')):
                splitB.append(seq + ' ' + str(i).zfill(5) + ' ' + 'm')

    # Create train split
    split_type = 'train'
    wf = open(os.path.join(split_root, split_type + 'A' + '.txt'), "w")
    for entry in splitA:
        wf.write(entry + '\n')
    wf.close()

    wf = open(os.path.join(split_root, split_type + 'B' + '.txt'), "w")
    for entry in splitB:
        wf.write(entry + '\n')
    wf.close()

    # Copy test set
    srcf = os.path.join('/media/shengjie/other/Depins/Depins/splits/eigen', 'test_files.txt')
    dstf = os.path.join(split_root, 'test_files.txt')
    docs = shutil.copyfile(srcf, dstf)
    # Create test split
    srcf = os.path.join('/media/shengjie/other/Depins/Depins/splits/eigen', 'test_files.txt')
    with open(srcf, 'r') as f:
        test_entries = f.readlines()

    split_type = 'val'
    wf1 = open(os.path.join(split_root, split_type + 'A' + '.txt'), "w")
    wf2 = open(os.path.join(split_root, split_type + 'B' + '.txt'), "w")
    for idx, entry in enumerate(test_entries):
        wf1.write(entry)
        wf2.write(splitB[idx] + '\n')
    wf1.close()
    wf2.close()


def confirm_projectedGt():
    txt_path = os.path.join('/media/shengjie/other/Depins/Depins/splits/sfnorm', 'trainA.txt')
    data_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/projected_groundtruth'
    with open(txt_path) as f:
        lines = f.readlines()
    mapping = {'l':'image_02', 'r':'image_03'}
    for entry in lines:
        folder, ind, dir = entry.split(' ')
        filepath = os.path.join(data_root, folder, mapping[dir[0]], ind + '.png')
        gt = cv2.imread(filepath, -1)
        if gt is None:
            print("Err")
    print("Evaluation finished")


def create_sfnorm2_split():
    split_types = ['train', 'val']
    split_root = '../splits/sfnorm2'

    mapping = {0 : 'l', 1 : 'r'}

    vir_root = '/media/shengjie/other/Data/virtual_kitti_organized'
    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]

    splitB = list()
    # Copy
    dst_root = os.path.join('..', 'splits', 'sfnorm2')
    copy_root = os.path.join('..', 'splits', 'eigen_zhou')
    copy_list = ['train', 'val']
    for entry in copy_list:
        srcf = os.path.join(copy_root, '{}_files.txt'.format(entry))
        dstf = os.path.join(dst_root, '{}_files.txt'.format(entry))
        shutil.copyfile(srcf, dstf)

    for seq in vir_seqs:
        img_list = glob.glob(os.path.join(vir_root, seq, 'rgb', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(vir_root, seq, 'rgb', str(i).zfill(5) + '.png')):
                splitB.append(seq + ' ' + str(i).zfill(5) + ' ' + 'm' + '\n')

    random.shuffle(splitB)
    split_ind = int(len(splitB) * 0.8)
    split_train = splitB[0:split_ind:]
    split_val = splitB[split_ind::]

    split_type = 'train'
    wf1 = open(os.path.join(split_root, 'syn_{}_files.txt'.format(split_type)), "w")
    for idx, entry in enumerate(split_train):
        wf1.write(entry)
    wf1.close()

    split_type = 'val'
    wf1 = open(os.path.join(split_root, 'syn_{}_files.txt'.format(split_type)), "w")
    for idx, entry in enumerate(split_val):
        wf1.write(entry)
    wf1.close()


def create_sfnorm_pair_split():
    split_types = ['train', 'val']
    split_root = '../splits/sfnorm_pair'

    mapping = {0 : 'l', 1 : 'r'}

    vir_root = '/media/shengjie/other/Data/virtual_kitti_organized'
    vir_seqs = ['0001',
                '0002',
                '0006',
                '0018',
                '0020'
                ]

    real_root = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    real_seqs = ['2011_09_26/2011_09_26_drive_0009_sync',
                 '2011_09_26/2011_09_26_drive_0011_sync',
                 '2011_09_26/2011_09_26_drive_0018_sync',
                 '2011_09_29/2011_09_29_drive_0004_sync',
                 '2011_10_03/2011_10_03_drive_0047_sync'
                ]

    splitA = list()
    splitB = list()
    # Copy
    dst_root = os.path.join('..', 'splits', 'sfnorm_pair')
    copy_root = os.path.join('..', 'splits', 'eigen_zhou')
    copy_list = ['train', 'val']
    for entry in copy_list:
        srcf = os.path.join(copy_root, '{}_files.txt'.format(entry))
        dstf = os.path.join(dst_root, '{}_files.txt'.format(entry))
        shutil.copyfile(srcf, dstf)

    for ii, seq in enumerate(vir_seqs):
        img_list = glob.glob(os.path.join(vir_root, seq, 'rgb', '*.png'))
        for i in range(0, len(img_list)):
            if os.path.exists(os.path.join(real_root, real_seqs[ii], 'velodyne_points', 'data', str(i).zfill(10) + '.bin')):
                if os.path.exists(os.path.join(vir_root, seq, 'rgb', str(i).zfill(5) + '.png')):
                    splitB.append(seq + ' ' + str(i).zfill(5) + ' ' + 'm' + '\n')
                if os.path.exists(os.path.join(real_root, real_seqs[ii], 'image_02', 'data', str(i).zfill(10) + '.png')):
                    splitA.append(real_seqs[ii] + ' ' + str(i).zfill(10) + ' ' + 'l' + '\n')



    split_ind = int(len(splitB) * 0.8)
    split_train = copy.deepcopy(splitB)
    random.shuffle(split_train)
    split_val = split_train[split_ind::] # Validation file consist part of the training

    split_type = 'train'
    wf1 = open(os.path.join(split_root, 'syn_{}_files.txt'.format(split_type)), "w")
    for idx, entry in enumerate(splitB):
        wf1.write(entry)
    wf1.close()

    split_type = 'val'
    wf1 = open(os.path.join(split_root, 'syn_{}_files.txt'.format(split_type)), "w")
    for idx, entry in enumerate(split_val):
        wf1.write(entry)
    wf1.close()

    wf1 = open(os.path.join(split_root, 'train_files.txt'), "w")
    for idx, entry in enumerate(splitA):
        wf1.write(entry)
    wf1.close()

def create_sfnorm_pair_with_pole(opts):
    from datasets_sfgan import SFGAN_Base_Dataset
    from torch.utils.data import DataLoader
    from utils import readlines
    import torch
    from utils import tensor2disp

    fpath = os.path.join(os.path.dirname(__file__), "..", "splits", opts.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    syn_train_filenames = readlines(fpath.format("syn_train"))
    syn_val_filenames = readlines(fpath.format("syn_val"))

    train_dataset = SFGAN_Base_Dataset(
        opts.data_path, train_filenames, syn_train_filenames, opts.height, opts.width,
        opts.frame_ids, 4, opts=opts, is_train=False, load_seman=True)
    train_loader = DataLoader(
        train_dataset, 1, shuffle=not opts.noShuffle,
        num_workers=opts.num_workers, pin_memory=True, drop_last=False)

    min_num = 100
    poleId = 5
    pole_ind_rec = list()
    for batch_idx, inputs in enumerate(train_loader):
        num_syn = torch.sum(inputs['syn_semanLabel'] == poleId)
        num_real = torch.sum(inputs['real_semanLabel'] == poleId)

        if num_syn > min_num and num_real > min_num:
            pole_ind_rec.append(batch_idx)

        print(batch_idx)


    split_root = '../splits/sfnorm_pole'

    wf1 = open(os.path.join(split_root, 'train_files.txt'), "w")
    for pole_ind in pole_ind_rec:
        wf1.write(train_filenames[pole_ind] + '\n')
    wf1.close()

    wf1 = open(os.path.join(split_root, 'syn_train_files.txt'), "w")
    for pole_ind in pole_ind_rec:
        wf1.write(syn_train_filenames[pole_ind] + '\n')
    wf1.close()



def create_sfnorm_pair_with_pole_shuffle(opts):
    from datasets_sfgan import SFGAN_Base_Dataset
    from torch.utils.data import DataLoader
    from utils import readlines
    import torch
    from utils import tensor2disp

    fpath = os.path.join(os.path.dirname(__file__), "..", "splits", opts.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    syn_train_filenames = readlines(fpath.format("syn_train"))
    syn_val_filenames = readlines(fpath.format("syn_val"))

    train_dataset = SFGAN_Base_Dataset(
        opts.data_path, train_filenames, syn_train_filenames, opts.height, opts.width,
        opts.frame_ids, 4, opts=opts, is_train=False, load_seman=True)
    train_loader = DataLoader(
        train_dataset, 1, shuffle=not opts.noShuffle,
        num_workers=opts.num_workers, pin_memory=True, drop_last=False)

    min_num = 100
    poleId = 5
    pole_ind_rec = list()
    maxDepth = 40
    for batch_idx, inputs in enumerate(train_loader):
        num_syn = torch.sum(inputs['syn_semanLabel'] == poleId)
        num_real = torch.sum(inputs['real_semanLabel'] == poleId)

        if num_syn > min_num and num_real > min_num:
            pole_ind_rec.append(batch_idx)

        print(batch_idx)


    split_root = '../splits/sfnorm_pole_shuffle'

    pole_ind_rec1 = copy.deepcopy(pole_ind_rec)
    pole_ind_rec2 = copy.deepcopy(pole_ind_rec)

    random.shuffle(pole_ind_rec1)
    random.shuffle(pole_ind_rec2)

    wf1 = open(os.path.join(split_root, 'train_files.txt'), "w")
    for pole_ind in pole_ind_rec1:
        wf1.write(train_filenames[pole_ind] + '\n')
    wf1.close()

    wf1 = open(os.path.join(split_root, 'syn_train_files.txt'), "w")
    for pole_ind in pole_ind_rec2:
        wf1.write(syn_train_filenames[pole_ind] + '\n')
    wf1.close()

from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()
if __name__ == "__main__":
    create_sfnorm_pair_with_pole_shuffle(opts)
    # create_sfnorm_pair_split()