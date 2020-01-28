import glob
import os
import shutil
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


if __name__ == "__main__":
    create_kitti_style_real2virtual()
