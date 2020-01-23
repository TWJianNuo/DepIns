import glob
import os

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
                entry_list.append(img_path)

            wf = open(os.path.join(split_root, split_type + set + '.txt'), "w")
            for entry in entry_list:
                wf.write(entry + '\n')
            wf.close()


if __name__ == "__main__":
    generate_cycleGan_split()
