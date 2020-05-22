import os
import PIL.Image as pil
import cv2
import random
from kitti_utils import *
from utils import *
import matplotlib.pyplot as plt
import time

def cvt_png2depth_PreSIL(tsv_depth):
    maxM = 1000
    sMax = 255 ** 3 - 1

    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:, :, 0] * 255 * 255 + tsv_depth[:, :, 1] * 255 + tsv_depth[:, :, 2]) / sMax * maxM
    return depthIm

PreSIL_root = '/home/shengjie/Documents/Data/PreSIL_organized'
dummyLidarMask = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/train_files.txt'
dummyMaskAddresses = open(dummyLidarMask, 'r').readlines()
dirMapping = {'l': 'image_02', 'r':'image_03'}
xx, yy = np.meshgrid(range(1024), range(448), indexing='xy')
st = time.time()
for index in range(51075):
    seq = int(index / 5000)
    depth_path = os.path.join(PreSIL_root, "{:06d}".format(seq), 'depth', "{:06d}.png".format(index))
    depth = pil.open(depth_path)

    dummyMask = random.choice(dummyMaskAddresses)
    comps = dummyMask.split(' ')
    dummyMask = pil.open(os.path.join('/home/shengjie/Documents/Data/Kitti/filtered_lidar', comps[0], dirMapping[comps[2][0]], comps[1].zfill(10) + '.png'))
    dummyMask = dummyMask.resize([dummyMask.size[0], 448], pil.NEAREST)
    dummyMask = np.array(dummyMask).astype(np.float32) / 256
    dummyMask = dummyMask > 0
    dummyMaskCropped = np.zeros([448, 1024])
    dummyMask = dummyMask[:, 0:1024]
    dummyMaskCropped[448-dummyMask.shape[0]:448, :] = dummyMask
    lidar_depthMap = cvt_png2depth_PreSIL(np.array(depth)) * dummyMaskCropped
    lidar_depthMap_clipped = np.clip(lidar_depthMap, a_max=255, a_min=-1)
    lidar_depthMap_write = (lidar_depthMap_clipped * 256).astype(np.uint16)

    os.makedirs(os.path.join(PreSIL_root, "{:06d}".format(seq), 'prohjlidar'), exist_ok=True)
    cv2.imwrite(os.path.join(PreSIL_root, "{:06d}".format(seq), 'prohjlidar', "{:06d}.png".format(index)), lidar_depthMap_write)
    # testRead = np.array(pil.open(os.path.join(PreSIL_root, "{:06d}".format(seq), 'prohjlidar', "{:06d}.png".format(index)))).astype(np.float32) / 256
    # print(np.abs(testRead - lidar_depthMap_clipped).max())
    # tensor2disp(torch.from_numpy(dummyMask).unsqueeze(0).unsqueeze(0), ind = 0, vmax = 1).show()
    # tensor2disp(torch.from_numpy(dummyMaskCropped).unsqueeze(0).unsqueeze(0), ind=0, vmax=1).show()
    # tensor2disp(torch.from_numpy(lidar_depthMap > 0).unsqueeze(0).unsqueeze(0), ind=0, vmax=1).show()

    if random.randint(1,100) == 50:
        rgb_path = os.path.join(PreSIL_root, "{:06d}".format(seq), 'rgb', "{:06d}.png".format(index))
        rgb = pil.open(rgb_path)
        selectordraw = (lidar_depthMap > 0)
        drawx = xx[selectordraw]
        drawy = yy[selectordraw]
        z = lidar_depthMap_clipped[selectordraw]
        z = 2 / z
        cm = plt.get_cmap('magma')
        z = cm(z)
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
        plt.imshow(rgb)
        ax.axis('equal')
        ax.set_xlim(0, rgb.size[0])  # decreasing time
        ax.set_ylim(rgb.size[1], 0)  # decreasing time
        plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
        plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/presil_projected_lidar_random_vls', str(index).zfill(6) + '.png'))
        plt.close()

    dr = time.time() - st
    indexf = float(index) + 1
    print("%d finished, %f hours to go" % (index, (51075.0 - indexf) * (dr / indexf) / 60 / 60))