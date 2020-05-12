from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch import nn
from torch.autograd import Function
from utils import *
import torch
from kitti_utils import *
import scipy.io as sio
import LidarFilterCuda
import pickle
import time
import cv2
torch.manual_seed(42)


def generate_depth_map_local(calib_dir, velo_filename, cam=2, filterIndicator=None):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    # velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0) & (velo[:, 0] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    if filterIndicator is not None:
        val_inds = val_inds & (filterIndicator == 1)
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


def get_entry_from_path(imgpath):
    comps = imgpath.split('/')
    if comps[-3] == 'image_02':
        direct = 'l'
    else:
        direct = 'r'
    entry = comps[-5] + '/' + comps[-4] + ' ' + comps[-1].split('.')[0] + ' ' + direct + '\n'
    return entry

def collect_all_entries(folder):
    import glob
    # folder = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders = subfolders[::-1]
    entries = list()
    for seqs_clusts in subfolders:
        seqs = [f.path for f in os.scandir(seqs_clusts) if f.is_dir()]
        for seq in seqs:
            imgFolder_02 = os.path.join(seq, 'image_02', 'data')
            imgFolder_03 = os.path.join(seq, 'image_03', 'data')
            for imgpath in glob.glob(imgFolder_02 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
            for imgpath in glob.glob(imgFolder_03 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
    return entries



class LidarFilter(nn.Module):
    def __init__(self, max_depth, h2, verRange, horRange, searchRange, mind1d2, maxd2):
        super(LidarFilter, self).__init__()
        self.lookTables = dict()
        self.max_depth = max_depth
        self.h2 = h2
        self.verRange = verRange
        self.horRange = horRange
        self.searchRange = searchRange
        self.mind1d2 = mind1d2
        self.maxd2 = maxd2

        # sxx, syy = np.meshgrid(list(range(-searchRange, searchRange + 1)), list(range(-searchRange, searchRange + 1)))
        # sxx = sxx.flatten().astype(np.int)
        # syy = syy.flatten().astype(np.int)
        # tmps = (sxx == 0) * (syy == 0)
        # tmps = (1 - tmps) == 1
        # sxx = sxx[tmps]
        # syy = syy[tmps]
        # self.itnum = len(sxx)
        # self.sxx = torch.from_numpy(sxx).float().cuda()
        # self.syy = torch.from_numpy(syy).float().cuda()

    def get_lookupkey(self, intrinsic, camExtrinsic, lidarExtrinsic):
        key = torch.sum(torch.abs(intrinsic @ camExtrinsic)) + torch.sum(torch.abs(intrinsic @ lidarExtrinsic))
        key = str(float(key.cpu().numpy()))
        return key
    def init_lookupTable(self, intrinsic, camExtrinsic, lidarExtrinsic, h, w):
        lookTable = torch.zeros([h2, w, self.max_depth + 1, 2]).float().cuda()

        lidarPos = torch.inverse(lidarExtrinsic) @ torch.Tensor([[0,0,0,1]]).float().cuda().t()
        epp = intrinsic @ camExtrinsic @ lidarPos
        epp[0] = epp[0] / epp[2]
        epp[1] = epp[1] / epp[2]
        epp = epp[0:2,0]
        lookTable, = LidarFilterCuda.init_lookupTable(lookTable, epp, int(w), int(h), int(self.h2), self.searchRange, self.max_depth, self.verRange, self.horRange)
        self.lookTables[self.get_lookupkey(intrinsic, camExtrinsic, lidarExtrinsic)] = lookTable
        print("lookTables Stored")
        # print(lookTable[:, :, :, 0].max())
        # print(torch.unique(lookTable[:,:,0,0]))
        #
        # yy = 300
        # xx = 800
        # fig, ax = plt.subplots()
        # plt.scatter([xx], [yy], s = 20, c = 'r')
        # num = int(lookTable[yy, xx, 0, 0].cpu().numpy())
        # vxx = lookTable[yy, xx, 1 : num + 1, 0].cpu().numpy()
        # vyy = lookTable[yy, xx, 1: num + 1, 1].cpu().numpy()
        # plt.scatter(vxx, vyy, s = 20, c = 'g')
        # plt.plot([xx, float(epp[0].cpu().numpy())], [yy, float(epp[1].cpu().numpy())], c='r')
        # ax.set_xlim(0, w)  # decreasing time
        # ax.set_ylim(h2, 0)  # decreasing time
        # ax.axis('equal')
    def denoise_lidar(self, velo, intrinsic, camExtrinsic, lidarExtrinsic, interp_depth, h, w):
        camExtrinsic = torch.from_numpy(camExtrinsic).float().cuda()
        intrinsic = torch.from_numpy(intrinsic).float().cuda()
        lidarExtrinsic = torch.from_numpy(lidarExtrinsic).float().cuda()
        interp_depth = torch.from_numpy(interp_depth).float().cuda()

        lidarPos = torch.inverse(lidarExtrinsic) @ torch.Tensor([[0,0,0,1]]).float().cuda().t()
        epp = intrinsic @ camExtrinsic @ lidarPos
        epp[0] = epp[0] / epp[2]
        epp[1] = epp[1] / epp[2]
        epp = epp[0:2,0]

        camPos = torch.inverse(camExtrinsic) @ torch.Tensor([[0, 0, 0, 1]]).float().cuda().t()
        epp2 = intrinsic @ lidarExtrinsic @ camPos
        epp2[0] = epp2[0] / epp2[2]
        epp2[1] = epp2[1] / epp2[2]
        epp2 = epp2[0:2,0]
        # lookuptable = pickle.load(open("lookuptable.p", "rb"))
        # self.lookTables[self.get_lookupkey(intrinsic, camExtrinsic, lidarExtrinsic)] = lookuptable
        if self.get_lookupkey(intrinsic, camExtrinsic, lidarExtrinsic) not in self.lookTables:
            self.init_lookupTable(intrinsic, camExtrinsic, lidarExtrinsic, h, w)

        lookuptable = self.lookTables[self.get_lookupkey(intrinsic, camExtrinsic, lidarExtrinsic)]

        velo = torch.from_numpy(velo).float().cuda()
        velo_projected = intrinsic @ camExtrinsic @ velo.t()
        velo_projected = velo_projected.t()
        velo_projected[:, 0] = velo_projected[:, 0] / velo_projected[:, 2]
        velo_projected[:, 1] = velo_projected[:, 1] / velo_projected[:, 2]
        onimgSelector = (velo[:, 0] > 0) * (velo_projected[:, 0] > 0) * (velo_projected[:, 0] < w) * (velo_projected[:, 1] > 0) * (velo_projected[:, 1] < h)

        nvelo_projected = intrinsic @ lidarExtrinsic @ velo.t()
        nvelo_projected = nvelo_projected.t()
        nvelo_projected[:, 0] = nvelo_projected[:, 0] / nvelo_projected[:, 2]
        nvelo_projected[:, 1] = nvelo_projected[:, 1] / nvelo_projected[:, 2]

        nvelo_projected_img = torch.zeros([h2, w, 7]).float().cuda()
        rnx = torch.round(nvelo_projected[:, 0]).long()
        rny = torch.round(nvelo_projected[:, 1]).long()
        rnd = nvelo_projected[:, 2]
        nval = (rnx >= 0) * (rnx < w) * (rny >= 0) * (rny < h2) * (rnd > 0) * onimgSelector
        indicesrec = torch.Tensor(list(range(len(nval)))).cuda()


        xxx, yyy = np.meshgrid(range(w), range(self.h2), indexing='xy')
        interp_sel = interp_depth > 0

        xxx = torch.from_numpy(xxx).cuda()[interp_sel]
        yyy = torch.from_numpy(yyy).cuda()[interp_sel]
        ddd = interp_depth[interp_sel]
        interped_pts = torch.stack([xxx.float() * ddd, yyy.float() * ddd, ddd, torch.ones_like(ddd)], dim=1)
        interped_pts_projected = intrinsic @ camExtrinsic @ torch.inverse(intrinsic @ lidarExtrinsic) @ interped_pts.t()
        interped_pts_projected = interped_pts_projected.t()
        interped_pts_projected[:, 0] = interped_pts_projected[:, 0] / interped_pts_projected[:, 2]
        interped_pts_projected[:, 1] = interped_pts_projected[:, 1] / interped_pts_projected[:, 2]

        nvelo_projected_img[yyy, xxx, 0:6] = torch.cat([torch.stack([xxx.float(), yyy.float(), ddd], dim=1), interped_pts_projected[:,0:3]], dim=1)
        # tensor2disp(torch.from_numpy(interp_depth).unsqueeze(0).unsqueeze(0) > 0, vmax=1, ind=0).show()
        # tensor2disp((nvelo_projected_img[:,:,5]).unsqueeze(0).unsqueeze(0) > 0, vmax=1, ind=0).show()


        nvelo_projected_img[rny[nval], rnx[nval], :] = torch.cat([nvelo_projected[nval, 0:3], velo_projected[nval, 0:3], indicesrec[nval].unsqueeze(1)], dim=1)
        # diff = torch.abs(interp_depth[rny[nval], rnx[nval]] - nvelo_projected[nval, 2])
        # tensor2disp(interp_depth.unsqueeze(0).unsqueeze(0), vmax=40, ind=0).show()

        noocc_mask = torch.zeros([h2, w]).float().cuda()
        noocc_mask[rny[nval], rnx[nval]] = 1

        org_mask = torch.zeros([h2, w]).float().cuda()
        org_mask[rny[nval], rnx[nval]] = 1

        noocc_mask, = LidarFilterCuda.lidar_denoise(nvelo_projected_img, lookuptable, noocc_mask, epp, w, h2, self.mind1d2, self.maxd2)

        filterIndicator = torch.zeros_like(onimgSelector)
        filterIndicator[nvelo_projected_img[:,:,6][noocc_mask == 1].long()] = 1
        # filterIndicator = onimgSelector.float() * (1 - filterIndicator.float())
        filterIndicator = filterIndicator.cpu().numpy().astype(np.uint8)
        # tensor2disp(noocc_mask.unsqueeze(0).unsqueeze(0), vmax = 1, ind = 0).show()
        # tensor2disp(org_mask.unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

        # drawx = nvelo_projected_img[:, :, 0][org_mask == 1].cpu().numpy()
        # drawy = nvelo_projected_img[:, :, 1][org_mask == 1].cpu().numpy()
        # z = nvelo_projected_img[:, :, 2][org_mask == 1].cpu().numpy()
        # # drawx = velo_projected[onimgSelector,0].cpu().numpy()
        # # drawy = velo_projected[onimgSelector, 1].cpu().numpy()
        # # z = velo_projected[onimgSelector, 2].cpu().numpy()
        # z = z / 40
        # cm = plt.get_cmap('magma')
        # z = cm(z)
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, w)  # decreasing time
        # ax.set_ylim(h2, 0)  # decreasing time
        # ax.axis('equal')
        # plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
        # #
        # drawx = nvelo_projected_img[:, :, 3][noocc_mask == 1].cpu().numpy()
        # drawy = nvelo_projected_img[:, :, 4][noocc_mask == 1].cpu().numpy()
        # z = nvelo_projected_img[:, :, 2][noocc_mask == 1].cpu().numpy()
        # z = z / 40
        # cm = plt.get_cmap('magma')
        # z = cm(z)
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, w)  # decreasing time
        # ax.set_ylim(h2, 0)  # decreasing time
        # ax.axis('equal')
        # plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
        # #
        # #
        # #
        # #
        # #
        # # ch = 180
        # # cw = 1075
        # # ch = 181
        # # cw = 1083
        # ch = 195
        # cw = 977
        # ch = 275
        # cw = 1048
        #
        #
        # reffed_pts = list()
        # reffed_pts_camCoord = list()
        # reffed_pts_occ = list()
        # if org_mask[ch][cw] > 1e-3:
        #     refx = nvelo_projected_img[ch][cw][3]
        #     refy = nvelo_projected_img[ch][cw][4]
        #
        #     lrefx = nvelo_projected_img[ch][cw][0]
        #     lrefy = nvelo_projected_img[ch][cw][1]
        #     for j in range(int(lookuptable[ch][cw][0][0].cpu().numpy())):
        #         xx = lookuptable[ch][cw][j+1][0].long()
        #         yy = lookuptable[ch][cw][j+1][1].long()
        #
        #         reffed_pts.append(np.array([float(nvelo_projected_img[yy][xx][0].cpu().numpy()), float(nvelo_projected_img[yy][xx][1].cpu().numpy())]))
        #         reffed_pts_camCoord.append(np.array([float(nvelo_projected_img[yy][xx][3].cpu().numpy()), float(nvelo_projected_img[yy][xx][4].cpu().numpy())]))
        #
        #
        #         distance2 = ((nvelo_projected_img[yy][xx][3] - refx) * (epp[0] - refx) + (nvelo_projected_img[yy][xx][4] - refy) * (epp[1] - refy)) / \
        #                     (torch.sqrt((epp[0] - refx)**2 + (epp[1] - refy)**2))
        #         distance1 = ((nvelo_projected_img[yy][xx][0] - lrefx) * (epp[0] - lrefx) + (nvelo_projected_img[yy][xx][1] - lrefy) * (epp[1] - lrefy)) / \
        #                     (torch.sqrt((epp[0] - lrefx)**2 + (epp[1] - lrefy)**2))
        #         # print("distance1:%f, distance2:%f" % (distance1, distance2))
        #         if distance1 > 0 and distance2 < 0 and distance1 - distance2 > self.mind1d2 and torch.abs(distance2) < self.maxd2:
        #         # if distance2 < 0:
        #         #     print(distance1 - distance2)
        #             print("Occluded: %d, %d" % (xx, yy))
        #             reffed_pts_occ.append(np.array([float(nvelo_projected_img[yy][xx][3].cpu().numpy()), float(nvelo_projected_img[yy][xx][4].cpu().numpy())]))
        #         # mul1 = (nvelo_projected_img[yy][xx][3] - refx) * (epp[0] - refx) + (nvelo_projected_img[yy][xx][4] - refy) * (epp[1] - refy)
        #         # mul2 = (nvelo_projected_img[yy][xx][3] - epp[0]) * (refx - epp[0]) + (nvelo_projected_img[yy][xx][4] - epp[1]) * (refy - epp[1])
        #         # if ((mul1 < 0) or (mul2 < 0)):
        #         #     print("Occluded: %d, %d" % (xx, yy))
        #         #     reffed_pts_occ.append(np.array([float(nvelo_projected_img[yy][xx][3].cpu().numpy()), float(nvelo_projected_img[yy][xx][4].cpu().numpy())]))
        # xx = cw
        # yy = ch
        #
        # drawx = nvelo_projected_img[:, :, 0][org_mask == 1].cpu().numpy()
        # drawy = nvelo_projected_img[:, :, 1][org_mask == 1].cpu().numpy()
        # z = nvelo_projected_img[:, :, 2][org_mask == 1].cpu().numpy()
        # z = z / 40
        # cm = plt.get_cmap('magma')
        # z = cm(z)
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, w)  # decreasing time
        # ax.set_ylim(h2, 0)  # decreasing time
        # ax.axis('equal')
        # plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
        # plt.scatter([float(nvelo_projected_img[yy][xx][0].cpu().numpy())], [float(nvelo_projected_img[yy][xx][1].cpu().numpy())], s = 20, c = 'r')
        # reffed_pts = np.stack(reffed_pts, axis=0)
        # plt.scatter(reffed_pts[:,0], reffed_pts[:,1], s = 20, c = 'g')
        #
        #
        # drawx = nvelo_projected_img[:, :, 3][org_mask == 1].cpu().numpy()
        # drawy = nvelo_projected_img[:, :, 4][org_mask == 1].cpu().numpy()
        # z = nvelo_projected_img[:, :, 2][org_mask == 1].cpu().numpy()
        # z = z / 40
        # cm = plt.get_cmap('magma')
        # z = cm(z)
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, w)  # decreasing time
        # ax.set_ylim(h2, 0)  # decreasing time
        # ax.axis('equal')
        # plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
        # plt.scatter([float(nvelo_projected_img[yy][xx][3].cpu().numpy())], [float(nvelo_projected_img[yy][xx][4].cpu().numpy())], s = 20, c = 'r')
        # reffed_pts_camCoord = np.stack(reffed_pts_camCoord, axis=0)
        # plt.scatter(reffed_pts_camCoord[:,0], reffed_pts_camCoord[:,1], s = 20, c = 'g')
        # reffed_pts_occ = np.stack(reffed_pts_occ, axis=0)
        # plt.scatter(reffed_pts_occ[:,0], reffed_pts_occ[:,1], s = 20, c = 'b')
        #
        #
        # sx = 0
        # sy = 0
        # dist = 1e10
        # # tarx = 962.3
        # # tary = 172.6
        # tarx = 963.6
        # tary = 174.4
        # for i in range(940, 990):
        #     for j in range(160, 200):
        #         if org_mask[j, i] > 0:
        #             if torch.sqrt((nvelo_projected_img[j, i, 3] - tarx)**2 + (nvelo_projected_img[j, i, 4] - tary)**2) < dist:
        #                 sx = i
        #                 sy = j
        #                 dist = torch.sqrt((nvelo_projected_img[j, i, 3] - tarx)**2 + (nvelo_projected_img[j, i, 4] - tary)**2)
        return nvelo_projected_img, noocc_mask, org_mask, filterIndicator

# lrMapping = {2:'image_02', 3:'image_03'}
# root_path = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
# dayc = '2011_09_26'
# seqc = '2011_09_26_drive_0005_sync'
# frameind = 10
# cam = 3
#
# # Read RGB
# rgb_path = os.path.join(root_path, dayc, seqc, lrMapping[cam], 'data', str(frameind).zfill(10) + '.png')
# rgb = pil.open(rgb_path)
# w = rgb.size[0]
# h = rgb.size[1]
#
# # Read Lidar
# velo_filename = os.path.join(root_path, dayc, seqc, 'velodyne_points', 'data', "{:010d}.bin".format(frameind))
# velo = load_velodyne_points(velo_filename)
#
# # Read Intrinsic
# cam2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_cam_to_cam.txt'))
# velo2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_velo_to_cam.txt'))
# velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
# velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
#
# R_cam2rect = np.eye(4)
# R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
# P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
#
#
# intrinsic = np.eye(4)
# intrinsic[0:3,0:3] = P_rect[0:3,0:3]
# P_rect_ex = np.eye(4)
# P_rect_ex[0:3,:] = P_rect
# tmp_ex = np.linalg.inv(intrinsic) @ P_rect_ex
# extrinsic = tmp_ex @ R_cam2rect @ velo2cam
# siol = sio.loadmat(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/raytrace_ptc/matlab', 'nextrinsic.mat'))
# nextrinsic = np.array(siol['nextrinsic'])
#


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path', type=str)
parser.add_argument('--vls_path', type=str)
parser.add_argument('--depth_path', type=str)
parser.add_argument('--interpolated_root', type=str)
parser.add_argument('--day_seq', type=str)
parser.add_argument('--nextrinsic_path', type=str)
args = parser.parse_args()


# root_path = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
# vls_path = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/lidar_vls_random_new'
# depth_path = '/home/shengjie/Documents/Data/Kitti/kitti_filteredGt'
# interpolated_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/interpolated_depth'
# kitti_dense_gt = '/home/shengjie/Documents/Data/Kitti/kitti_gtDepthMap'

root_path = args.root_path
vls_path = args.vls_path
depth_path = args.depth_path
interpolated_root = args.interpolated_root
lines = collect_all_entries(args.root_path)
print("Exporting ground truth depths")

lrMapping = {'l': 'image_02', 'r': 'image_03'}
mapping_cam = {'l': 2, 'r': 3}

h2 = 450 # To store lidar projeciton in scanner view
max_depth = 500
verRange = 100
horRange = 3
searchRange = 55
# mind1d2 = 2
mind1d2 = 1.5
# maxd2 = 30
maxd2 = 10
lidarfilter = LidarFilter(max_depth = max_depth, h2 = h2, verRange = verRange, horRange = horRange, searchRange = searchRange, mind1d2 = mind1d2, maxd2 = maxd2)

st = time.time()
imgCount = 0
startind = 0

import random
from datetime import datetime
random.seed(datetime.now())
candidate_date = ['2011_09_26', '2011_10_03']
args.day_seq = random.choice(candidate_date)
random.shuffle(lines)
for kk in range(startind, len(lines)):
    line = lines[kk]
    folder, frameind, direction = line.split()

    # frameind = 2
    # direction = 'r'
    # folder = '2011_09_28/2011_09_28_drive_0002_sync'

    # frameind = 302
    # direction = 'l'
    # folder = '2011_09_26/2011_09_26_drive_0093_sync'
    #
    # frameind = 352
    # direction = 'r'
    # folder = '2011_09_28/2011_09_28_drive_0002_sync'

    # '2011_09_28_drive_0002_sync_352_r'
    frameind = int(frameind)
    dayc, seqc = folder.split('/')
    cam = mapping_cam[direction]

    if dayc not in [args.day_seq]:
        continue
    if os.path.isfile(os.path.join(os.path.join(depth_path, folder, lrMapping[direction]), str(frameind).zfill(10) + '.png')):
        continue
    velo_filename = os.path.join(root_path, dayc, seqc, 'velodyne_points', 'data', "{:010d}.bin".format(frameind))
    if not os.path.isfile(velo_filename):
        continue
    velo = load_velodyne_points(velo_filename)

    cam2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    h, w = cam2cam["S_rect_02"][::-1].astype(np.int32)

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = P_rect[0:3, 0:3]
    P_rect_ex = np.eye(4)
    P_rect_ex[0:3, :] = P_rect
    tmp_ex = np.linalg.inv(intrinsic) @ P_rect_ex
    extrinsic = tmp_ex @ R_cam2rect @ velo2cam
    # siol = sio.loadmat(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/raytrace_ptc/matlab', 'nextrinsic.mat'))
    siol = sio.loadmat(args.nextrinsic_path)
    nextrinsic = np.array(siol['nextrinsic'])

    interp_depth = pil.open(os.path.join(interpolated_root, dayc, seqc, "{:010d}.png".format(frameind)))
    interp_depth = np.array(interp_depth).astype(np.float32) / 256
    # tensor2disp(torch.from_numpy(interp_depth / 80).unsqueeze(0).unsqueeze(0), ind = 0, percentile=80).show()


    nvelo_projected_img, noocc_mask, org_mask, filterIndicator = lidarfilter.denoise_lidar(velo=velo, intrinsic=intrinsic, camExtrinsic=extrinsic, lidarExtrinsic=nextrinsic, interp_depth = interp_depth, h=h, w=w)

    nvelo_projected_img_np = nvelo_projected_img.cpu().numpy()
    noocc_mask_np = noocc_mask.cpu().numpy()
    org_mask_np = org_mask.cpu().numpy()
    #
    #
    # depthmapGt = np.zeros([h,w])
    # camsel = org_mask_np == 1
    # camx = nvelo_projected_img_np[:, :, 3][camsel]
    # camy = nvelo_projected_img_np[:, :, 4][camsel]
    # camd = nvelo_projected_img_np[:, :, 5][camsel]
    #
    # camx = np.round(camx) - 1
    # camy = np.round(camy) - 1
    # camsel = (camx >= 0) * (camy >= 0) * (camx < w) * (camy < h)
    # camx = camx[camsel]
    # camy = camy[camsel]
    # camd = camd[camsel]
    #
    # depthmapGt[camy.astype(np.int), camx.astype(np.int)] = camd
    #
    # # find the duplicate points and choose the closest depth
    # inds = sub2ind(depthmapGt.shape, camy, camx)
    # dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    # for dd in dupe_inds:
    #     pts = np.where(inds == dd)[0]
    #     x_loc = int(camx[pts[0]])
    #     y_loc = int(camy[pts[0]])
    #     depthmapGt[y_loc, x_loc] = camd[pts].min()
    # depthmapGt[depthmapGt < 0] = 0

    depthmap_ref = generate_depth_map_local(os.path.join(root_path, dayc), os.path.join(velo_filename), cam=mapping_cam[direction], filterIndicator = filterIndicator)
    # # depthmap_ref2 = generate_depth_map(os.path.join(root_path, dayc), os.path.join(velo_filename), cam=mapping_cam[direction], vel_depth = False)
    #
    # # diff = np.abs(depthmap_ref - depthmap_ref2)
    # # tensor2disp(torch.from_numpy(diff).unsqueeze(0).unsqueeze(0), vmax = 1, ind=0).show()
    # # tensor2disp(torch.from_numpy(depthmap_ref).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()
    # # tensor2disp(torch.from_numpy(depthmap_ref2).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()
    #
    depthmap_ref = np.uint16(depthmap_ref * 256)
    output_folder = os.path.join(depth_path, folder, lrMapping[direction])
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, str(frameind).zfill(10) + '.png')
    cv2.imwrite(save_path, depthmap_ref)
    #
    vls_path_fold = os.path.join(vls_path, folder, lrMapping[direction])
    os.makedirs(vls_path_fold, exist_ok=True)
    vls_path_png1 = os.path.join(vls_path_fold, str(frameind).zfill(10) + '_1.png')
    vls_path_png2 = os.path.join(vls_path_fold, str(frameind).zfill(10) + '_2.png')
    vls_path_png3 = os.path.join(vls_path_fold, str(frameind).zfill(10) + '_3.png')
    #
    # cm = plt.get_cmap('magma')
    # drawx = nvelo_projected_img_np[:, :, 3][org_mask_np == 1]
    # drawy = nvelo_projected_img_np[:, :, 4][org_mask_np == 1]
    # z = nvelo_projected_img_np[:, :, 2][org_mask_np == 1]
    # z = z / 40
    # z = cm(z)
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
    # ax.set_xlim(0, w)  # decreasing time
    # ax.set_ylim(h2, 0)  # decreasing time
    # plt.savefig(vls_path_png1)
    # plt.close()

    # vls_path_png2 = os.path.join(vls_path, folder.split('/')[1]+'_'+str(frameind)+'_'+direction + '.png')
    drawx = nvelo_projected_img_np[:, :, 3][noocc_mask_np == 1]
    drawy = nvelo_projected_img_np[:, :, 4][noocc_mask_np == 1]
    z = nvelo_projected_img_np[:, :, 2][noocc_mask_np == 1]
    z = z / 40
    cm = plt.get_cmap('magma')
    z = cm(z)
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    # fig, ax = plt.subplots()
    ax.axis('equal')
    plt.scatter(drawx, drawy, s=5, marker='.', c=z[:, 0:3])
    ax.set_xlim(0, w)  # decreasing time
    ax.set_ylim(h2, 0)  # decreasing time
    plt.savefig(vls_path_png2)
    plt.close()

    # kitti_semidense_gt = pil.open(os.path.join(kitti_dense_gt, dayc, seqc, lrMapping[direction], "{:010d}.png".format(frameind)))
    # kitti_semidense_gt = np.array(kitti_semidense_gt).astype(np.float32) / 256
    # valsel = kitti_semidense_gt > 0
    # densexx, denseyy = np.meshgrid(range(kitti_semidense_gt.shape[1]), range(kitti_semidense_gt.shape[0]))
    # densexx = densexx[valsel]
    # denseyy = denseyy[valsel]
    # densed = kitti_semidense_gt[valsel]
    # densed = densed / 40
    # z = cm(densed)
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # plt.scatter(densexx, denseyy, s=5, marker='.', c=z[:, 0:3])
    # ax.set_xlim(0, w)  # decreasing time
    # ax.set_ylim(h2, 0)  # decreasing time
    # plt.savefig(vls_path_png3)
    # plt.close()

    # kitti_semidense_gt = pil.open(os.path.join(kitti_dense_gt, dayc, seqc, lrMapping[direction], "{:010d}.png".format(frameind)))
    # kitti_semidense_gt = np.array(kitti_semidense_gt).astype(np.float32) / 256
    # valsel = kitti_semidense_gt > 0
    # densexx, denseyy = np.meshgrid(range(kitti_semidense_gt.shape[1]), range(kitti_semidense_gt.shape[0]))
    # densexx = densexx[valsel]
    # denseyy = denseyy[valsel]
    # densed = kitti_semidense_gt[valsel]
    # densed = densed / 40
    # z = cm(densed)
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # plt.scatter(densexx, denseyy, s=5, marker='.', c=z[:, 0:3])
    # ax.set_xlim(0, w)  # decreasing time
    # ax.set_ylim(h2, 0)

    dr = time.time() - st
    imgCount = imgCount + 1
    # print("%d finished, %f hours left" % (kk, (dr) / imgCount * (len(lines) - imgCount - startind) / 60 / 60))
    print("%s finished, %f hours left" % (os.path.join(os.path.join(folder, lrMapping[direction]), str(frameind).zfill(10) + '.png'), ((dr) / imgCount * (len(lines) - imgCount - startind) / 60 / 60)))