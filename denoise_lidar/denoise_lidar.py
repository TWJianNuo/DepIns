# Read Lidar
# Get the Intrinsic and Extrinsic Camera
# Compute the Eppipolar Line And the Movement distance
import os
import PIL.Image as pil
from kitti_utils import *
import matplotlib.pyplot as plt
import scipy.io as sio
import numba
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


from numba import jit, prange
@jit(nopython=True, parallel=True)
def occlusion_detection(nvelo_projected_img, noocc_mask, sxx, syy, w, h2, verRange, horRange):
    for xx in range(w):
        for yy in range(h2):
            if nvelo_projected_img[yy, xx, 2] > 0:
                curloc = nvelo_projected_img[yy, xx, 0:2]
                curdir = epp - curloc
                curLen = np.sqrt(np.sum(curdir * curdir))
                oviewloc = nvelo_projected_img[yy, xx, 3:5]
                for tt in range(len(sxx)):
                    lookx = xx + sxx[tt]
                    looky = yy + syy[tt]

                    if (lookx >= 0) and (lookx < w) and (looky >= 0) and (looky < h2) and nvelo_projected_img[
                        looky, lookx, 2] > 0:
                        # Check if inside the eppl range
                        refvec = (nvelo_projected_img[looky, lookx, 0:2] - curloc)
                        projvec = np.sum(refvec * curdir) / curLen / curLen * curdir
                        verl = np.sqrt(np.sum(projvec * projvec))
                        horl = np.sqrt(np.sum((refvec - projvec) * (refvec - projvec)))
                        if verl < verRange and horl < horRange and np.sum(curdir * projvec) > 0 and (np.sqrt(np.sum(projvec * projvec)) < curLen):
                            # ck_list.append(np.array([lookx, looky]))
                            # after movement, if it is outside the epp line
                            oview_refvec1 = (nvelo_projected_img[looky, lookx, 3:5] - oviewloc)
                            oview_refvec2 = (nvelo_projected_img[looky, lookx, 3:5] - epp)
                            if np.sum(oview_refvec1 * (epp - oviewloc)) < 0 or np.sum(oview_refvec2 * (oviewloc - epp)) < 0:
                                noocc_mask[yy, xx] = 0


lrMapping = {2:'image_02', 3:'image_03'}
root_path = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
dayc = '2011_09_26'
seqc = '2011_09_26_drive_0005_sync'
frameind = 10
cam = 2


# Read RGB
rgb_path = os.path.join(root_path, dayc, seqc, lrMapping[cam], 'data', str(frameind).zfill(10) + '.png')
rgb = pil.open(rgb_path)
w = rgb.size[0]
h = rgb.size[1]
h2 = 400 # To store lidar projeciton in scanner view

# Read Lidar
velo_filename = os.path.join(root_path, dayc, seqc, 'velodyne_points', 'data', "{:010d}.bin".format(frameind))
velo = load_velodyne_points(velo_filename)

# Read Intrinsic
cam2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_cam_to_cam.txt'))
velo2cam = read_calib_file(os.path.join(root_path, dayc, 'calib_velo_to_cam.txt'))
velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

R_cam2rect = np.eye(4)
R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)

velo_projected = P_rect @ R_cam2rect @ velo2cam @ velo.T
velo_projected = velo_projected.T
velo_projected[:,0] = velo_projected[:,0] / velo_projected[:,2]
velo_projected[:,1] = velo_projected[:,1] / velo_projected[:,2]
onimgSelector = (velo[:,0] > 0) * (velo_projected[:,0] > 0) * (velo_projected[:,0] < w) * (velo_projected[:,1] > 0) * (velo_projected[:,1] < h)

intrinsic = np.eye(4)
intrinsic[0:3,0:3] = P_rect[0:3,0:3]
P_rect_ex = np.eye(4)
P_rect_ex[0:3,:] = P_rect
tmp_ex = np.linalg.inv(intrinsic) @ P_rect_ex
extrinsic = tmp_ex @ R_cam2rect @ velo2cam


camPos = np.array([[0,0,0,1]]).T
camPos = np.linalg.inv(extrinsic) @ camPos

siol = sio.loadmat(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/raytrace_ptc/matlab', 'nextrinsic.mat'))
nextrinsic = np.array(siol['nextrinsic'])
lidarPos = np.linalg.inv(nextrinsic) @ np.array([[0,0,0,1]]).T

lidarEpp = intrinsic @ extrinsic @ lidarPos
lidarEpp[0] = lidarEpp[0] / lidarEpp[2]
lidarEpp[1] = lidarEpp[1] / lidarEpp[2]

camEpp = intrinsic @ nextrinsic @ camPos
camEpp[0] = camEpp[0] / camEpp[2]
camEpp[1] = camEpp[1] / camEpp[2]
assert np.sum(np.abs(lidarEpp[0:2] - camEpp[0:2])) < 1e-1

epp = lidarEpp[0:2].T

nvelo_projected = intrinsic @ nextrinsic @ velo.T
nvelo_projected = nvelo_projected.T
nvelo_projected[:,0] = nvelo_projected[:,0] / nvelo_projected[:,2]
nvelo_projected[:,1] = nvelo_projected[:,1] / nvelo_projected[:,2]

nvelo_projected_img = np.zeros([h2, w, 5])
rnx = np.round(nvelo_projected[:,0]).astype(np.int)
rny = np.round(nvelo_projected[:,1]).astype(np.int)
rnd = nvelo_projected[:,2]
nval = (rnx >= 0) * (rnx < w) * (rny >= 0) * (rny < h2) * (rnd > 0) * onimgSelector
nvelo_projected_img[rny[nval], rnx[nval], :] = np.concatenate([nvelo_projected[nval,0:3], velo_projected[nval,0:2]], axis=1)
noocc_mask = np.zeros([h2, w])
noocc_mask[rny[nval], rnx[nval]] = 1

org_mask = np.zeros([h2, w])
org_mask[rny[nval], rnx[nval]] = 1

searchW = 20
sxx, syy = np.meshgrid(list(range(-searchW,searchW+1)), list(range(-searchW,searchW+1)))
sxx = sxx.flatten().astype(np.int)
syy = syy.flatten().astype(np.int)
tmps = (sxx == 0) * (syy == 0)
tmps = (1 - tmps) == 1
sxx = sxx[tmps]
syy = syy[tmps]
itnum = len(sxx)

verRange = 30
horRange = 3


occlusion_detection(nvelo_projected_img, noocc_mask, sxx, syy, w, h2, verRange, horRange)

drawx = nvelo_projected_img[:,:,0][org_mask == 1]
drawy = nvelo_projected_img[:,:,1][org_mask == 1]
z = nvelo_projected_img[:,:,2][org_mask == 1]
z = z / 40
cm = plt.get_cmap('magma')
z = cm(z)
fig, ax = plt.subplots()
ax.set_xlim(0, w)  # decreasing time
ax.set_ylim(h2, 0)  # decreasing time
ax.axis('equal')
plt.scatter(drawx, drawy, s=5, marker='.', c = z[:,0:3])

drawx = nvelo_projected_img[:,:,0][noocc_mask == 1]
drawy = nvelo_projected_img[:,:,1][noocc_mask == 1]
z = nvelo_projected_img[:,:,2][noocc_mask == 1]
z = z / 40
cm = plt.get_cmap('magma')
z = cm(z)
fig, ax = plt.subplots()
ax.set_xlim(0, w)  # decreasing time
ax.set_ylim(h2, 0)  # decreasing time
ax.axis('equal')
plt.scatter(drawx, drawy, s=5, marker='.', c = z[:,0:3])


drawx = nvelo_projected_img[:,:,3][noocc_mask == 1]
drawy = nvelo_projected_img[:,:,4][noocc_mask == 1]
z = nvelo_projected_img[:,:,2][noocc_mask == 1]
z = z / 40
cm = plt.get_cmap('magma')
z = cm(z)
fig, ax = plt.subplots()
plt.imshow(rgb)
ax.set_xlim(0, w)  # decreasing time
ax.set_ylim(h2, 0)  # decreasing time
ax.axis('equal')
plt.scatter(drawx, drawy, s=5, marker='.', c = z[:,0:3])

drawx = nvelo_projected_img[:,:,3][org_mask == 1]
drawy = nvelo_projected_img[:,:,4][org_mask == 1]
z = nvelo_projected_img[:,:,2][org_mask == 1]
z = z / 40
cm = plt.get_cmap('magma')
z = cm(z)
fig, ax = plt.subplots()
ax.set_xlim(0, w)  # decreasing time
ax.set_ylim(h2, 0)  # decreasing time
ax.axis('equal')
plt.scatter(drawx, drawy, s=5, marker='.', c = z[:,0:3])
# xx = 807
# yy = 202
# xx = 804
# yy = 208
xx = 798
yy = 266
recorded_pixels = list()
filtered_pixels = list()
occ_pixels = list()
if nvelo_projected_img[yy, xx, 2] > 0:
    curloc = nvelo_projected_img[yy, xx, 0:2]
    curdir = epp - curloc
    curLen = np.sqrt(np.sum(curdir * curdir))
    oviewloc = nvelo_projected_img[yy, xx, 3:5]
    for tt in range(len(sxx)):
        lookx = xx + sxx[tt]
        looky = yy + syy[tt]

        if (lookx >= 0) and (lookx < w) and (looky >= 0) and (looky < h2) and nvelo_projected_img[looky, lookx, 2] > 0:
            # Check if inside the eppl range
            recorded_pixels.append(nvelo_projected_img[looky, lookx, :])
            refvec = (nvelo_projected_img[looky, lookx, 0:2] - curloc)
            projvec = np.sum(refvec * curdir) / curLen / curLen * curdir
            verl = np.sqrt(np.sum(projvec * projvec))
            horl = np.sqrt(np.sum((refvec - projvec) * (refvec - projvec)))
            if verl < verRange and horl < horRange and np.sum(curdir * projvec) > 0 and (np.sqrt(np.sum(projvec * projvec)) < curLen):
                # ck_list.append(np.array([lookx, looky]))
                # after movement, if it is outside the epp line
                filtered_pixels.append(nvelo_projected_img[looky, lookx, :])
                oview_refvec1 = (nvelo_projected_img[looky, lookx, 3:5] - oviewloc)
                oview_refvec2 = (nvelo_projected_img[looky, lookx, 3:5] - epp)
                if np.sum(oview_refvec1 * (epp - oviewloc)) < 0 or np.sum(oview_refvec2 * (oviewloc - epp)) < 0:
                    occ_pixels.append(nvelo_projected_img[looky, lookx, :])
                    noocc_mask[yy, xx] = 0

drawx = nvelo_projected[onimgSelector, 0]
drawy = nvelo_projected[onimgSelector, 1]
z = velo_projected[onimgSelector, 2]
z = z / 40
cm = plt.get_cmap('magma')
z = cm(z)
fig, ax = plt.subplots()
ax.set_xlim(0, w)  # decreasing time
ax.set_ylim(h2, 0)  # decreasing time
ax.axis('equal')
plt.scatter(drawx, drawy, s=5, marker='.', c = z[:,0:3])
plt.scatter(nvelo_projected_img[yy, xx, 0], nvelo_projected_img[yy, xx, 1], s=50, marker='.', c = 'r')
if len(recorded_pixels) > 0:
    recorded_pixels = np.stack(recorded_pixels, axis=0)
    plt.scatter(recorded_pixels[:, 0], recorded_pixels[:, 1], s=25, marker='.', c = 'g')
if len(filtered_pixels) > 0:
    filtered_pixels = np.stack(filtered_pixels, axis=0)
    plt.scatter(filtered_pixels[:, 0], filtered_pixels[:, 1], s=25, marker='.', c = 'b')
if len(occ_pixels) > 0:
    occ_pixels = np.stack(occ_pixels, axis=0)
    plt.scatter(occ_pixels[:, 0], occ_pixels[:, 1], s=25, marker='.', c = 'k')

# fig, ax = plt.subplots()
# ax.set_xlim(0, w)  # decreasing time
# ax.set_ylim(h2, 0)  # d
# ax.axis('equal')
# plt.scatter(nvelo_projected_img[yy, xx, 3], nvelo_projected_img[yy, xx, 4], s=50, marker='.', c = 'r')
# plt.scatter(recorded_pixels[:, 3], recorded_pixels[:, 4], s=25, marker='.', c = 'g')
# plt.scatter(filtered_pixels[:, 3], filtered_pixels[:, 4], s=25, marker='.', c = 'b')
# plt.scatter(occ_pixels[:, 3], occ_pixels[:, 4], s=25, marker='.', c = 'k')


# xx = 767
# yy = 221
# ck_list = list()
# fail_list = list()
# curloc = nvelo_projected_img[yy, xx, 0:2]
# curdir = epp - curloc
# curLen = np.sqrt(np.sum(curdir * curdir))
# for tt in range(itnum):
#     lookx = xx + sxx[tt]
#     looky = yy + syy[tt]
#
#     refvec = (np.array([lookx, looky]) - curloc)
#     projvec = np.sum(refvec * curdir) / curLen / curLen * curdir
#     verl = np.sqrt(np.sum(projvec * projvec))
#     horl = np.sqrt(np.sum((refvec - projvec) * (refvec - projvec)))
#     if verl < verRange and horl < horRange and np.sum(curdir * projvec) > 0 and (np.sqrt(np.sum(projvec * projvec)) < curLen):
#         ck_list.append(np.array([lookx, looky]))
#     else:
#         fail_list.append(np.array([lookx, looky]))
# fail_list = np.stack(fail_list, axis = 0)
# ck_list = np.stack(ck_list, axis = 0)
# plt.figure()
# plt.scatter([xx], [yy], 3, 'r')
# plt.scatter(ck_list[:,0], ck_list[:,1], 3, 'g')
# plt.scatter(fail_list[:,0], fail_list[:,1], 3, 'b')


# for xx in range(w):
#     for yy in range(h2):
#         if nvelo_projected_img[yy, xx, 2] > 0:
#             curloc = nvelo_projected_img[yy, xx, 0:2]
#             curdir = epp - curloc
#             curLen = np.sqrt(np.sum(curdir * curdir))
#             oviewloc = nvelo_projected_img[yy, xx, 3:5]
#             for tt in range(itnum):
#                 lookx = xx + sxx[tt]
#                 looky = yy + syy[tt]
#
#                 if (lookx >= 0) and (lookx < w) and (looky >= 0) and (looky < h2) and nvelo_projected_img[looky, lookx, 2] > 0:
#                     # Check if inside the eppl range
#                     refvec = (nvelo_projected_img[looky, lookx, 0:2] - curloc)
#                     projvec = np.sum(refvec * curdir) / curLen / curLen * curdir
#                     verl = np.sqrt(np.sum(projvec * projvec))
#                     horl = np.sqrt(np.sum((refvec - projvec) * (refvec - projvec)))
#                     if verl < verRange and horl < horRange and np.sum(curdir * projvec) > 0 and (np.sqrt(np.sum(projvec * projvec)) < curLen):
#                         # ck_list.append(np.array([lookx, looky]))
#                         # after movement, if it is outside the epp line
#                         oview_refvec1 = (nvelo_projected_img[looky, lookx, 3:5] - oviewloc)
#                         oview_refvec2 = (nvelo_projected_img[looky, lookx, 3:5] - epp)
#                         if np.sum(oview_refvec1 * (epp - oviewloc)) > 0 and np.sum(oview_refvec2 * (oviewloc - epp)) > 0:
#                             noocc_mask[yy, xx] = 0



#
#
# drawx = velo[:,0]
# drawy = velo[:,1]
# drawz = velo[:,2]
# drawx = matlab.double(drawx.tolist())
# drawy = matlab.double(drawy.tolist())
# drawz = matlab.double(drawz.tolist())
#
# drawcx = camPos[0:1,:]
# drawcy = camPos[1:2,:]
# drawcz = camPos[2:3,:]
# drawcx = matlab.double(drawcx.tolist())
# drawcy = matlab.double(drawcy.tolist())
# drawcz = matlab.double(drawcz.tolist())
#
# drawlx = camPos[0:1,:]
# drawly = camPos[1:2,:]
# drawlz = camPos[2:3,:]
# drawlx = matlab.double(drawlx.tolist())
# drawly = matlab.double(drawly.tolist())
# drawlz = matlab.double(drawlz.tolist())
#
#
# eng.scatter3(drawx, drawy, drawz, 5, '.', nargout=0)
# eng.eval('hold on', nargout=0)
# eng.scatter3(drawcx, drawcy, drawcz, 50, 'r.', nargout=0)
# eng.eval('axis equal', nargout=0)
# eng.scatter3(drawlx, drawly, drawlz, 50, 'g.', nargout=0)
# eng.eval('axis equal', nargout=0)
# xlim = matlab.double([0, 50])
# ylim = matlab.double([-10, 10])
# zlim = matlab.double([-5, 5])
# eng.xlim(xlim, nargout=0)
# eng.ylim(ylim, nargout=0)
# eng.zlim(zlim, nargout=0)
# eng.eval('view([-79 17])', nargout=0)
# eng.eval('camzoom(1.2)', nargout=0)
# eng.eval('grid off', nargout=0)