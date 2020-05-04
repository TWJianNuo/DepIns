# Read Lidar
# Get the Intrinsic and Extrinsic Camera
# Compute the Eppipolar Line And the Movement distance
import os
import PIL.Image as pil
from kitti_utils import *
import matplotlib.pyplot as plt
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()

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
# rgb.show()

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

drawx = velo_projected[onimgSelector, 0]
drawy = velo_projected[onimgSelector, 1]
z = velo_projected[onimgSelector, 2]

# drawx = matlab.double(drawx.tolist())
# drawy = matlab.double(drawy.tolist())
z = z / 40
cm = plt.get_cmap('magma')
z = cm(1 / z)
plt.figure()
plt.imshow(rgb)
plt.scatter(drawx, drawy, s=0.5, marker='.', c = z[:,0:3])

h = eng.scatter3(drawX_mono, drawY_mono, drawZ_mono, 5, draw_mono_sampledColor, 'filled', nargout=0)
eng.eval('axis equal', nargout=0)
xlim = matlab.double([0, 50])
ylim = matlab.double([-10, 10])
zlim = matlab.double([-5, 5])
eng.xlim(xlim, nargout=0)
eng.ylim(ylim, nargout=0)
eng.zlim(zlim, nargout=0)
eng.eval('view([-79 17])', nargout=0)
eng.eval('camzoom(1.2)', nargout=0)
eng.eval('grid off', nargout=0)
a =1