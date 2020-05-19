import os
import shutil
organizedRoot = '/home/shengjie/Documents/Data/PreSIL_organized'
lidarroot = '/home/shengjie/Documents/Data/velodyne/'
for i in range(51075):
    seqNum = int(i / 5000)
    seq_path = os.path.join(organizedRoot, "{:06d}".format(seqNum))
    target_lidar_path = os.path.join(seq_path, "lidar")
    os.makedirs(target_lidar_path, exist_ok=True)

    target_lidar_path = os.path.join(target_lidar_path, str(i).zfill(6) + '.bin')
    source_lidar_path = os.path.join(lidarroot, str(i).zfill(6) + '.bin')
    target_dir = os.path.join(target_lidar_path, str(i).zfill(6) + '.bin')
    shutil.copyfile(source_lidar_path, target_lidar_path)
    print(target_lidar_path)