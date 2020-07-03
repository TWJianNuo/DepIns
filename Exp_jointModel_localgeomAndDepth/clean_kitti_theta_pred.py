import glob
import os
import shutil

dataroot = '/home/shengjie/Documents/Data/Kitti/kitti_theta_pred'
dates = glob.glob(os.path.join(dataroot, '*'))
delete_enytr = ['htheta_vls_flipped', 'vtheta_vls', 'vtheta_vls_flipped', 'htheta_vls']
for date in dates:
    seqs = glob.glob(os.path.join(date, '*'))
    for seq in seqs:
        folds = glob.glob(os.path.join(seq, '*'))
        for fold in folds:
            if fold.split('/')[-1] in delete_enytr:
                shutil.rmtree(fold)
                print("%s deleted" % fold)
