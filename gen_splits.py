import os
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

if __name__ == "__main__":
    gen_split_quadruplets()