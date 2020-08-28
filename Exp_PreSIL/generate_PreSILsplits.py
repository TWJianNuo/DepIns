# Generate splits used for Train/Val
import os
def get_entry_from_path(imgpath):
    comps = imgpath.split('/')
    entry = comps[-3] + '/' + comps[-2] + ' ' + comps[-1].split('.')[0] + '\n'
    return entry

def collect_all_entries(folder):
    import glob
    import random
    seqs = [f.path for f in os.scandir(folder) if f.is_dir()]
    entries = list()
    for seq in seqs:
        imgFolder = os.path.join(seq, 'rgb')
        for imgpath in glob.glob(imgFolder + '/*.png'):
            entries.append(get_entry_from_path(imgpath))
    trainval_ratio = 0.8

    splitind = int(len(entries) * trainval_ratio)
    transplits = entries[0:splitind]
    valsplits = entries[splitind::]

    random.shuffle(valsplits)
    valsplits = valsplits[0:1000]
    return transplits, valsplits

dataset_dir = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/PreSIL_organized'
split_folder = '/home/shengjie/Documents/Project_SemanticDepth/splits/preSIL'
transplits, valsplits = collect_all_entries(dataset_dir)

with open(os.path.join(split_folder, 'train_files.txt'), "w") as text_file:
    for entry in transplits:
        text_file.write(entry)

with open(os.path.join(split_folder, 'val_files.txt'), "w") as text_file:
    for entry in valsplits:
        text_file.write(entry)

