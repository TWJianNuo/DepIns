import os
import glob
import random

csroot = '/home/shengjie/Documents/Data/cityscapes'
svroot = '/home/shengjie/Documents/Project_SemanticDepth/splits/cityscape'
splits = ['train', 'val', 'test']
dirmapping = {'leftImg8bit': 'l', 'rightImg8bit': 'r'}

for split in splits:
    fnames = list()
    folds = glob.glob(os.path.join(csroot, 'leftImg8bit', '*'))
    for fold in folds:
        cat = fold.split('/')[-1]
        if split in fold:
            cities = glob.glob(os.path.join(fold, '*'))
            for city in cities:
                pngs = glob.glob(os.path.join(city, '*.png'))

                for png in pngs:
                    comps = png.split('/')

                    cityname = comps[-2]
                    pngnames = comps[-1]
                    inds = '{}_{}'.format(pngnames.split('_')[1], pngnames.split('_')[2])
                    dir = dirmapping[pngnames.split('_')[3].split('.')[0]]

                    if os.path.exists(os.path.join(csroot, 'disparity', cat, cityname, "{}_{}_{}.png".format(cityname, inds, 'disparity'))):
                        fnames.append('{} {} {} {}'.format(cat, cityname, inds, dir))

    random.shuffle(fnames)
    with open(os.path.join(svroot, "{}_files.txt".format(split)), "w") as text_file:
        for fname in fnames:
            text_file.write(fname + '\n')




