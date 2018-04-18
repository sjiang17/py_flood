import torch.utils.data as data
import os
import os.path
import h5py
import numpy as np
import xml.etree.ElementTree as ET
    
def is_featuremap_file(filename):
    filename_lower = filename.lower()
    return filename_lower.endswith('.h5')

def make_dataset(dir):
    feature_maps = []
    dir = os.path.expanduser(dir)
    
    d0 = os.path.join(dir, '0')
    d1 = os.path.join(dir, '1')
    if (not os.path.isdir(d0)) or (not os.path.isdir(d1)):
        raise(RuntimeError("directory not correct!"))
    for fname in sorted(os.listdir(d0)):        
        if is_featuremap_file(fname):
            path0 = os.path.join(d0, fname)
            path1 = os.path.join(d1, fname)
            assert os.path.exists(path1)
            item = (path1, path0)
            feature_maps.append(item)

    return feature_maps

def get_occlusion_level(dir):
    file = os.path.join(os.path.expanduser(dir), 'occlusion_level.txt')
    f = open(file, 'r')
    occ_level = [tuple(line.strip('\n').split(' ')) for line in f.readlines()]
    return occ_level

def mask_loader(dir, basename):
    # annot_files = [f for f in sorted(os.path.listdir(dir) if f.endswith('.xml'))]
    annot_file = os.path.join(os.path.expanduser(dir), '../../masked_annotations',basename + '.xml')
    tree = ET.parse(annot_file)
    width = float(tree.find('size').find('width').text)
    height = float(tree.find('size').find('height').text)
    
    objs = tree.findall('object')
    masks = []
    for ix, obj in enumerate(objs):
        occ_level = int(obj.find('occluded').text)
        if occ_level == 0:
            continue
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) / width
        y1 = float(bbox.find('ymin').text) / height
        x2 = float(bbox.find('xmax').text) / width
        y2 = float(bbox.find('ymax').text) / height
        masks.append((x1, y1, x2, y2))
    return masks
         

def h5_loader(path):
    return np.array(h5py.File(path, 'r')['data'])

class FeatureReader(data.Dataset):
    def __init__(self, root, loader=h5_loader, mask_loader=mask_loader):

        feature_maps = make_dataset(root)
        occ_level = get_occlusion_level(root)
        if len(feature_maps) == 0:
            raise(RuntimeError("Found 0 feature maps in subfolders of: " + root + "\n"))
        assert len(occ_level) == len(feature_maps)
        self.root = root
        self.feature_maps = feature_maps
        self.occ_level = occ_level
        self.loader = loader
        self.mask_loader = mask_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path1, path0 = self.feature_maps[index]        
        fm = self.loader(path1)
        gt = self.loader(path0)
        
        occ = self.occ_level[index]
        # print path1
        basename = os.path.basename(path1).split('.png.h5')[0]
        assert basename == occ[0], \
            "{} not {}".format(basename, occ[0])
        masks = self.mask_loader(self.root, basename)

        return fm, gt, int(occ[1]), masks

    def __len__(self):
        return len(self.feature_maps)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str