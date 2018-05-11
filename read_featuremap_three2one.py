import torch.utils.data as data
import os
import os.path
import h5py
import numpy as np
import xml.etree.ElementTree as ET
    
def is_featuremap_file(filename):
    filename_lower = filename.lower()
    return filename_lower.endswith('.h5')

def make_dataset(dir, phase):
    feature_maps = []
    dir = os.path.expanduser(dir)
    conv3_dir = os.path.join(dir, 'feature_map-conv3pool', phase, '1')
    conv4_dir = os.path.join(dir, 'feature_map-conv4pool', phase, '1')
    conv5_d0 = os.path.join(dir, 'feature_map-conv5pool', phase, '0')
    conv5_d1 = os.path.join(dir, 'feature_map-conv5pool', phase, '1')
    if (not os.path.isdir(conv5_d0)) or (not os.path.isdir(conv5_d1)):
        raise(RuntimeError("directory not correct!"))
    for fname in sorted(os.listdir(conv5_d0)):
        if is_featuremap_file(fname):
            path_conv3 = os.path.join(conv3_dir, fname)
            path_conv4 = os.path.join(conv4_dir, fname)
            path0_conv5 = os.path.join(conv5_d0, fname)
            path1_conv5 = os.path.join(conv5_d1, fname)
            assert os.path.exists(path1_conv5)
            item = (path_conv3, path_conv4, path1_conv5, path0_conv5)
            feature_maps.append(item)

    return feature_maps

def get_occlusion_level(dir, phase):
    file = os.path.join(os.path.expanduser(dir), phase + '-occlusion_level.txt')
    f = open(file, 'r')
    occ_level = [tuple(line.strip('\n').split(' ')) for line in f.readlines()]
    return occ_level

def mask_loader(dir, basename):
    # annot_files = [f for f in sorted(os.path.listdir(dir) if f.endswith('.xml'))]
    annot_file = os.path.join(os.path.expanduser(dir), 'masked_annotations', basename + '.xml')
    tree = ET.parse(annot_file)
    width = float(tree.find('size').find('width').text)
    height = float(tree.find('size').find('height').text)
    
    objs = tree.findall('object')
    masks = []
    for ix, obj in enumerate(objs):
        occ_level = int(obj.find('occluded').text)
        trunc = int(obj.find('truncated').text)
        if occ_level == 0 and trunc == 0:
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
    def __init__(self, root, phase=None ,loader=h5_loader, mask_loader=mask_loader):

        feature_maps = make_dataset(root, phase)
        occ_level = get_occlusion_level(root, phase)
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
        path_conv3, path_conv4, path1_conv5, path0_conv5 = self.feature_maps[index]
        fm_conv3 = self.loader(path_conv3)
        fm_conv4 = self.loader(path_conv4)
        fm_conv5 = self.loader(path1_conv5)
        gt = self.loader(path0_conv5)
        
        occ = self.occ_level[index]
        # print path1
        basename = os.path.basename(path0_conv5).split('.png.h5')[0]
        assert basename == occ[0], \
            "{} not {}".format(basename, occ[0])
        masks = self.mask_loader(self.root, basename)
        assert (len(masks) > 0) == (not (int(occ[1]) == 0)), "{}, {}, {}".format(basename, len(masks), occ[1])
        return fm_conv3, fm_conv4, fm_conv5, gt, masks

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
