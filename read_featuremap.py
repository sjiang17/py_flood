import torch.utils.data as data
import os
import os.path
import h5py
import numpy as np
    
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

def h5_loader(path):
    return np.array(h5py.File(path, 'r')['data'])

class FeatureReader(data.Dataset):
    def __init__(self, root, loader=h5_loader):

        feature_maps = make_dataset(root)
        if len(feature_maps) == 0:
            raise(RuntimeError("Found 0 feature maps in subfolders of: " + root + "\n"))
        self.root = root
        self.feature_maps = feature_maps
        self.loader = loader

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

        return fm, gt

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