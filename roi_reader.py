import torch.utils.data as data
import numpy as np
import os
import h5py
import cPickle
import random

def h5_loader(path):
    assert path.endswith('.h5')
    return np.array(h5py.File(path, 'r')['data'])

def make_dataset(occ_data_dir, unocc_data_dir, pairFile_dir, phase):
    occ_data_dir = os.path.expanduser(os.path.join(occ_data_dir, phase))
    unocc_data_dir = os.path.expanduser(os.path.join(unocc_data_dir, phase))
    pairFile_dir = os.path.expanduser(pairFile_dir)

    assert (os.path.exists(occ_data_dir)), "{} not exist".format(occ_data_dir) 
    assert (os.path.exists(unocc_data_dir)), "{} not exist".format(unocc_data_dir) 
    assert (os.path.exists(pairFile_dir)), "{} not exist".format(pairFile_dir) 

    occ2unocc_pair = cPickle.load(open(os.path.join(pairFile_dir, phase+'_occ2unocc_pair.pkl'), 'r'))
    unocc2unocc_pair = cPickle.load(open(os.path.join(pairFile_dir, phase+'_unocc2unocc_pair.pkl'), 'r'))

    data_list = []
    for pair in occ2unocc_pair.items():
        p0 = os.path.join(occ_data_dir, pair[0])
        p1 = os.path.join(unocc_data_dir, pair[1])
        assert (os.path.exists(p0), p0)
        assert (os.path.exists(p1), p1)
        data_list.append((p0, p1))
    for pair in unocc2unocc_pair.items():
        p1 = os.path.join(unocc_data_dir, pair[0])
        assert (os.path.exists(p1), p1)
        data_list.append((p1, p1))

    random.shuffle(data_list)
    return data_list

class FeatureReader(data.Dataset):
    def __init__(self, occ_data_dir, unocc_data_dir, pairFile_dir, phase, fm_loader=h5_loader):
        data_list = make_dataset(occ_data_dir, unocc_data_dir, pairFile_dir, phase)
        if len(data_list) == 0:
            raise(RuntimeError("Found 0 feature in subfolders of: " + img_dir + "\n"))

        self.occ_data_dir = occ_data_dir
        self.unocc_data_dir = unocc_data_dir
        self.data_list = data_list
        self.fm_loader = fm_loader

    def __getitem__(self, index):

        input_path, target_path = self.data_list[index]
        inputs = self.fm_loader(input_path)
        target = self.fm_loader(target_path)
        return inputs, target

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.img_dir)
        return fmt_str
