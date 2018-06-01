import os
import cPickle
from random import shuffle
import copy

occ2unocc_pair = {}
unocc2unocc_pair = {}

phase = 'test'
occ_dir = '/pvdata/dataset/kitti/vehicle/roi/occ/roi_feature/' + phase
unocc_dir = '/pvdata/dataset/kitti/vehicle/roi/unocc/roi_feature/' + phase

occ_feas = os.listdir(occ_dir)
unocc_feas = os.listdir(unocc_dir)

print len(occ_feas), len(unocc_feas)

unocc_feas_shuffle = copy.deepcopy(unocc_feas)
shuffle(unocc_feas_shuffle)
ix = 0
for occ_fea in occ_feas:
	if ix == len(unocc_feas_shuffle):
		ix = 0
	occ2unocc_pair[occ_fea] = unocc_feas_shuffle[ix]
	ix += 1

for unocc_fea in unocc_feas:
	unocc2unocc_pair[unocc_fea] = unocc_fea

cPickle.dump(occ2unocc_pair, open('/pvdata/dataset/kitti/vehicle/roi/{}_occ2unocc_pair.pkl'.format(phase), 'w'))
cPickle.dump(unocc2unocc_pair, open('/pvdata/dataset/kitti/vehicle/roi/{}_unocc2unocc_pair.pkl'.format(phase), 'w'))
