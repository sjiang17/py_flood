import os
import cPickle
from random import shuffle
import copy
import numpy as np

occ2unocc_pair = {}

phase = 'test'
occ_dir = '/pvdata/dataset/kitti/vehicle/roi/occ/roi_feature/' + phase
unocc_dir = '/pvdata/dataset/kitti/vehicle/roi/unocc/roi_feature/' + phase

occ_feas = os.listdir(occ_dir)

print len(occ_feas)

obj_alpha_pair = cPickle.load(open('/pvdata/dataset/kitti/vehicle/roi/occ_obj_alpha_pair.pkl', 'r'))
alpha_objs_dict = cPickle.load(open('/pvdata/dataset/kitti/vehicle/roi/unocc_alpha_objs_dict.pkl', 'r'))

print len(obj_alpha_pair)

group_ind = np.zeros(8, dtype=int)
group_len = np.zeros(8, dtype=int)
for i in range(8):
	group_len[i] = len(alpha_objs_dict[i])

print group_len

alpha_occ2unocc_pair = {}
for occ_fea in occ_feas:
	basename = occ_fea.strip('.h5')
	group = obj_alpha_pair[basename]

	if group_ind[group] >= group_len[group]:
		group_ind[group] = 0
	unocc_group_ind = group_ind[group]
	alpha_occ2unocc_pair[basename] = alpha_objs_dict[group][unocc_group_ind]
	group_ind[group] += 1


cPickle.dump(alpha_occ2unocc_pair, open('/pvdata/dataset/kitti/vehicle/roi/{}_alpha_occ2unocc_pair.pkl'.format(phase), 'w'))
