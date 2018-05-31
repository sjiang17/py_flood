import os
import cPickle
from random import shuffle

occ2unocc_pair = {}
unocc2unocc_pair = {}

occ_dir = ''
unocc_dir = ''

occ_feas = os.listdir(occ_dir)
unocc_feas = os.listdir(unocc_dir)

print len(occ_feas), len(unocc_dir)

unocc_feas_shuffle = shuffle(unocc_feas)
ix = 0
for ix, occ_fea in enumerate(occ_feas):
	if ix == len(unocc_feas_shuffle):
		ix = 0
	occ2unocc_pair[occ_fea] = unocc_feas_shuffle[ix]
	ix += 1

for unocc_fea in unocc_feas:
	unocc2unocc_pair[unocc_fea] = unocc_fea

cPickle.dump(occ2unocc_pair, open('', 'w'))
cPickle.dump(unocc2unocc_pair, open('', 'w'))
