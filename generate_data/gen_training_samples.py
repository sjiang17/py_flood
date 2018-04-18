import random
import cv2
from my_voc import my_voc
import numpy as np
import os
import sys
import shutil

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def box_area(xmin, ymin, xmax, ymax):
	return (float(xmax) - float(xmin)) * (float(ymax) - float(ymin))

def mask_img(box):
	xmin, ymin, xmax, ymax = box
	satified = False
	while not satified:
		mask_xmin, mask_xmax = sorted(np.random.randint(xmin, xmax+1, 2))
		mask_ymin, mask_ymax = sorted(np.random.randint(ymin, ymax+1, 2))
		ratio = box_area(mask_xmin, mask_ymin, mask_xmax, mask_ymax) / \
			box_area(xmin, ymin, xmax, ymax) 
		if ratio >= 0.3 and ratio < 0.7:
			satified = True
	mask = [mask_xmin, mask_ymin, mask_xmax, mask_ymax]
	return mask

def black_patch(img, annot):
	# for every object
	# random mask black patch
	boxes = annot['boxes']
	gt_classes = annot['gt_classes']
	person_exist = False
	for (box, cls) in zip(boxes, gt_classes):
		if not cls == 15:
			continue
		person_exist = True
		if random.random() > 0.0:
			mask = mask_img(box)
			cv2.rectangle(img, (mask[0], mask[1]), (mask[2], mask[3]), (104,117,123), -1)
	return person_exist
	# return masked image

if __name__ == '__main__':
	r_size = 600

	random.seed(100)
	voc_dir = os.path.join(sys.path[0], '../dataset/VOC2012')
	save_dir = os.path.join(sys.path[0], '../dataset/classify600')
	train_dir_0 = os.path.join(save_dir, 'train/0')
	train_dir_1 = os.path.join(save_dir, 'train/1')
	test_dir_0 = os.path.join(save_dir, 'test/0')
	test_dir_1 = os.path.join(save_dir, 'test/1')
	create_dir(train_dir_0)
	create_dir(train_dir_1)
	create_dir(test_dir_0)
	create_dir(test_dir_1)

	voc = my_voc(voc_dir)
	# get all images 
	img_list = voc._list_imgs()
	# get all annotations
	annot_list = voc._list_annots()
	assert len(img_list) == len(annot_list)
	# iterate every image
	for ix, (img_file, annot_file) in enumerate(zip(img_list, annot_list)):
		if ix % 200 == 0:
			print "processed {} out of {} images".format(ix, len(img_list))
		is_test = True if (ix%5 == 0) else False
		img = cv2.imread(img_file)
		origin_img = cv2.resize(img, (r_size, r_size), interpolation=cv2.INTER_CUBIC)
		# get annotations 
		annot = voc._read_annot(annot_file)
		# generate black patched image
		person_exist = black_patch(img, annot)
		if person_exist:
			# save original image to folder 0
			if is_test:
				origin_save_file = os.path.join(test_dir_0, os.path.basename(img_file))
				save_file = os.path.join(test_dir_1, os.path.basename(img_file))
			else:
				origin_save_file = os.path.join(train_dir_0, os.path.basename(img_file))
				save_file = os.path.join(train_dir_1, os.path.basename(img_file))

			cv2.imwrite(origin_save_file, origin_img)
			# shutil.copyfile(img_file, origin_save_file)
			# save masked image in folder 1
			# save_file = os.path.join(os.path.join(voc_dir, '1'), os.path.basename(img_file))
			# save_file = os.path.join(sys.path[0], os.path.basename(img_file))
			img = cv2.resize(img, (r_size, r_size), interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(save_file, img)

	

