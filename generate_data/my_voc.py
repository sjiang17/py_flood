import os
import xml.etree.ElementTree as ET
import numpy as np

class my_voc():
	def __init__(self, dir):
		self._dir = dir
		assert os.path.exists(self._dir), 'VOC data directory does not exist'
		self._img_dir = os.path.join(dir, 'JPEGImages')
		self._annot_dir = os.path.join(dir, 'Annotations')
		self._num_classes = 21
		self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
		self._class_to_ind = dict(list(zip(self._classes, list(range(self._num_classes)))))

	def _list_imgs(self):
		img_files = [os.path.join(self._img_dir, f) for f in sorted(os.listdir(self._img_dir))]
		return img_files 

	def _list_annots(self):
		annot_files = [os.path.join(self._annot_dir, f) for f in sorted(os.listdir(self._annot_dir))]
		return annot_files

	def _read_annot(self, annot_file):
		tree = ET.parse(annot_file)
		objs = tree.findall('object')
		num_objs = len(objs)
		boxes = np.zeros((num_objs, 4), dtype=np.uint16)
		gt_classes = np.zeros((num_objs), dtype=np.int32)

		for ix, obj in enumerate(objs):
			bbox = obj.find('bndbox')
			x1 = float(bbox.find('xmin').text) - 1
			y1 = float(bbox.find('ymin').text) - 1
			x2 = float(bbox.find('xmax').text) - 1
			y2 = float(bbox.find('ymax').text) - 1
			cls = self._class_to_ind[obj.find('name').text.lower().strip()]
			boxes[ix, :] = [x1, y1, x2, y2]
			gt_classes[ix] = cls

		return {'boxes': boxes,
				'gt_classes': gt_classes}

	def _modify_annot(self, annot_file, mask_areas, save_dir):
		
		# mask_areas = [20.01, 30.22]
		tree = ET.parse(annot_file)
		objs = tree.findall('object')
		ix = 0
		for obj in objs:
			mask_area = ET.Element('mask_area')
			if obj.find('name').text == 'person':
				mask_area.text = str(mask_areas[ix])
				ix += 1
			else:
				mask_area.text = '0'
			obj.append(mask_area)

		save_file = os.path.join(save_dir, os.path.basename(annot_file))
		tree.write(open(save_file, 'w'))












