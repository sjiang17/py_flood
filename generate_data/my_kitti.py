import os
import xml.etree.ElementTree as ET
import numpy as np

class my_kitti():
	def __init__(self, dir, cls):
		self._dir = dir
		assert os.path.exists(self._dir), 'Kitti data directory does not exist'
		self._img_dir = os.path.join(dir, 'image_2')
		self._annot_dir = os.path.join(dir, cls, 'annot_xml')
		self._num_classes = 2
		self._classes = ('__background__',  # always index 0
                     'car')
		self._class_to_ind = dict(list(zip(self._classes, list(range(self._num_classes)))))

	def _list_imgs(self):
		img_files = [os.path.join(self._img_dir, f.split('.xml')[0] + '.png') for f in sorted(os.listdir(self._annot_dir)) if f.endswith('.xml')]
		return img_files 

	def _list_annots(self):
		annot_files = [os.path.join(self._annot_dir, f) for f in sorted(os.listdir(self._annot_dir)) if f.endswith('.xml')]
		return annot_files

	def _read_annot(self, annot_file):
		# print annot_file
		tree = ET.parse(annot_file)
		objs = tree.findall('object')
		num_objs = len(objs)
		boxes = np.zeros((num_objs, 4), dtype=np.uint16)
		gt_classes = np.zeros(num_objs, dtype=np.int32)
		occs = np.zeros(num_objs, dtype=np.int32)

		for ix, obj in enumerate(objs):
			bbox = obj.find('bndbox')
			x1 = float(bbox.find('xmin').text) 
			y1 = float(bbox.find('ymin').text) 
			x2 = float(bbox.find('xmax').text) 
			y2 = float(bbox.find('ymax').text) 
			cls = self._class_to_ind[obj.find('name').text.lower().strip()]
			boxes[ix, :] = [x1, y1, x2, y2]
			gt_classes[ix] = cls
			occs[ix] = int(obj.find('occluded').text)

		return {'boxes': boxes,
				'gt_classes': gt_classes,
				'occluded': occs}

	def _modify_annot(self, annot_file, mask_areas, mask_boxes, save_dir):
		
		# mask_areas = [20.01, 30.22]
		tree = ET.parse(annot_file)
		objs = tree.findall('object')
		for ix, obj in enumerate(objs):
			mask_box = ET.Element('mask_box')
			
			mask_area = ET.Element('mask_area')
			xmin = ET.Element('xmin')
			ymin = ET.Element('ymin')
			xmax = ET.Element('xmax')
			ymax = ET.Element('ymax')
			
			mask_area.text = str(mask_areas[ix])
			xmin.text, ymin.text, xmax.text, ymax.text = [str(x) for x in mask_boxes[ix]]
			
			mask_box.append(xmin)
			mask_box.append(ymin)
			mask_box.append(xmax)
			mask_box.append(ymax)
			mask_box.append(mask_area)
			
			obj.append(mask_box)

		save_file = os.path.join(save_dir, os.path.basename(annot_file))
		tree.write(open(save_file, 'w'))












