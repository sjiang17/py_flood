from PIL import Image 
import os

SCALE = 600.0
MAX_SIZE = 1000

basedirs = ['/train/0', '/train/1', '/test/0', '/test/1']

for basedir in basedirs:
	print basedir
	kitti_img_dir = '/siyuvol/dataset/kitti/mask_noresize' + basedir
	save_dir = '/siyuvol/dataset/kitti/greymask' + basedir 
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	kitti_img_files = [os.path.join(kitti_img_dir, f) for f in sorted(os.listdir(kitti_img_dir)) if f.endswith('.png')]

	for img_file in kitti_img_files:
		img = Image.open(img_file)
		width, height = img.size
		if width <= height:
			r_width = int(SCALE)
			r_height = int(min(SCALE / width * height, MAX_SIZE))
		else:
			r_height = int(SCALE)
			r_width = int(min(SCALE / height * width, MAX_SIZE))
		# print width, r_width, height, r_height
		resized = img.resize((r_width, r_height), Image.BICUBIC)

		resized.save(os.path.join(save_dir, os.path.basename(img_file)), format='png')

