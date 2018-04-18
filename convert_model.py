import torch 
# model_file = '/home/tmu/py_flood/pytorch_model/faster_rcnn_1_10_625.pth'
# model = torch.load(model_file)

# cp_file = '/home/tmu/py_flood/pytorch_model/vgg16_faster_rcnn_iter_1190000.pth'
# cp = torch.load(cp_file)

# cp['RCNN_bbox_pred.bias'] = cp['bbox_pred_net.bias']
# cp.pop('bbox_pred_net.bias')
# cp['RCNN_bbox_pred.weight'] = cp['bbox_pred_net.weight']
# cp.pop('bbox_pred_net.weight')
# cp['RCNN_cls_score.bias'] = cp['cls_score_net.bias']
# cp.pop('cls_score_net.bias')
# cp['RCNN_cls_score.weight'] = cp['cls_score_net.weight']
# cp.pop('cls_score_net.weight')
# cp['RCNN_rpn.RPN_Conv.bias'] = cp['rpn_net.bias']
# cp.pop('rpn_net.bias')
# cp['RCNN_rpn.RPN_Conv.weigth'] = cp['rpn_net.weight']
# cp.pop('rpn_net.weight')
# cp['RCNN_rpn.RPN_bbox_pred.bias'] = cp['rpn_bbox_pred_net.bias']
# cp.pop('rpn_bbox_pred_net.bias')
# cp['RCNN_rpn.RPN_bbox_pred.weight'] = cp['rpn_bbox_pred_net.weight']
# cp.pop('rpn_bbox_pred_net.weight')
# cp['RCNN_rpn.RPN_cls_score.bias'] = cp['rpn_cls_score_net.bias']
# cp.pop('rpn_cls_score_net.bias')
# cp['RCNN_rpn.RPN_cls_score.weight'] = cp['rpn_cls_score_net.weight']
# cp.pop('rpn_cls_score_net.weight')
# cp['RCNN_top.0.bias'] = cp['vgg.classifier.0.bias']
# cp.pop('vgg.classifier.0.bias')
# cp['RCNN_top.0.weight'] = cp['vgg.classifier.0.weight']
# cp.pop('vgg.classifier.0.weight')
# cp['RCNN_top.3.bias'] = cp['vgg.classifier.3.bias']
# cp.pop('vgg.classifier.3.bias')
# cp['RCNN_top.3.weight'] = cp['vgg.classifier.3.weight']
# cp.pop('vgg.classifier.3.weight')

# for i in [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]:
# 	cp['RCNN_base.{}.bias'.format(str(i))] = cp['vgg.features.{}.bias'.format(str(i))]
# 	cp['RCNN_base.{}.weight'.format(str(i))] = cp['vgg.features.{}.weight'.format(str(i))]
# 	cp.pop('vgg.features.{}.weight'.format(str(i)))

# model['model'] = cp

# torch.save(model, '/home/tmu/py_flood/pytorch_model/faster_rcnn_vgg_coco.pth')

file_name = '/home/tmu/py_flood/pytorch_model/faster_rcnn_vgg_coco.pth'
cp = torch.load(file_name)
print len(cp.keys())
for key in sorted(cp.keys()):
	print key