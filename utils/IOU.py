# -*- coding:utf-8 -*-
"""
This file is used to calculate IOU of two boxes.
"""


def IOU(box1=(0, 0, 0, 0), box2=(0, 0, 0, 0)):
	# each box is consisted of [x, y, w, h]
	xmin1 = box1[0] - box1[2]/2
	ymin1 = box1[1] - box1[3]/2
	xmin2 = box2[0] - box2[2]/2
	ymin2 = box2[1] - box2[3]/2
	xmin = max(xmin1, xmin2)
	ymin = max(ymin1, ymin2)

	xmax1 = box1[0] + box1[2] / 2
	ymax1 = box1[1] + box1[3] / 2
	xmax2 = box2[0] + box2[2] / 2
	ymax2 = box2[1] + box2[3] / 2
	xmax = min(xmax1, xmax2)
	ymax = min(ymax1, ymax2)

	w = xmax - xmin
	h = ymax - ymin
	if w <= 0 or h <= 0:
		return 0
	else:
		area_intersect = w*h
		area_union = box1[2]*box1[3] + box2[2]*box2[3] - area_intersect
		return area_intersect/area_union


if __name__ == "__main__":
	print(IOU([0, 0, 34, 50], [0, 0, 52.93, 70.13]))
