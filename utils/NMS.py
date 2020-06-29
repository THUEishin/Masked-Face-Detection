# -*- coding:utf-8 -*-
"""
This file provides function to do Non-Maximum-Suppress
"""
from utils.IOU import IOU
import torch


def NMS(labels, IOU_threshold=0.4):
	# labels has the shape of num_labels*7
	num_labels = labels.shape[0]
	for i in range(num_labels):
		iou_mask = []
		try:
			box1 = labels[i]
		except ValueError:
			break
		except IndexError:
			break

		for j in range(num_labels):
			try:
				box2 = labels[i+j+1]
			except ValueError:
				break
			except IndexError:
				break

			if IOU(box1[:4], box2[:4]) < IOU_threshold:
				iou_mask.append(1)
			else:
				iou_mask.append(0)

		iou_mask = torch.tensor(iou_mask).float().unsqueeze(1)
		labels[i+1:] *= iou_mask

		non_zero_index = torch.nonzero(labels[:, 4]).squeeze()
		labels = labels[non_zero_index].view(-1, 7)

	return labels
