# -*- coding:utf-8 -*-
"""
This file is to train the YOLO model.
"""

from utils.load_data import load_data_for_train
from YOLOmodel import YOLONet
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.IOU import IOU


def train():
	epoch = 100
	train_data, train_label = load_data_for_train("./minitrain_resized", 64)
	val_data, val_label = load_data_for_train("./minival_resized")
	train_loss = []
	val_loss = []

	model = YOLONet()
	#model = torch.load("./temp/model/model_100.pkl")
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
	for i in range(epoch):
		train_loss.append(0.0)
		val_loss.append(0.0)

		for j, train_batch in enumerate(train_data):
			train_imgs, train_index = train_batch
			train_imgs = train_imgs.float()
			# train_prediction的形状是：
			# batch_size, height_size, width_size, num_anchors, bbox_attrs
			train_prediction = model(train_imgs)
			loss_t, loss_items_t = calculate_loss(train_prediction, train_index, train_label)
			print("\nEpoch:{}, Batch:{}\n\tTotal_loss={}\n\tLoss_items=({}, {}, {}, {})".format(
				i, j, loss_t.item(), loss_items_t[0].item(), loss_items_t[1].item(), loss_items_t[2].item(), loss_items_t[3].item()
			))
			train_loss[i] += loss_t.item()

			optimizer.zero_grad()
			loss_t.backward()
			optimizer.step()

		scheduler.step()

		with torch.no_grad():
			for j, val_batch in enumerate(val_data):
				val_imgs, val_index = val_batch
				val_imgs = val_imgs.float()
				val_prediction = model(val_imgs)
				loss_v, loss_items_v = calculate_loss(val_prediction, val_index, val_label)
				val_loss[i] += loss_v.item()
			print("\nEpoch:{}\n\tTrain: Total_loss={}\n\tValidation: Total_loss={}\n\t\tLoss_items=({}, {}, {}, {})".format(
				i, train_loss[i]/16, val_loss[i],
				loss_items_v[0].item(), loss_items_v[1].item(), loss_items_v[2].item(), loss_items_v[3].item()
			))

		if i % 10 == 9:
			plt.plot(train_loss)
			plt.show()
			plt.plot(val_loss)
			plt.show()
			torch.save(model, "./temp/model/model_"+str(i+1)+".pkl")


def calculate_loss(prediction, label_index, labels):
	loss_xy = torch.tensor(0.0)
	loss_wh = torch.tensor(0.0)
	loss_conf = torch.tensor(0.0)
	loss_cls = torch.tensor(0.0)
	anchors = [(6.08, 8.14), (15.05, 19.62), (22.45, 30.32), (52.93, 70.13)]
	MSE = nn.MSELoss(reduction='sum')
	CE = nn.CrossEntropyLoss(reduction='sum')
	BCE = nn.BCELoss(reduction='sum')
	# loss coefficient from YOLO paper
	lambda_coord = torch.tensor(5.0)
	lambda_non_ob = torch.tensor(0.001)
	batch_size = prediction.shape[0]
	batch_labels = [labels[index[0]] for index in label_index]

	for i in range(batch_size):
		label = batch_labels[i]
		# 高*宽*anchor数*7
		single_pred = prediction[i]
		# 定位label在哪个grid_cell, 确定与label匹配度最高的anchor大小
		# 对label的中心点坐标和宽高进行归一化，以及返回label中的类别
		lxy, lwh, lcls, lindex = label_transform(label, anchors)
		anchor_index, grid_loc_y, grid_loc_x = lindex
		lconf = torch.zeros_like(single_pred[..., 0])

		if len(anchor_index):
			object_pred = single_pred[grid_loc_y, grid_loc_x, anchor_index]
			lconf[grid_loc_y, grid_loc_x, anchor_index] = 1.0
			object_conf = lconf[grid_loc_y, grid_loc_x, anchor_index]
			loss_xy += lambda_coord*MSE(object_pred[..., 0:2], torch.tensor(lxy, dtype=torch.float))
			loss_wh += lambda_coord*MSE(object_pred[..., 2:4], torch.tensor(lwh, dtype=torch.float))
			loss_cls += torch.tensor(1.0)*CE(object_pred[..., 5:], torch.tensor(lcls, dtype=torch.long))
			loss_conf += (torch.tensor(1.0)-lambda_non_ob)*BCE(object_pred[..., 4], object_conf)

		loss_conf += lambda_non_ob*BCE(single_pred[..., 4], lconf)

	loss = loss_xy + loss_wh + loss_conf + loss_cls
	return loss, [loss_xy, loss_wh, loss_conf, loss_cls]


def label_transform(label, anchors, IOU_threshold=0.5):
	anchor_index = []
	label_index = []
	grid_location_x = []
	grid_location_y = []
	label_xy = []
	label_wh = []
	label_cls = []
	# 与位置无关，只看那个anchor最贴合目标
	for i in range(len(label)):
		max_IOU = 0
		max_index = -1
		for j in range(len(anchors)):
			temp_IOU = IOU([0, 0, anchors[j][0], anchors[j][1]],
						   [0, 0, label[i][2], label[i][3]])

			if temp_IOU > max_IOU:
				max_IOU = temp_IOU
				max_index = j

		if max_IOU > IOU_threshold:
			# 我们放弃预测和先验anchor框大小严重不符的物体
			anchor_index.append(max_index)
			label_index.append(i)

	if len(label_index):
		# 对于确定要进行预测的标签，我们对其进行坐标变换
		for j, i in enumerate(label_index):
			# 208*208 to 13*13
			grid_x = int(label[i][0] / 16)
			grid_y = int(label[i][1] / 16)
			# imread的图片通道是 高*宽*3
			grid_location_x.append(grid_x)
			grid_location_y.append(grid_y)
			# 预测中心点在这个网格内的偏移量
			label_xy.append([label[i][0]/16 - grid_x, label[i][1]/16 - grid_y])
			# 按YOLO文章中说的那样对宽和高做对数变换\
			label_wh.append([torch.log(torch.tensor(label[i][2]/anchors[anchor_index[j]][0])),
							 torch.log(torch.tensor(label[i][3]/anchors[anchor_index[j]][1]))])
			# 给出label中cls的预测值
			if label[i][4] == 1:
				# face with mask
				label_cls.append(0)
			else:
				# unmask face
				label_cls.append(1)

	return label_xy, label_wh, label_cls, [anchor_index, grid_location_y, grid_location_x]


if __name__ == "__main__":
	train()