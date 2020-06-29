# -*- coding:utf-8 -*-
"""
This file is used to define the neural network for yolo model.
"""

import torch
import torch.nn as nn
import numpy as np


class YOLONet(nn.Module):
	def __init__(self):
		super(YOLONet, self).__init__()
		# 我的网络结构是确定的，所以不需要输入参数
		# Layer 1:
		# 16通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		# 2*2的max pooling，stride=2
		self.con1 = nn.Conv2d(3, 16, 3, padding=1)
		self.active1 = nn.LeakyReLU(0.1)
		self.maxpool1 = nn.MaxPool2d(2, 2)

		# Layer 2:
		# 32通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		# 2*2的max pooling，stride=2
		self.con2 = nn.Conv2d(16, 32, 3, padding=1)
		self.active2 = nn.LeakyReLU(0.1)
		self.maxpool2 = nn.MaxPool2d(2, 2)

		# Layer 3:
		# 64通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		# 2*2的max pooling，stride=2
		self.con3 = nn.Conv2d(32, 64, 3, padding=1)
		self.active3 = nn.LeakyReLU(0.1)
		self.maxpool3 = nn.MaxPool2d(2, 2)

		# Layer 4:
		# 128通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		# 2*2的max pooling，stride=2
		self.con4 = nn.Conv2d(64, 128, 3, padding=1)
		self.active4 = nn.LeakyReLU(0.1)
		self.maxpool4 = nn.MaxPool2d(2, 2)

		# Layer 5:
		# 128通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		self.con5 = nn.Conv2d(128, 128, 3, padding=1)
		self.active5 = nn.LeakyReLU(0.1)

		# Layer 6:
		# 256通道的3*3卷积核，padding=1，stride=1
		# leakyReLU激活函数
		self.con6 = nn.Conv2d(128, 256, 3, padding=1)
		self.active6 = nn.LeakyReLU(0.1)

		# Layer 7:
		# 28通道的1*1卷积核，padding=0，stride=1
		self.con7 = nn.Conv2d(256, 28, 1, padding=0)

	def forward(self, x, pred=False):
		x = self.maxpool1(self.active1(self.con1(x)))
		x = self.maxpool2(self.active2(self.con2(x)))
		x = self.maxpool3(self.active3(self.con3(x)))
		x = self.maxpool4(self.active4(self.con4(x)))
		x = self.active5(self.con5(x))
		x = self.active6(self.con6(x))
		x = self.con7(x)
		return prediction(x, pred)


def prediction(x, pred=False):
	# x 的形状为：batch*filter*height*width
	batch_size = x.shape[0]

	# image size在我的模型中确定为208
	image_size = 208
	stride = image_size // x.shape[2]		# stride = 16
	grid_size = image_size // stride		# grid_size = 13
	bbox_attrs = 7							# 4 locs + 1 conf + 2 cls
	num_anchors = 4
	anchors = [(6.08, 8.14), (15.05, 19.62), (22.45, 30.32), (52.93, 70.13)]
	# 把anchor的大小调整到降采样后的图片大小
	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	# 把 x 的最后两维数据拼在一起
	x = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
	# 将特征维调整到数据的最后一维，方便后续的张量操作
	x = x.transpose(1, 2).contiguous()
	x = x.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

	# 对预测的边界框中心坐标和置信度进行sigmoid
	x[:, :, 0] = torch.sigmoid(x[:, :, 0])
	x[:, :, 1] = torch.sigmoid(x[:, :, 1])
	x[:, :, 4] = torch.sigmoid(x[:, :, 4])

	if pred:		# 只有在预测时需要，训练时不需要
		# 给坐标中心的预测值添加offset
		grid_len = np.arange(grid_size)
		a, b = np.meshgrid(grid_len, grid_len)
		x_offset = torch.tensor(a, dtype=torch.float).view(-1, 1)
		y_offset = torch.tensor(b, dtype=torch.float).view(-1, 1)
		x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

		x[:, :, :2] += x_y_offset

		# 对高和宽进行指数变换
		anchors = torch.tensor(anchors, dtype=torch.float)
		anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
		x[:, :, 2:4] = torch.exp(x[:, :, 2:4])*anchors

		x[:, :, :4] *= stride

		# softmax the class scores
		x[:, :, 5:7] = torch.softmax(x[:, :, 5:7], dim=2)

	# batch_size*height*width*num_anchors*bbox_attrs
	if not pred:
		x = x.view(batch_size, grid_size, grid_size, num_anchors, bbox_attrs)

	return x
