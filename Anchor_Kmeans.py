# -*- coding:utf-8 -*-

from utils.image_process import read_just_all_labels
from utils.IOU import IOU
import numpy as np


def Initial(labels, num_anchor):
	# 批次初始化
	center = np.zeros((num_anchor, 2))
	label_number = len(labels)

	# 首先随机选择一个作为起始的中心点
	index = np.random.choice(label_number, 1)[0]
	center[0, :] = labels[index, :]
	count = 1
	for i in range(num_anchor - 1):
		distance_list = []

		for label in labels:
			min_distance = 1
			for c in range(count):
				distance = 1 - IOU([0, 0, label[0], label[1]], [0, 0, center[c, 0], center[c, 1]])
				if distance < min_distance:
					min_distance = distance
			distance_list.append(min_distance)

		max_distance = 0
		max_index = 0
		for j in range(label_number):
			if max_distance < distance_list[j]:
				max_distance = distance_list[j]
				max_index = j

		center[count, :] = labels[max_index, :]
		count += 1


	center_index = np.zeros(label_number, dtype=np.int)
	new_center = np.zeros((num_anchor, 2))
	length = np.zeros(num_anchor, dtype=np.int)
	for i in range(label_number):
		label = labels[i]
		min_distance = 1
		index = -1
		for c in range(num_anchor):
			distance = 1 - IOU([0, 0, label[0], label[1]], [0, 0, center[c, 0], center[c, 1]])
			if distance < min_distance:
				min_distance = distance
				index = c
		center_index[i] = index
		new_center[index, :] += label[:]
		length[index] += 1

	for i in range(num_anchor):
		new_center[i, :] /= length[index]

	loss = np.zeros(num_anchor)
	for i in range(label_number):
		label = labels[i]
		index = center_index[i]
		distance = 1 - IOU([0, 0, label[0], label[1]], [0, 0, center[index, 0], center[index, 1]])
		loss[index] += distance

	return new_center, length, loss, center_index


def Kmeans(labels, center, length, loss, center_index, num_anchor):
	label_number = len(labels)
	for i in range(label_number):
		index1 = center_index[i]
		label = labels[i]
		temp_center1 = (center[index1, :]*length[index1] - label)/(length[index1] - 1)
		distance = 0
		for j in range(label_number):
			index2 = center_index[j]
			if index1 == index2 and not i == j:
				label_j = labels[j]
				distance += (1 - IOU([0, 0, label_j[0], label_j[1]], [0, 0, temp_center1[0], temp_center1[1]]))

		min_distance = 1.0e10
		min_center2 = 0
		min_index = 0
		for c in range(num_anchor):
			if not c == index1:
				temp_distance = distance
				temp_center2 = (center[c, :]*length[c] + label)/(length[c] + 1)
				temp_distance += (1 - IOU([0, 0, label[0], label[1]], [0, 0, temp_center2[0], temp_center2[1]]))
				for j in range(label_number):
					index2 = center_index[j]
					if c == index2:
						label_j = labels[j]
						temp_distance += (1 - IOU([0, 0, label_j[0], label_j[1]], [0, 0, temp_center2[0], temp_center2[1]]))

				if temp_distance < min_distance:
					min_distance = temp_distance
					min_center2 = temp_center2
					min_index = c
		if min_distance < (loss[min_index] + loss[index1]):
			center_index[i] = min_index
			loss[index1] = distance
			loss[min_index] = min_distance - distance
			center[index1] = temp_center1
			center[min_index] = min_center2
	return center, length, loss, center_index


if __name__ == "__main__":
	labels = read_just_all_labels("./minitrain_resized")
	print("Finish reading")
	# 将所有的label全部展开成一列
	# 原来的维数为：图片数量*每张图中的label数*6
	# 现在的维数为：所有label的数量*6
	label_expand = []
	for label in labels:
		label_expand.extend(label)
	label_expand = [[l[2], l[3]] for l in label_expand]
	label_expand = np.array(label_expand)

	num_anchor = 6
	center, length, loss, center_index = Initial(label_expand, num_anchor)
	old_loss = loss.sum()
	print(old_loss)
	for c in center:
		print(c[0], c[1])
	print("Finish Initializing")

	loss_convergence = 1e-9
	max_iteration = 100
	center, length, loss, center_index = Kmeans(label_expand, center, length, loss, center_index, num_anchor)
	old_loss = loss.sum()
	print(old_loss)
	for c in center:
		print(c[0], c[1])
	iteration = 1
	while True:
		iteration += 1
		center, length, loss, center_index = Kmeans(label_expand, center, length, loss, center_index, num_anchor)
		new_loss = loss.sum()
		print(new_loss)
		for c in center:
			print(c[0], c[1])
		if abs(new_loss - old_loss) < loss_convergence or iteration > max_iteration:
			break

		old_loss = new_loss

	print("\nResult:")
	for c in center:
		print(c[0], c[1])