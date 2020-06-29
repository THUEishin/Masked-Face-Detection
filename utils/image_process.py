# -*- coding:utf-8 -*-
import cv2
import numpy as np
from xml.etree import ElementTree as ET
import os


def read_image_and_label(filename, resize_shape=(208, 208), flag=True):
	root = ET.parse(filename + ".xml")
	# 读取图片总宽度和总高度
	width = 0
	height = 0
	leaf = root.find('./size')
	for child in leaf.getchildren():
		if child.tag == "width":
			width = int(child.text)

		if child.tag == "height":
			height = int(child.text)

	# 读取图片中所有的label
	labels = []
	leaves = root.findall('./object')
	for leaf in leaves:
		label = np.zeros(6, dtype=np.int)
		for child in leaf.getchildren():
			if child.tag == "name":
				if child.text == "face_mask":
					label[4] = 1 		# 戴口罩的类别
				else:
					label[5] = 1		# 不戴口罩的类别
			if child.tag == "bndbox":
				for subchild in child.getchildren():
					if subchild.tag == "xmin":
						label[0] = int(subchild.text)
					elif subchild.tag == "ymin":
						label[1] = int(subchild.text)
					elif subchild.tag == "xmax":
						label[2] = int(subchild.text)
					else:
						label[3] = int(subchild.text)
		labels.append(label)

	# 读取图片并且进行缩放
	ratio = min(resize_shape[0] / width, resize_shape[1] / height)
	resized_width = int(ratio * width)
	resized_height = int(ratio * height)
	start_h = (resize_shape[1] - resized_height) // 2
	start_w = (resize_shape[0] - resized_width) // 2
	image = None
	canvas = None
	if flag:
		image = cv2.imread(filename + '.jpg')
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
		canvas = np.full((resize_shape[1], resize_shape[0], 3), 128)
		canvas[start_h:start_h+resized_height, start_w:start_w+resized_width, :] = image_resized

	# 修正labels
	for i in range(len(labels)):
		lw = (labels[i][2] - labels[i][0])*ratio
		lh = (labels[i][3] - labels[i][1])*ratio
		x_center = (labels[i][2] + labels[i][0])*ratio/2 + start_w
		y_center = (labels[i][3] + labels[i][1])*ratio/2 + start_h
		labels[i][0] = int(x_center)
		labels[i][1] = int(y_center)
		labels[i][2] = lw
		labels[i][3] = lh

	return image, canvas, labels, [start_w, start_h, ratio]


def Plot_image(filename, image, labels, coeff=(0, 0, 1), threshold=0.5):
	#cv2.imwrite(filename + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	for i in range(len(labels)):
		x_center = (labels[i][0] - coeff[0])/coeff[2]
		y_center = (labels[i][1] - coeff[1])/coeff[2]
		lw = labels[i][2]/coeff[2]
		lh = labels[i][3]/coeff[2]
		xmin = int(x_center - lw/2)
		ymin = int(y_center - lh/2)
		xmax = int(x_center + lw/2)
		ymax = int(y_center + lh/2)
		if labels[i][-2] > threshold:
			color = (0, 255, 0)
		else:
			color = (255, 0, 0)
		cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

	cv2.imwrite(filename + "_with_box.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_all_image(folder, resize_shape=(208, 208)):
	dir_list = os.listdir(folder)
	flag = 1
	total_image = None
	total_labels = []
	for file in dir_list:
		filename = file.split('.')
		if filename[1] == "xml":
			_, image, labels, _ = read_image_and_label(folder + "/" + filename[0], resize_shape)
			image = np.expand_dims(image/255.0, axis=0)
			image = image.transpose((0, 3, 1, 2))
			if flag:
				flag = 0
				total_image = image
				total_labels.append(labels)
			else:
				total_image = np.concatenate((total_image, image), axis=0)
				total_labels.append(labels)

	return total_image, total_labels


def XML_Lost_Value(filename):
	# 存在大量的xml文件没有图像的宽和高，这里做一下数据清洗
	tree = ET.parse(filename + ".xml")
	root = tree.getroot()
	leaf = root.find("./size")
	image = cv2.imread(filename + '.jpg')
	if leaf == None:
		node = ET.Element('size')
		width = ET.Element('width')
		height = ET.Element('height')
		depth = ET.Element('depth')
		width.text = str(image.shape[1])
		height.text = str(image.shape[0])
		depth.text = str(image.shape[2])
		node.append(width)
		node.append(height)
		node.append(depth)
		root.append(node)
		tree.write(filename + ".xml", encoding='utf-8', xml_declaration=False)
	else:
		width = leaf.find("./width")
		height = leaf.find("./height")
		depth = leaf.find("./depth")
		width.text = str(image.shape[1])
		height.text = str(image.shape[0])
		depth.text = str(image.shape[2])
		tree.write(filename + ".xml", encoding='utf-8', xml_declaration=False)


def Lost_Value(folder):
	# 存在大量的xml文件没有图像的宽和高，这里做一下数据清洗
	dir_list = os.listdir(folder)
	for file in dir_list:
		filename = file.split('.')
		if filename[1] == "xml":
			root = ET.parse(folder + "/" + filename[0] + ".xml")
			leaf = root.find("./size")
			if leaf == None:
				print(1, filename[0])
				XML_Lost_Value(folder + "/" + filename[0])
			else:
				width = leaf.find("./width")
				height = leaf.find("./height")
				if int(width.text) == 0 or int(height.text) == 0:
					print(2, filename[0])
					XML_Lost_Value(folder + "/" + filename[0])


def read_just_all_labels(folder, resize_shape=(208, 208)):
	dir_list = os.listdir(folder)
	total_labels = []
	for file in dir_list:
		filename = file.split('.')
		if filename[1] == "xml":
			try:
				_, _, labels, _ = read_image_and_label(folder + "/" + filename[0], resize_shape, False)
				total_labels.append(labels)
				for i in range(len(labels)):
					if labels[i][2] < 1e-5 or labels[i][3] < 1e-5:
						print(filename[0])
			except:
				print(filename[0])

	return total_labels


def Resize_train_and_valid(folder_from, folder_to, resize_shape=(208, 208)):
	dir_list = os.listdir(folder_from)
	for file in dir_list:
		filename = file.split('.')
		if filename[1] == "xml":
			_, canvas, labels, _ = read_image_and_label(folder_from + "/" + filename[0], resize_shape, True)
			cv2.imwrite(folder_to + "/" + filename[0] + ".jpg", canvas)
			root = ET.Element('annotation')
			node = ET.Element('size')
			width = ET.Element('width')
			height = ET.Element('height')
			depth = ET.Element('depth')
			width.text = str(canvas.shape[1])
			height.text = str(canvas.shape[0])
			depth.text = str(canvas.shape[2])
			node.append(width)
			node.append(height)
			node.append(depth)
			root.append(node)

			# 增加object
			for i in range(len(labels)):
				node = ET.Element('object')
				name = ET.Element('name')
				if labels[i][4] == 1:
					name.text = 'face_mask'
				elif labels[i][5] == 1:
					name.text = 'face'
				else:
					name.text = 'unkown'
				node.append(name)
				bndbox = ET.Element('bndbox')
				xmin = ET.Element('xmin')
				ymin = ET.Element('ymin')
				xmax = ET.Element('xmax')
				ymax = ET.Element('ymax')
				xmin.text = str(int(labels[i][0] - labels[i][2]/2))
				ymin.text = str(int(labels[i][1] - labels[i][3]/2))
				xmax.text = str(int(labels[i][0] + labels[i][2]/2))
				ymax.text = str(int(labels[i][1] + labels[i][3] / 2))
				bndbox.append(xmin)
				bndbox.append(ymin)
				bndbox.append(xmax)
				bndbox.append(ymax)
				node.append(bndbox)
				root.append(node)

			tree = ET.ElementTree(root)
			tree.write(folder_to + "/" + filename[0] + ".xml", encoding='utf-8', xml_declaration=False)


if __name__ == "__main__":
	# Lost_Value("../val")
	# Resize_train_and_valid("../minitrain", "../minitrain_resized")
	# filename = "../train/test_00000414"
	# filename = "../train/1_Handshaking_Handshaking_1_59"

	# Train
	# filename = "../minitrain/1_Handshaking_Handshaking_1_59"
	# image, _, labels, coeff = read_image_and_label(filename)
	# Plot_image("../../fig/train", image, labels, coeff)

	# Validation
	# filename = "../minival/4_Dancing_Dancing_4_21"
	# image, _, labels, coeff = read_image_and_label(filename)
	# Plot_image("../../fig/validation", image, labels, coeff)

	# Test
	# filename = "../minitest/32_Worker_Laborer_Worker_Laborer_32_209"
	# image, _, labels, coeff = read_image_and_label(filename)
	# Plot_image("../../fig/test", image, labels, coeff)

	# Resized Train
	filename = "../minitrain_resized/1_Handshaking_Handshaking_1_59"
	image, _, labels, coeff = read_image_and_label(filename)
	Plot_image("../../fig/train_resized", image, labels, coeff)

	# Validation
	filename = "../minival_resized/4_Dancing_Dancing_4_21"
	image, _, labels, coeff = read_image_and_label(filename)
	Plot_image("../../fig/validation_resized", image, labels, coeff)
