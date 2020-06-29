# -*- coding:utf-8 -*-
"""
This file provides function to do inference based on YOLO
"""
from utils.NMS import NMS
import os
import utils.image_process as im
import numpy as np
import torch


def inference(img_filename, model_filename, out_folder, confidence=0.5, folder_flag=True, plot_flag=True):
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	total_canvas = None
	total_images = []
	total_labels = []
	total_coeffs = []
	total_filename = []
	if folder_flag:
		flag = 1
		dir_list = os.listdir(img_filename)
		for file in dir_list:
			filename = file.split('.')
			if filename[1] == "xml":
				image, canvas, label, coeff = im.read_image_and_label(img_filename + "/" + filename[0])
				canvas = np.expand_dims(canvas / 255.0, axis=0)
				canvas = canvas.transpose((0, 3, 1, 2))
				if flag:
					flag = 0
					total_canvas = canvas
					total_labels.append(label)
					total_images.append(image)
					total_coeffs.append(coeff)
					total_filename.append(filename[0])
				else:
					total_canvas = np.concatenate((total_canvas, canvas), axis=0)
					total_labels.append(label)
					total_images.append(image)
					total_coeffs.append(coeff)
					total_filename.append(filename[0])
	else:
		image, canvas, label, coeff = im.read_image_and_label(img_filename)
		canvas = np.expand_dims(canvas / 255.0, axis=0)
		canvas = canvas.transpose((0, 3, 1, 2))
		total_canvas = canvas
		total_labels.append(label)
		total_images.append(image)
		total_coeffs.append(coeff)
		total_filename.append(img_filename.split('/')[-1].split('.')[0])

	model = torch.load(model_filename)
	total_canvas = torch.from_numpy(total_canvas).float()
	with torch.no_grad():
		pred = model(total_canvas, True)
		cls_face_mask = (pred[:, :, 5] > pred[:, :, 6]).float()
		pred[:, :, 5] = cls_face_mask
		cls_face = (pred[:, :, 5] < pred[:, :, 6]).float()
		pred[:, :, 6] = cls_face

		conf_mask = (pred[:, :, 4] > confidence).float().unsqueeze(2)
		pred = pred*conf_mask

		batch_size = pred.shape[0]
		label_face = []
		label_face_mask = []
		labels = []
		for index in range(batch_size):
			single_pred = pred[index]
			non_zero_index = torch.nonzero(single_pred[:, 4])

			single_pred = single_pred[non_zero_index.squeeze(), :]
			if non_zero_index.shape[0] == 1:
				single_pred = single_pred.unsqueeze(0)

			# class face with mask
			non_zero_face_mask = torch.nonzero(single_pred[:, 5])
			face_mask_pred = single_pred[non_zero_face_mask.squeeze(), :]
			if non_zero_face_mask.shape[0] == 1:
				face_mask_pred = face_mask_pred.unsqueeze(0)
			conf_sort_index = torch.sort(face_mask_pred[:, 4], descending=True)[1]
			face_mask_pred = face_mask_pred[conf_sort_index]
			face_mask_pred = NMS(face_mask_pred)
			label_face_mask.append(face_mask_pred)
			# class face
			non_zero_face = torch.nonzero(single_pred[:, 6])
			face_pred = single_pred[non_zero_face.squeeze(), :]
			if non_zero_face.shape[0] == 1:
				face_pred = face_pred.unsqueeze(0)
			conf_sort_index = torch.sort(face_pred[:, 4], descending=True)[1]
			face_pred = face_pred[conf_sort_index]
			face_pred = NMS(face_pred)
			label_face.append(face_pred)

			labels.append(torch.cat((face_mask_pred, face_pred), 0))

	if plot_flag:
		for index in range(batch_size):
			im.Plot_image(out_folder+'/'+total_filename[index], total_images[index], labels[index], total_coeffs[index])

	return total_labels, label_face_mask, label_face


if __name__ == "__main__":
	out_dir = "./infer"
	image_folder = "./test_images"
	model_file = "./temp/model/model_400.pkl"
	inference(image_folder, model_file, out_dir)