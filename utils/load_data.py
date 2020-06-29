# -*- coding:utf-8 -*-
"""
This file provides functions for data loading.
"""

from utils.image_process import read_all_image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime
from PIL import Image


def load_data_for_train(folder, batch_size=0):
	total_image, total_labels = read_all_image(folder)
	imgs = torch.from_numpy(total_image)
	label_index = torch.from_numpy(np.expand_dims(np.arange(imgs.shape[0]), axis=0)).view(-1, 1)
	img_data = TensorDataset(imgs, label_index)
	if batch_size == 0:
		batch_size = imgs.shape[0]
	data_loader = DataLoader(dataset=img_data,
							 batch_size=batch_size,
							 shuffle=True,
							 num_workers=2)
	return data_loader, total_labels


if __name__ == "__main__":
	t1 = datetime.datetime.now()
	data, label = load_data_for_train("../minival_resized", 64)
	t2 = datetime.datetime.now()
	print(t2-t1)
	for i, batch_data in enumerate(data):
		imgs, label_index = batch_data
		print(label_index.shape[0])
		img = np.array(np.array(imgs[0]*255).transpose((1, 2, 0)))
		Image.fromarray(np.uint8(img)).show()