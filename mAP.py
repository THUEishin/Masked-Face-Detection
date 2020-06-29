# -*- coding:utf-8 -*-
"""
This file provides function to calculate mAP on test dataset
"""
from inference import inference
import matplotlib.pyplot as plt
from utils.IOU import IOU


def Precision_and_Recall(IOU_threshold, total_label, cls_pred, cls_number):
	pred_count = 0
	positive_count = 0
	true_positive_count = 0
	batch_size = len(total_label)
	for index in range(batch_size):
		label = total_label[index]
		pred = cls_pred[index]
		for i in range(len(label)):
			if label[i][cls_number-2] > 0.5:
				positive_count += 1

		for i in range(pred.shape[0]):
			pred_count += 1
			for j in range(len(label)):
				if label[j][cls_number - 2] > 0.5:
					if IOU(pred[i][:4], label[j][:4]) > IOU_threshold:
						true_positive_count += 1
						break

	if pred_count == 0:
		precision = 1.0
	else:
		precision = true_positive_count/pred_count

	recall = true_positive_count/positive_count

	return precision, recall


def mAP(IOU_threshold, img_filename, model_filename):
	out_folder = "./infer"
	presicion_face_mask = []
	recall_face_mask = []
	presicion_face = []
	recall_face = []
	confidence_list = [0.05*(20-i) for i in range(20)]
	confidence_list.append(0.01)
	confidence_list.append(0.005)
	confidence_list.append(0.001)
	for index in range(23):
		confidence = confidence_list[index]
		print(confidence)
		total_labels, pred_face_mask, pred_face = inference(img_filename, model_filename, out_folder, confidence, True, False)
		precision, recall = Precision_and_Recall(IOU_threshold, total_labels, pred_face_mask, 0)
		presicion_face_mask.append(precision)
		recall_face_mask.append(recall)

		precision, recall = Precision_and_Recall(IOU_threshold, total_labels, pred_face, 1)
		presicion_face.append(precision)
		recall_face.append(recall)

	presicion_face_mask.append(0)
	recall_face_mask.append(1)
	presicion_face.append(0)
	recall_face.append(1)
	plt.plot(recall_face_mask, presicion_face_mask, label='Masked Face')
	plt.plot(recall_face, presicion_face, label='Unmasked Face')
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.legend()
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("400 Epoch model @.5")
	plt.show()
	print('\n')
	for i in range(23):
		print(presicion_face_mask[i], recall_face_mask[i])

	print('\n')
	for i in range(23):
		print(presicion_face[i], recall_face[i])


if __name__ == "__main__":
	img_filename = "./minitest"
	model_filename = "./temp/model/model_400.pkl"
	IOU_threshold = 0.5
	mAP(IOU_threshold, img_filename, model_filename)