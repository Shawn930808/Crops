import torch
import math
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

def rotation_transform_2d(rotcenter, rot):
	t = torch.eye(3)
	cosr = math.cos(rot)
	sinr = math.sin(rot)
	tx, ty = rotcenter[0],rotcenter[1]
	t[0][2] = -tx
	t[1][2] = -ty

	ti = torch.eye(3)
	ti[0][2] = tx
	ti[1][2] = ty

	r = torch.eye(3)
	r[0][0] = cosr
	r[0][1] = -sinr
	r[1][0] = sinr
	r[1][1] = cosr

	return ti.mm(r.mm(t))

def translation_scale_transform_2d(translation, scale):
	t = torch.eye(3)
	tx, ty = translation[0],translation[1]
	t[0][2] = -tx
	t[1][2] = -ty

	s = torch.eye(3)
	s[0][0] = scale
	s[1][1] = scale

	return s.mm(t)

def rotate_tensor(image, theta):
	# Currently converts to and from a PIL image - slow!
	# Pytorch will eventually contain image rotation code as torch does
	pil_image = to_pil_image(image)
	return to_tensor(pil_image.rotate((theta * 180.0) / 3.1415926, Image.BILINEAR, False, None))

import cv2

def rotate_tensor_np(image, theta):
	# Currently converts to and from a PIL image - slow!
	# Pytorch will eventually contain image rotation code as torch does
	npimage = image.numpy().transpose([1,2,0])

	num_rows, num_cols = npimage.shape[:2]
	rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
	img_rotation = cv2.warpAffine(npimage, rotation_matrix, (num_cols, num_rows))
	return to_tensor(img_rotation)

