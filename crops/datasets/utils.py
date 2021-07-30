import torch
import math
import collections

# Gaussian Drawing
cache = {}

def gaussian_kernel_2d(size, sigma):
	gaussian = torch.FloatTensor(size, size)
	centre = (size / 2.0) + 0.5

	twos2 = 2 * math.pow(sigma, 2)

	for x in range(1,size+1):
		for y in range(1,size+1):
			gaussian[y-1,x-1] = -((math.pow(x - centre, 2)) + (math.pow(y - centre, 2))) / twos2

	return gaussian.exp()


def draw_gaussian_2d(img, pt, sigma):
	height,width = img.size(0),img.size(1)

	# Draw a 2D gaussian
	# Check that any part of the gaussian is in-bounds
	tmpSize = math.ceil(3 * sigma)

	ul = [math.floor(pt[0] - tmpSize), math.floor(pt[1] - tmpSize)]
	br = [math.floor(pt[0] + tmpSize), math.floor(pt[1] + tmpSize)]

	# If not, return the image as is
	if (ul[0] >= width or ul[1] >= height or br[0] < 0 or br[1] < 0):
		return

	# Generate gaussian
	size = 2 * tmpSize + 1
	if size not in cache:
		cache[size] = gaussian_kernel_2d(int(size), sigma)

	g = cache[size]

	# Usable gaussian range
	g_x = [int(max(0, -ul[0])), int(min(size-1, size + (width - 2 - br[0])))]
	g_y = [int(max(0, -ul[1])), int(min(size-1, size + (height - 2 - br[1])))]
	
	# Image range
	img_x = [int(max(0, ul[0])), int(min(br[0], width - 1))]
	img_y = [int(max(0, ul[1])), int(min(br[1], height - 1))]

	sub_img = img[img_y[0]:img_y[1]+1, img_x[0]:img_x[1]+1]
	torch.max(sub_img, g[g_y[0]:g_y[1]+1, g_x[0]:g_x[1]+1], out=sub_img)


def render_heatmap_2d(img, pts, sd):
	if pts is not None:
		for p in pts:
			draw_gaussian_2d(img, p, sd)

# Image Loading
from torchvision.transforms.functional import to_tensor

def _pil_loader(path):
    from PIL import Image
    with open(path, 'rb') as f:
        img = Image.open(f)
        return to_tensor(img)

def _accimage_loader(path):
    import accimage
    try:
        return to_tensor(accimage.Image(path))
    except IOError:
        return _pil_loader(path)

def get_image_loader():
	try:
		import accimage
		return _accimage_loader
	except ImportError:
		return _pil_loader

from torch.utils.data import DataLoader

def heatmap_collate(batch):
    if isinstance(batch[0], collections.Mapping):
        return {key: heatmap_collate([d[key] for d in batch]) for key in batch[0]}
    elif torch.is_tensor(batch[0]):
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        # This is true when number of threads > 1
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
    elif isinstance(batch[0], collections.Sequence):
        return batch
    else:
        raise TypeError((error_msg.format(type(batch[0]))))
