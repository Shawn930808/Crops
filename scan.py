import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
from torchvision import transforms
from torch.autograd import Variable
import importlib
from datasets.utils import heatmap_collate
from train import Trainer
from torch.optim.lr_scheduler import StepLR
from utils import ColourPrinter, Logger
import os
import os.path as path
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import gc
from evaluation import nonmaximalsuppression as nms
from rtree import index

def scan_image(img, model, args):
    # Dimensions
    capture_res = args.capture_res
    input_res = args.input_res
    output_res = args.output_res
    capture_scale = input_res / capture_res
    output_scale =  input_res / output_res

    # Apply capture scale to input image
    if capture_scale != 1.0:
        img = img.resize((round(img.width * capture_scale),round(img.height * capture_scale)), Image.BILINEAR)

    # To Tensor
    img_tensor = to_tensor(img).unsqueeze(0)

    # Overlap and stride
    overlap = 0.3
    stride = round(input_res - (input_res * overlap))

    input_width = img_tensor.size(3)
    input_height = img_tensor.size(2)

    # Model setup
    model.eval()

    # Output tensor
    output_tensor = None
    output_points = None
    padding = (output_res * overlap) // 3

    def in_bounds (pt, l, r, t, b):
        return (l or pt[0] > padding) and (r or pt[0] < output_res - padding) and (t or pt[1] > padding) and (b or pt[1] < output_res - padding)

    for y in range(0, input_height, stride):
        for x in range(0, input_width, stride):
            # Bounds check
            x = min(x, input_width - input_res)
            y = min(y, input_height - input_res)
            
            l = x == 0
            r = x == input_width - input_res
            t = y == 0
            b = y == input_height - input_res

            block = Variable(img_tensor[:, :, y:y+input_res, x:x+input_res].cuda())
            output = model.forward(block)[-1].data.cpu()

            if output_points is None:
                output_points = []
                for i in range(output.size(1)):
                    output_points.append([])

            for i in range(output.size(1)):
                pts = nms(output.squeeze()[i], 0.6)
                pruned = [pt for pt in pts if in_bounds(pt, l, r, t, b)]
                translated = [[int(pt[0] * output_scale + x), int(pt[1] * output_scale + y)] for pt in pruned]
                if len(translated) > 0:
                    output_points[i].extend(translated)

            if output_tensor is None:
                output_tensor = torch.zeros(output.size(1), input_height, input_width)
            
            # If upsampling
            if output_scale != 1:
                #output_scaled = F.upsample(output, scale_factor=int(output_scale), mode='bilinear', align_corners=False)[0]
                output_scaled = F.upsample(output, scale_factor=int(output_scale), mode='bilinear')[0].data

            max_tensor = torch.max(output_tensor[:,y:y+input_res,x:x+input_res], output_scaled)
            output_tensor[:,y:y+input_res,x:x+input_res].copy_(max_tensor)

    trees = []
    for c in range(len(output_points)):
        idx = index.Index(interleaved=False)
        distance_threshold = 64 # 8^2
        for i,pt in enumerate(output_points[c]):
            neighbour = next(idx.nearest((pt[0],pt[0],pt[1],pt[1])),None)
            if neighbour is None:
                idx.insert(i, (pt[0], pt[0], pt[1], pt[1]))
            else:
                n_pt = output_points[c][neighbour]
                if ((pt[0] - n_pt[0])**2 + (pt[1] - n_pt[1])**2) > distance_threshold:
                    idx.insert(i, (pt[0], pt[0], pt[1], pt[1]))
        trees.append(idx)
   
    three_channel_output = torch.zeros(3,input_height, input_width)
    three_channel_output[0].copy_(torch.max(output_tensor[0],output_tensor[1]))
    three_channel_output[1].copy_(output_tensor[0])

    counts = []
    for c in range(len(output_points)):
        channel_points = [output_points[c][pt] for pt in trees[c].intersection((-1,1281,-1,1025))]
        counts.append(len(channel_points))
        for ch_p in channel_points:
            three_channel_output[:, min(ch_p[1], input_height - 1), min(ch_p[0], input_width - 1)].fill_(1)

    result_image = to_pil_image(torch.cat([img_tensor.squeeze(),three_channel_output],2))
    return (result_image, counts)

def scan_directory(dir, model, args):
    text_file = open("test/counts.csv", "w")

    for filename in os.listdir(dir):
        im_path = os.path.join(dir, filename)
        print (im_path)
        im = Image.open(im_path)
        result_image, counts = scan_image(im, model, args)
        im.close()
        result_image.save('test/' + os.path.basename(filename) + '.png')
        text_file.write('{0:s},{1:d},{2:d}\n'.format(filename, counts[0], counts[1]))
    
    text_file.close()
