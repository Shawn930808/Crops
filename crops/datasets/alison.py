import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import json
from .utils import render_heatmap_2d, get_image_loader
from .transforms import rotation_transform_2d, translation_scale_transform_2d, rotate_tensor, rotate_tensor_np
from torchvision.transforms.functional import to_pil_image, to_tensor
import math

class Alison(data.Dataset):

    def __init__(self, args, train=True):
        self.imagefolder = os.path.join(args.data_directory, 'alison/')
        self.datafile = os.path.join(args.cache_directory,'alison.pt')
        self.train = train
        self.args = args

        if not self._check_exists():
            print("Dataset file not found, initialising")            
            a = self._create_datafile()
        else:
            a = torch.load(self.datafile)

        self.data = a['train'] if self.train else a['valid']

        self.imagechannelcount = 3
        self.channelcount = 2
        self.channelthresholds = [ 0.1, 0.1 ]
        self.classcount = 0
        self.load_image = get_image_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.data)
        image = self.load_image(os.path.join(self.imagefolder, self.data[index]['filename']))
        label = self.data[index]['annotation']

        fertile = label['fertile'].clone()
        sterile = label['sterile'].clone()

        use_augmentation = self.train

        capture_dim = self.args.capture_res
        input_dim = self.args.input_res
        output_dim = self.args.output_res
        capture_scale = input_dim / float(capture_dim)
        output_scale = output_dim / float(input_dim)
        capture_pad = self.args.capture_pad
        random_crop = self.args.random_crop
        random_flip = self.args.random_flip
        gaussian_sd = self.args.gaussian_sd
        random_offset = self.args.jitter_size
      
        world_base_length = 100

        img_width = image.size(2)
        img_height = image.size(1)

        # Sanity check on size
        aug_scale = min(max(pow(2,self._rnd(self.args.scale)),0.5),2.0) if use_augmentation else 1

        # Clamped to +- 30 Degrees
        aug_rotate = min(max(pow(2,self._rnd(self.args.rotate)),-0.52),0.52) if use_augmentation else 0
        
        # Random flip variable with 50% probability
        flip = torch.rand(1)[0] > 0.5 if random_flip else False


        # Crop centre choice
        fn, sn = 0, 0
        if fertile.dim() > 0: fn = fertile.size(0)
        if sterile.dim() > 0: sn = sterile.size(0)

        centre = None
        chx = torch.rand(1)[0]

        # 10% of the time use a random position along the image edge
        if chx < 0.1:
            edge = int(torch.rand(1)[0] * 4)
            aug_rotate = aug_rotate * 0.1 # Scale down rotation as we are on an edge
            p = capture_dim / 2;
            r = torch.rand(1)[0];

            if edge == 0:
                centre = torch.Tensor([p, r * img_height])
            elif edge == 1:
                centre = torch.Tensor([img_width - p, r * img_height])
            elif edge == 2:
                centre = torch.Tensor([r * img_width, p])
            else:
                centre = torch.Tensor([r * img_width, img_height - p])
        # 30% of the time use a random image position
        elif chx < 0.4:
            centre = torch.rand(2).mul_(torch.Tensor([img_width,img_height]))
        # 60% of the time, chose a random branch point
        else:
            sbx = (fn + sn)
            if sbx == 0:
                centre = torch.rand(2).mul_(torch.Tensor([img_width,img_height]))
            else:
                spikeidx = int(torch.rand(1)[0] * sbx)

                if spikeidx < fn:
                    centre = fertile[spikeidx].clone()
                else:
                    centre = sterile[spikeidx - fn].clone()

                # Add random jitter
                centre.add_(torch.randn(2) * random_offset)


        # Create crop bounds based on centre point
        cb = torch.Tensor([centre[0] - (capture_dim / 2), centre[1] - (capture_dim / 2), capture_dim, capture_dim]);

        if use_augmentation and aug_scale != 1:
            scaled_dim = capture_dim * aug_scale;
            diff = scaled_dim - capture_dim
            cb[0:2].add_(-(diff/2))
            cb[2:4].mul_(aug_scale)
        
        # Round CB into pixel co-ordinates
        cb.round_()

        # Clone to rotated bounds
        rcb = cb.clone()

        # If rotating, add necessary padding
        rotate_pad = 0
        if use_augmentation and aug_rotate != 0:
            rotate_pad = math.ceil(rcb[2] * (math.sqrt(2) * math.sin((math.pi) / 4 + abs(aug_rotate)) / 2 - 0.5))
            rcb[0:2].add_(-rotate_pad)
            rcb[2:4].add_(2 * rotate_pad)
        
        # Standard bounds checks
        if rcb[0] < 0: rcb[0] = 0
        if rcb[1] < 0: rcb[1] = 0

        if rcb[0] + rcb[2] >= img_width: rcb[0] = img_width - rcb[2] - 1
        if rcb[1] + rcb[3] >= img_height: rcb[1] = img_height - rcb[3] - 1

        # Reset CB in case of bound check movement
        cb[0:2].copy_(torch.add(rcb[0:2], rotate_pad))
        cb[2:4].copy_(torch.add(rcb[2:4], -2 * rotate_pad))

        # Calculate rotation centre
        rotation_centre = torch.Tensor([ cb[0] + cb[2] / 2, cb[1] + cb[3] / 2])

        # RCB crop
        image = image[:, int(rcb[1]):int(rcb[1]+rcb[3]), int(rcb[0]):int(rcb[0] + rcb[2])]

        # Convert to PIL image for further processing - Remove when torch supports tensor transforms
        pil_image = to_pil_image(image)

        # If rotation is required, perform a rotation and then an additional crop to CB
        if use_augmentation and aug_rotate != 0:
            pil_image = pil_image.rotate((aug_rotate * 180.0) / 3.1415926, Image.BILINEAR, False, None)
            pil_image = pil_image.crop((int(rotate_pad), int(rotate_pad), int(rcb[3] - rotate_pad), int(rcb[2] - rotate_pad)))

        # Scale to input resolution
        pil_image = pil_image.resize((input_dim, input_dim), Image.BILINEAR)

        # Flip if required
        if use_augmentation and flip:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Image transformations are done, ground truth etc.:
        world_scale = (1/aug_scale) * capture_scale * output_scale

        mat = translation_scale_transform_2d(cb[0:2], world_scale)
        world_base_length = world_base_length * world_scale

        # Add rotation if in use
        if use_augmentation and aug_rotate != 0:
            rotation_mat = rotation_transform_2d(rotation_centre, -aug_rotate)
            mat = mat.mm(rotation_mat)

        if fn > 0:
            transformed_fertile = mat.mm(torch.cat([fertile,torch.ones(fertile.size(0)).unsqueeze(1)],1).t()).t().ceil()[:,0:2]
        else:
            transformed_fertile = None
        if sn > 0:
            transformed_sterile = mat.mm(torch.cat([sterile,torch.ones(sterile.size(0)).unsqueeze(1)],1).t()).t().ceil()[:,0:2]
        else:
            transformed_sterile = None
        
        # Flip tips if horizontal flipping in use
        if use_augmentation and flip:
            # Flip X co-ord only
            if transformed_fertile is not None:
                transformed_fertile[:,0].mul_(-1).add_(output_dim - 1)

            if transformed_sterile is not None:
                transformed_sterile[:,0].mul_(-1).add_(output_dim - 1)

        fertileGTPoints = self._capture_inbounds(transformed_fertile, output_dim)
        sterileGTPoints = self._capture_inbounds(transformed_sterile, output_dim)

        hm = torch.zeros(self.channelcount, output_dim, output_dim);
        render_heatmap_2d(hm[0], transformed_fertile, gaussian_sd)
        render_heatmap_2d(hm[1], transformed_sterile, gaussian_sd)


        return {
            'input': to_tensor(pil_image),
            'heatmap': hm,
            'scale': world_base_length,
            'groundtruth': [fertileGTPoints, sterileGTPoints]
        }

    def __len__(self):
        return len(self.data) * 100
    
    def _rnd(self, x):
        return max(-2*x,min(2*x,torch.randn(1)[0]*x))

    def _inbounds(self, pt, size):
        return pt[0] >= 0 and pt[0] < size and pt[1] >= 0 and pt[1] < size

    def _capture_inbounds(self, pts, output_dim):
        tbl = []
        if pts is not None:
            for t in range(pts.size(0)):
                if self._inbounds(pts[t], output_dim):
                    tbl.append((pts[t][0], pts[t][1]))
        return tbl

    def _check_exists(self):
        return os.path.exists(self.datafile)

    def _scanfile(self, path):
        jsonpath = os.path.splitext(path)[0] + '.json'

        if os.path.isfile(jsonpath):
            with open(jsonpath) as data_file:    
                jsondata = json.load(data_file)

            metadata = jsondata['metadata']
            data = jsondata['data']

            if metadata['Ignore']:
                return None

            fertile = []
            sterile = []

            for value in data:
                t = value['Type']
                n = value['Name']

                # Currently handles only one point out of a possible list
                row = value['Points'][0].split(',')
                p = [float(row[0]), float(row[1])]

                # Assume t == point - not general!
                if n == 'Fertile':
                    fertile.append(p)
                elif n == 'Sterile':
                    sterile.append(p)

            
            if len(fertile) + len(sterile) == 0:
                return None

            return {
                "fertile": torch.FloatTensor(fertile),
                "sterile": torch.FloatTensor(sterile),
            }

        return None

    def _create_datafile(self):
        print ("Generating dataset cache.")
        allfiles = []
        for root, dirs, files in os.walk(self.imagefolder):
            for filename in files:
                if filename.endswith(('.jpg', '.JPG', '.tif')):
                    allfiles.append(filename)

        imagecount = len(allfiles)
        traincount = int(round(imagecount * 0.8))
        validcount = imagecount - traincount

        shuffle = torch.randperm(imagecount)
        trainidx = shuffle[0:traincount]
        valididx = shuffle[traincount:]

        traindata = []

        # All training files
        print ("Scanning training data...")
        for idx in trainidx:
            filename = allfiles[idx]
            dt = self._scanfile(os.path.join(self.imagefolder, filename))
            if dt:
                traindata.append({'filename': filename, 'annotation': dt})

        print ("Training data size: ", len(traindata))
        
        validdata = []

        print ("Scanning validation data...")
        for idx in valididx:
            filename = allfiles[idx]
            dt = self._scanfile(os.path.join(self.imagefolder, filename))
            if dt:
                validdata.append({'filename': filename, 'annotation': dt})
        
        print ("Validation data size: ", len(validdata))

        
        print ("Saving data...")

        cachedata = {
            'train': traindata,
            'valid': validdata
        }

        torch.save(cachedata, self.datafile)
        return cachedata