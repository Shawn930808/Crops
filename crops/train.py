import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import MNIST
from torch.autograd import Variable
from evaluation import AverageMeter, AccuracyMeter, evaluate_heatmap_batch
from utils import ProgressMonitor
from math import ceil
import gc

class Trainer():
    def __init__(self, train_loader, valid_loader, model, criterion, optimizer, args):
        # Save necessary variables ready for training or validation
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.progress = ProgressMonitor()

    def _step(self, tag):
        # Training or validation, depending on tag
        model = self.model

        if tag == 'train':
            model.train()
            loader = self.train_loader
        else:
            model.eval()
            loader = self.valid_loader

        self.progress.reset()
        step_iteration_count = len(loader)

        # Accuracy and loss
        step_avg_loss = AverageMeter()
        step_accuracy = AccuracyMeter(loader.dataset.channelcount)
        channelthresholds = torch.Tensor(loader.dataset.channelthresholds).unsqueeze(1).t()

        for idx, sample in enumerate(loader):
            input_image = sample['input']
            input_heatmap = sample['heatmap']
            input_scale = sample['scale']
            input_groundtruth = sample['groundtruth']

            if self.args.cuda:
                input_image = input_image.cuda()
                input_heatmap = input_heatmap.cuda(async=True)

            # Input variables
            input_var = torch.autograd.Variable(input_image)
            target_var = torch.autograd.Variable(input_heatmap)

            # Forward pass
            output = model(input_var)
            output_heatmap = output[-1].data.cpu()

            # Calculate loss
            loss = self.criterion(output[0], target_var)
            for j in range(1, len(output)):
                loss += self.criterion(output[j], target_var)
            
            # Gradient and optimization step
            if tag == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
            # Calculate accuracy, loss etc.
            step_avg_loss.update(loss.data[0], output[-1].size(0))
            normalisedthresholds = input_scale.unsqueeze(1).mm(channelthresholds)

            eval_result = evaluate_heatmap_batch(output_heatmap, input_groundtruth, 0.5, normalisedthresholds)
            step_accuracy.update(eval_result)

            self.progress.update(idx+1, step_iteration_count)
            #self.progress(idx+1, step_iteration_count, status='')

            # Clean up
            gc.collect()

        return step_avg_loss.average(), step_accuracy.f1()

    def train(self):
        return self._step('train')

    def valid(self):
        return self._step('valid')
