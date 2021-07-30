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

# Terminal colours
printer = ColourPrinter()

# Arguments
import opts
args = opts.parse()

# Manual seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Root snapshot folder and experiment directory
snapshot_root = path.abspath('./snapshots')
if not path.exists(snapshot_root):
    os.makedirs(snapshot_root)

# Output sub-directory
if args.snapshot_directory is not None:
    output_directory = path.join(snapshot_root, args.snapshot_directory)
else:
    output_directory = path.join(snapshot_root, 'output')

if output_directory is not None and not path.exists(output_directory):
    os.makedirs(output_directory)

# Dynamically load dataset from name
datasets = importlib.import_module('datasets')
dataset_ = getattr(datasets, args.dataset)

trainset = dataset_(args, train=True)
validset = dataset_(args, train=False)

kwargs = {'num_workers': args.threads, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, collate_fn = heatmap_collate, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch, shuffle=True, collate_fn = heatmap_collate, **kwargs)

# Load model
models = importlib.import_module('models')
model_ = getattr(models, args.model)

model = model_(args.stack, args.blocks, args.features, trainset.imagechannelcount, trainset.channelcount)

# Load existing weights
if args.weights is not None:
    state = torch.load(args.weights)
    if state['model'] != args.model or state['features'] != args.features \
        or state['blocks'] != args.blocks or state['stack'] != args.stack:
        raise Exception("Supplied network parameters do not match the saved state")
    model.load_state_dict(state['model_state'])

# Criterion
criterion = nn.MSELoss(size_average=True)

# Using cuda
if args.cuda:
    model = model.cuda()#torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

# Optimizer and LR reduction
optimizer = torch.optim.RMSprop(model.parameters(), 
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma) # TODO: last_epoch=
epoch = 1

if args.optimstate is not None:
    state = torch.load(args.optimstate)
    epoch = state['last_epoch'] + 1
    scheduler.last_epoch = state['last_epoch']
    optimizer.load_state_dict(state['optim_state'])

# Trainer
trainer = Trainer(train_loader, valid_loader, model, criterion, optimizer, args)

# Loggers
header = ['epoch','lr'] + ['f1-{0}'.format(x+1) for x in range(trainset.channelcount)]
formats = ['d','g'] + ['.6f'] * trainset.channelcount
train_logger = Logger(path.join(output_directory,'train.log'), args.optimstate is not None, header, formats)
valid_logger = Logger(path.join(output_directory,'valid.log'), args.optimstate is not None, header, formats)

def write(loss, scores, train=True):
    c = printer.GREEN if train else printer.CYAN
    printer.write("\tTrain", c) if train else printer.write("\tValid", c)
    printer.write(" | Loss: {0:.6f}, F1: ".format(loss))
    printer.write(','.join(["{0:.6f}".format(a) for a in scores]) + '\n', c)

while epoch <= args.epochs:
    print ("Epoch {0:d}".format(epoch))

    # Scheduler and LR
    scheduler.step()
    if epoch != 1 and (epoch - 1) % args.lr_step == 0:
        print("Reducing learning rate to ", optimizer.state_dict()['param_groups'][0]['lr'])

    # Train and calculate loss
    loss, f1scores = trainer.train()
    write(loss, f1scores)

    # Logging
    log_result = {'epoch':epoch, 'lr':optimizer.state_dict()['param_groups'][0]['lr']}
    log_result.update({'f1-{0}'.format(i+1):f1scores[i] for i in range(len(f1scores))})
    train_logger.log(log_result)

    # Validation
    if args.validate != 0 and epoch % args.validate == 0:
        loss = 0.0
        f1scores = [0.0] * trainset.channelcount

        for i in range(args.validate_iters):
            
            # Valid iteration
            iter_loss, iter_f1scores = trainer.valid()

            # Accumulate accuracy
            loss += iter_loss
            f1scores = [sum(x) for x in zip(f1scores, iter_f1scores)]

        loss /= args.validate_iters
        f1scores = [x / args.validate_iters for x in f1scores]

        write(loss, f1scores, False)

        # Logging
        log_result = {'epoch':epoch, 'lr':optimizer.state_dict()['param_groups'][0]['lr']}
        log_result.update({'f1-{0}'.format(i+1):f1scores[i] for i in range(len(f1scores))})
        valid_logger.log(log_result)

    # If taking a snapshot this epoch
    if epoch % args.snapshot == 0:
        print("Saving model")

        training_state = {
            'last_epoch': epoch,
            'optim_state': optimizer.state_dict()
        }
        torch.save(training_state, path.join(output_directory, 'optim_state_{0:d}.pt'.format(epoch)))

        model_state = {
            'model': args.model,
            'features': args.features,
            'stack': args.stack,
            'blocks': args.blocks,
            'input_channel_count': trainset.imagechannelcount,
            'output_channel_count': trainset.channelcount,
            'model_state': model.state_dict()
        }
        torch.save(model_state, path.join(output_directory, 'model_state_{0:d}.pt'.format(epoch)))

    epoch = epoch + 1

exit()