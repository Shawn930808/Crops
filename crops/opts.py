import argparse
import torch

def parse():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Crop Training')
    
    # General
    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
        help='ID of the GPU to be used (default: 0)')
    parser.add_argument('--threads', type=int, default=4, metavar='T',
        help='Number of threads to be used (default: 4)')
    parser.add_argument('--snapshot-directory', type=str, metavar='SDIR',
        help='Output directory for snapshots and log files')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--cache-directory', type=str, metavar='CDIR', default='./cache',
        help='Cache directory location for dataset files')
    parser.add_argument('--data-directory', type=str, metavar='DDIR', default='./datasets',
        help='Root directory for dataset locations')

    # Model Parameters
    parser.add_argument('--model', type=str, default='StackedHourglass', metavar='M',
        help='Deep network model to use (default: hg)')
    parser.add_argument('--features', type=int, default=256, metavar='F',
        help='Number of features per layer (default: 256)')
    parser.add_argument('--stack', type=int, default=4, metavar='ST',
        help='Number of hourglass stacks in the network (default: 4)')
    parser.add_argument('--blocks', type=int, default=2, metavar='MO',
        help='Number of residual blocks per location of the network (default: 2)') 
    parser.add_argument('--weights', type=str, metavar='W',
        help='Path to pre-trained network weights')
    parser.add_argument('--optimstate', type=str, metavar='OS',
        help='Path to existing optimisation state')

    # Snapshot and Training options
    parser.add_argument('--snapshot', type=int, default=5, metavar='SN',
        help='How often to take a snapshot of the model (default: 5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
        help='Total number of training epochs (default: 100)')
    parser.add_argument('--train-batch', type=int, default=6, metavar='TB',
        help='Training batch size (default: 6)')
    parser.add_argument('--valid-batch', type=int, default=1, metavar='VB',
        help='Validation batch size (default: 1)')
    parser.add_argument('--validate', type=int, default=5, metavar='V',
        help='How often to validate the accuracy of the network (default: 5)')
    parser.add_argument('--validate-iters', type=int, default=1, metavar='VI',
        help='How many validation epochs to use per test (default: 1)')

    # Hyperparameters - Currently no support for optimizer specific params such as rmsprop.rho
    parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
        help='Learning rate (default: 2.5e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='LRD',
        help='Learning rate decay (default: 0.0)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='LRD',
        help='Optimizer momentum (default: 0.0)')
    parser.add_argument('--lr-step', type=int, default=50, metavar='LRS',
        help='After how many epochs is the learning rate reduced by gamma (default: 50)')
    parser.add_argument('--lr-step-gamma', type=float, default=0.1, metavar='LRSG',
        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--criterion', type=str, default='MSE', metavar='C',
        help='The criterion to use to calculate a loss (default: MSE)')
    parser.add_argument('--optimizer', type=str, default='rmsprop', metavar='O',
        help='The optimizer to use to adjust the network weights (default: rmsprop)')

    # Data Options
    parser.add_argument('--dataset', type=str, default='RiceBranching', metavar='D',
        help='The dataset to use (default: RiceBranching)')
    parser.add_argument('--capture-res', type=int, default=512, metavar='CR',
        help='Crop resolution of the input image prior to augmentation (default: 512)')
    parser.add_argument('--capture-pad', type=int, default=32, metavar='CP',
        help='Padding to add around captured object, if required (default: 32)')
    parser.add_argument('--input-res', type=int, default=256, metavar='IR',
        help='Input resolution of the network (default: 256)')
    parser.add_argument('--output-res', type=int, default=64, metavar='OR',
        help='Output resolution of the network (default: 64)')
    parser.add_argument('--image', type=str, metavar='IM',
        help='Input image if direct inference is required')

    # Augmentation
    parser.add_argument('--scale', type=float, default=0.25, metavar='SC',
        help='Degree of scale augmentation (default: 0.25)')
    parser.add_argument('--rotate', type=float, default=0.25, metavar='RT',
        help='Degree of rotation augmentation (default: 0.25)')
    parser.add_argument('--no-random-jitter', action='store_true', default=False,
        help='Disables any random jitter augmentation')
    parser.add_argument('--jitter-size', type=int, default=30, metavar='JT',
        help='Maximum random jitter to add, if enabled (default: 30)')
    parser.add_argument('--no-random-crop', action='store_true', default=False,
        help='Disables any random cropping augmentation')
    parser.add_argument('--no-random-flip', action='store_true', default=False,
        help='Disables any random flipping augmentation')
    parser.add_argument('--gaussian-sd', type=float, default=2.0, metavar='SD',
        help='Heatmap gaussian size (default: 2.0)')

    args = parser.parse_args()

    # Flip "negative" variables
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.random_flip = not args.no_random_flip
    args.random_crop = not args.no_random_crop
    args.random_jitter = not args.no_random_jitter

    delattr(args, 'no_cuda')
    delattr(args, 'no_random_flip')
    delattr(args, 'no_random_crop')
    delattr(args, 'no_random_jitter')

    return args