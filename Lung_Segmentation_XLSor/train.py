'''
Youbao Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
April 2019

For training, you need to modify some arguments according to your own setting and run the command "python train.py".
'''

import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import os
import os.path as osp
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataSet

import timeit
from tensorboardX import SummaryWriter
from utils.utils import inv_preprocess
from utils.encoding import DataParallelModel, DataParallelCriterion

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 16
DATA_DIRECTORY = './data'
DATA_LIST_PATH = './data/train_list.txt'
IGNORE_LABEL = 0
INPUT_SIZE = '512,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 50000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './XLSor_snapshots/'
WEIGHT_DECAY = 0.0005

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description="XLSor Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='0,1,2,3',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")

    return parser.parse_args()

args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

class Criterion(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.MSELoss(size_average=True)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4

def main():
    writer = SummaryWriter(args.snapshot_dir)
    
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    xlsor = XLSor(num_classes=args.num_classes)
    print(xlsor)

    saved_state_dict = torch.load(args.restore_from)
    new_params = xlsor.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0]=='fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 
    
    xlsor.load_state_dict(new_params)


    model = DataParallelModel(xlsor)
    model.train()
    model.float()
    model.cuda()    

    criterion = Criterion()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(XRAYDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN), 
                    batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, xlsor.parameters()), 'lr': args.learning_rate }],
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)


    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.float().cuda()
        if torch_ver == "0.3":
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images, args.recurrence)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        if i_iter % 100 == 0:
            images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
            if isinstance(preds, list):
                preds = preds[0]
            if isinstance(preds, list):
                preds = preds[0]
            preds = interp(preds)
            for index, img in enumerate(images_inv):
                writer.add_image('Images/'+str(index), torch.from_numpy(img/255.).permute(2,0,1), i_iter)
                writer.add_image('Labels/'+str(index), labels[index], i_iter)
                writer.add_image('preds/'+str(index), (preds[index]>0.5).float(), i_iter)

        print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(xlsor.state_dict(),osp.join(args.snapshot_dir, 'XLSor_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(xlsor.state_dict(),osp.join(args.snapshot_dir, 'XLSor_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
