import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import Counter
import pandas as pd
import numpy as np
from model.DatasetTestList import DecathlonData
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader
import nibabel as nib
import multiprocessing

def write_image_3d(writerobject, imagedata, iter, step = 8, nrow=4, name = 'image'):
            writeimage = imagedata[0,:,:,:,::step]
            if writeimage.shape[1]>3:
                writeimage = writeimage[0:3, :,:,:]
            writeimage = writeimage.permute(3,0,1,2)
            #print(writeimage.shape)
            writeimage = torchvision.utils.make_grid(writeimage, nrow = nrow,normalize =True)
            writerobject.add_image('data/' + name, writeimage , iter)


def load_checkpoint(model, optimizer, filename='./log/modelstate.ckpt'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

#---------------------------------------------------------------------------------------
def calc_dice(input, target, smooth = .0001):
    #input = torch.sigmoid(input)
    iflat = input.flatten()
    tflat = target.flatten()

    intersection = np.dot(iflat, tflat)
    return ((2. * intersection ) / (np.sum(iflat) + np.sum(tflat) + smooth))

#---------------------------------------------------------------------------------------
def dice_loss_two_class(input, target, smooth = 1.):
    #input = torch.sigmoid(input)
    iflat = input.contiguous().view(-1).to
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

#---------------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        #print(input.shape, 'loss input shape')
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


#---------------------------------------------------------------------------------------
class DiceLoss_batchproof(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_batchproof, self).__init__()

    def forward(self, input, target):
        loss_sum = 0

        for i in range(input.shape[0]):
            #assert input.shape[1] == 0
            #assert target.shape[1] == 0

            input_element = input[i,:,:,:,:]
            target_element = target[i,:,:,:,:]

            smooth = 1.
            iflat = input_element.contiguous().view(-1)
            tflat = target_element.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            loss =  1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
            #print(i, loss, "batch element loss")
            loss_sum = loss_sum + loss

        return loss_sum/input.shape[0]


#------------------------------------------------------------------------------------------
class DiceLoss_multichannel_batchproof(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(DiceLoss_multichannel_batchproof, self).__init__()

        def forward(self, input, target):
            loss_sum = 0
            for i in range(input.shape[1]):
                input_element = input[:,i,:,:,:]
                target_element = target[:,0,:,:,:] == i
                smooth = 1.
                loss = DiceLoss_batchproof(input_element, target_element)
                loss_sum = loss_sum +loss

            return loss_sum/input.shape[1]

#------------------------------------------------------------------------------------------
class FocalLoss_batchproof(nn.Module):
    def __init__(self, weight=None):
        super(FocalLoss_batchproof, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, input, target):
        #print(input.shape, 'loss input shape')
        loss_sum = 0
        for i in range(input.shape[0]):
            input_element = input[i,:,:,:,:]
            target_element = target[i,:,:,:,:]
            smooth = 1.
            iflat = input_element.contiguous().view(-1)
            tflat = target_element.contiguous().view(-1)
            #mport pdb; pdb.set_trace()
            #target = target.view(-1,1)
            tflat = tflat.float()
            iflat = iflat.float()
            pt = iflat * tflat + (1 - iflat) * (1 - tflat)
            logpt = pt.log()
            at = (1 - self.alpha) * tflat + (self.alpha) * (1 - tflat)
            logpt = logpt * at

            loss = -1 * (1 - pt) ** self.gamma * logpt
            loss = loss.mean()
            loss_sum = loss_sum +loss
            #print(loss_sum)

        return loss_sum/input.shape[0]

#------------------------------------------------------------------------------------------
class FocalLoss_batchproof2(nn.Module):
    def __init__(self, weight=None):
        super(FocalLoss_batchproof2, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, input, target):
        #print(input.shape, 'loss input shape')
        loss_sum = 0
        for i in range(input.shape[0]):
            input_element = input[i,:,:,:,:]
            target_element = target[i,:,:,:,:]
            inputs = input_element.contiguous().view(-1)
            targets = target_element.contiguous().view(-1)

            #if self.logits:
            #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
            #else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            loss_sum = loss_sum +torch.mean(F_loss)
            print(F_loss.shape)
            #if self.reduce:
            #return torch.mean(F_loss)
            #e#lse:
            #return F_loss
        return loss_sum/input.shape[0]

#------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

#------------------------------------------------------------------------------------------
class DiceLoss_multi(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multi, self).__init__()

    def forward(self, input, targetLabels, n_classes, OVERLAP_PENALTY=False):

        #print(input.shape, 'DiceLoss_multi input shape')
        #print(targetLabels.shape, 'DiceLoss_multi target labels shape')

        targetLabels=targetLabels.type(torch.LongTensor)
        dims = targetLabels.shape

        target = torch.LongTensor(dims[0],n_classes+1,dims[2],dims[3],dims[4]).zero_()

        target = target.scatter(1, targetLabels, 1)
        target = target[:,1:,:,:,:] # throwout 0's (background)

        #probs = F.softmax(input)
        probs = input
        probs = probs.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        num=probs*target#b,c,h,w,d--p*g
        num=torch.sum(num,dim=4)#b,c,h
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)

        den1=probs#*probs#--p^2
        den1=torch.sum(den1,dim=4)
        den1=torch.sum(den1,dim=3)
        den1=torch.sum(den1,dim=2)

        den2=target#*target#--g^2
        den2=torch.sum(den2,dim=4)#b,c,h
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c
        #print(den1,den2)
        #print(num, den1, den2)
        dice=2*(num/(den1+den2+0.005))

        if (OVERLAP_PENALTY):
            overlap = probs[:,0,:,:,:]*probs[:,1,:,:,:]
            overlap = torch.sum(overlap,dim=3)
            overlap = torch.sum(overlap,dim=2)
            overlap = torch.sum(overlap,dim=1)
            overlap_fraction = overlap/(den1[:,0]+den1[:,1])

        #dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
        if (OVERLAP_PENALTY):
            dice_total = 1-torch.mean(dice) + overlap_fraction
        else:
            dice_total = 1-torch.mean(dice)

        return dice_total.cuda()


#------------------------------------------------------------------------------------------
def dice_loss_multi(input, targetLabels, n_classes, OVERLAP_PENALTY=False):
    """
    rewritten by D.C. Elton 6-19-19

    input is a torch variable of size BatchxNclassesxHxWxD representing log probabilities for each class
    target is the groundtruth with integer labels, shoud have same size as the input

    """
    #print(input.shape, 'diceloss input shape in dice_loss_multi')
    #print(targetLabels.shape, 'diceloss target shapein dice_loss_multi')

    assert input.dim() == 5, "input to dice_loss_multi() must be a 5D Tensor."

    targetLabels=targetLabels.type(torch.LongTensor)
    dims = targetLabels.shape

    target =  torch.LongTensor(dims[0],n_classes+1,dims[2],dims[3],dims[4]).zero_()

    target = target.scatter(1, targetLabels, 1)
    target = target[:,1:,:,:,:] # throw out 0's (background)

    #probs = F.softmax(input)
    probs = input
    probs = probs.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    num=probs*target#b,c,h,w,d--p*g
    num=torch.sum(num,dim=4)#b,c,h
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)

    den1=probs#*probs#--p^2
    den1=torch.sum(den1,dim=4)
    den1=torch.sum(den1,dim=3)
    den1=torch.sum(den1,dim=2)

    den2=target#*target#--g^2
    den2=torch.sum(den2,dim=4)#b,c,h
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    #print(den1.shape(),den2.shape())
    #print(num, den1, den2)
    dice=2*(num/(den1+den2+0.005))

    if (OVERLAP_PENALTY):
        overlap = probs[:,0,:,:,:]*probs[:,1,:,:,:]
        overlap = torch.sum(overlap,dim=3)
        overlap = torch.sum(overlap,dim=2)
        overlap = torch.sum(overlap,dim=1)
        overlap_fraction = overlap/(den1[:,0]+den1[:,1])

    #dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
    if (OVERLAP_PENALTY):
        dice_total = 1-torch.mean(dice) + 0.5*overlap_fraction
    else:
        dice_total = 1-torch.mean(dice)

    return dice_total


#------------------------------------------------------------------------------------------
def dice_loss_multi_with_level(input, targetLabels, n_classes, targetLevel=None,  OVERLAP_PENALTY=False):
    """
    rewritten by D.C. Elton 6-19-19

    input is a torch variable of size BatchxNclassesxHxWxD representing log probabilities for each class
    target is the groundtruth with integer labels, shoud have same size as the input

    """
    #print(input.shape, 'diceloss input shape in dice_loss_multi')
    #print(targetLabels.shape, 'diceloss target shapein dice_loss_multi')

    assert input.dim() == 5, "input to dice_loss_multi() must be a 5D Tensor."

    targetLabels=targetLabels.type(torch.LongTensor)
    dims = targetLabels.shape

    target =  torch.LongTensor(dims[0],n_classes+1,dims[2],dims[3],dims[4]).zero_()

    target = target.scatter(1, targetLabels, 1)
    target = target[:,1:,:,:,:] # throw out 0's (background)

    #probs = F.softmax(input)
    probs = input
    probs = probs.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    #if (target_level != None):
    #    in_seg_shape = probs.shape
    #    for z in range(in_seg_shape[4]):
    #        tot = sum(probs[])
#
#               tot_pixel_sum = 0
#    avg_height = 0

    for i in range(pz):
        if (z_first):
            this_pixel_sum = np.sum(in_seg[i,:,:])
        else:
            this_pixel_sum = np.sum(in_seg[:,:,i])
        tot_pixel_sum += this_pixel_sum
        avg_height += this_pixel_sum*i

        avg_height = int(avg_height//(tot_pixel_sum+.00001)) #integer


    num=probs*target#b,c,h,w,d--p*g
    num=torch.sum(num,dim=4)#b,c,h
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)

    den1=probs#*probs#--p^2
    den1=torch.sum(den1,dim=4)
    den1=torch.sum(den1,dim=3)
    den1=torch.sum(den1,dim=2)

    den2=target#*target#--g^2
    den2=torch.sum(den2,dim=4)#b,c,h
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    #print(den1.shape(),den2.shape())
    #print(num, den1, den2)
    dice=2*(num/(den1+den2+0.005))

    if (OVERLAP_PENALTY):
        overlap = probs[:,0,:,:,:]*probs[:,1,:,:,:]
        overlap = torch.sum(overlap,dim=3)
        overlap = torch.sum(overlap,dim=2)
        overlap = torch.sum(overlap,dim=1)
        overlap_fraction = overlap/(den1[:,0]+den1[:,1])

    #dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
    if (OVERLAP_PENALTY):
        dice_total = 1-torch.mean(dice) + 0.5*overlap_fraction
    else:
        dice_total = 1-torch.mean(dice)

    return dice_total



#--------------------------------------------------------------------------------
# Untested
def numpy_resize_torch(x, X, Y, Z, mode = 'trilinear'):
    tensor = torch.from_numpy(x)
    tensor = torch.unsqueeze(tensor, dim=0)
    tensor = torch.unsqueeze(tensor, dim=0)
    return F.interpolate(tensor, size=(X,Y,Z) ,mode=mode, align_corners=None).numpy().squeeze()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (
            m1.sum(1) + m2.sum(1) + smooth)

        score = 1 - score.sum() / num
        return score

#----------------------------------------------------------------------------------------------------------
def get_data_loaders(path, originalXY, originalZ,  batch_size, usetasks, augmentversions=0, n_classes=1,
                    CVfold=0, number_train = None, deformfactor=10, testsplits=5, config=None, validationfolds=3):

    print("making train data set generator\n")
    train_dataset = DecathlonData(path , originalXY, originalZ, usetasks, n_augmented_versions=augmentversions, n_classes=n_classes,
                                mode='CVtrain', CVfold=CVfold, testsplits=testsplits, deformfactor=deformfactor,
                                    augment3d=True, nonrigid=False, number_train=number_train, config = config, validationfolds=validationfolds)

    print("making test data set generator\n")
    test_dataset = DecathlonData(path, originalXY, originalZ, usetasks, n_augmented_versions=0, mode='CVtest', n_classes=n_classes,
                                CVfold=CVfold, testsplits=testsplits, config = config, validationfolds=validationfolds)

    print("making val data set generator\n")
    val_dataset = DecathlonData(path, originalXY, originalZ, usetasks, n_augmented_versions=0, mode='CVval', n_classes=n_classes,
                                CVfold=CVfold, testsplits=testsplits, config = config, validationfolds=validationfolds)

    useCPUs = multiprocessing.cpu_count()
    if useCPUs > 32:
        useCPUs =32

    train_sampler = WeightedRandomSampler(train_dataset.weights(), num_samples=len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers= useCPUs)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,  num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,  num_workers=1)

    print('=====> #train, #val, #test = ', len(train_dataset),  len(val_dataset), len(test_dataset))
    #print(train_dataset.allfiles, train_dataset.files)
    return train_loader, validation_loader, test_loader, list(set(train_dataset.tasks))


#----------------------------------------------------------------------------------------------------------
def validate(model, validation_loader, alltasks, path, n_classes=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        os.makedirs(path)
    with torch.no_grad():
        model.train(False)
        results = list()
        for i, data in enumerate(validation_loader):
            inputs = data['data']
            labels = data['seg']
            task = data['task']
            inputs, labels, task = inputs.to(device), labels.to(device), task.to(device)
            result = model(inputs, task, alltasks)
            outputs = result['final_layerA']
            complete = result['complete']
            loss = dice_loss_multi(outputs.to(device), labels.to(device), n_classes=n_classes)
            dice = 1 - loss.detach().cpu().numpy()
            results.append( ( dice, task[0].cpu().numpy()[0] ) )
            numpy_segs = np.swapaxes(complete.cpu().numpy(), 1, 4).squeeze()
            nib.Nifti1Image(numpy_segs, np.eye(4)).to_filename(
                os.path.join(path,str(i) + '_' + '_val_complete.nii'))

    resultpd = pd.DataFrame(results,  columns = ('dice', 'task'))
    model.train(True)
    return 1-np.mean(resultpd['dice']), resultpd



class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
