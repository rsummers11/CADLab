import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import Counter
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader
import nibabel as nib
import multiprocessing
from torch.autograd import Variable




#------------------------------------------------------------------------------
def avg_x_seg(in_seg, z_first=False):
    '''
        returns (weighted) average x coordinate (left-right) of segementation
        rounded to nearest integer for indexing
    '''
    if (z_first):
        px = in_seg.shape[1]
    else:
        px = in_seg.shape[0]

    tot_pixel_sum = 0
    avg_x = 0

    for i in range(px):
        if (z_first):
            this_pixel_sum = np.sum(in_seg[:,i,:])
        else:
            this_pixel_sum = np.sum(in_seg[i,:,:])
        tot_pixel_sum += this_pixel_sum
        avg_x += this_pixel_sum*i

    return int(avg_x//(tot_pixel_sum+.00001)) #integer


#------------------------------------------------------------------------------
def avg_z_seg(in_seg):
    '''
        returns (weighted) average x coordinate (left-right) of segementation
        rounded to nearest integer for indexing
    '''

    pz = in_seg.shape[2]

    tot_pixel_sum = 0
    avg_z = 0

    for i in range(pz):
        this_pixel_sum = np.sum(in_seg[:,:,i])
        tot_pixel_sum += this_pixel_sum
        avg_z += this_pixel_sum*i

    return int(avg_z//(tot_pixel_sum+.00001)) #integer

#------------------------------------------------------------------------
def measure_z_bounds(seg):
    zsum = np.sum(seg, axis=(0,1))
    minn = 1000
    maxx = 0
    for i in range(seg.shape[2]):
        if (zsum[i] > 0):
            if i < minn:
                minn = i
            elif i > maxx:
                maxx = i

    return minn, maxx

#-------------------------------------------------------------------------------
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1).contiguous()




#---------------------------------------------------------------------------------------
class FP_and_FN_batch_proof_multi(nn.Module):
    def __init__(self, weight=None, size_average=True,n_classes=1):
        super(FP_and_FN_batch_proof_multi, self).__init__()
        self.n_classes=n_classes

    def forward(self, input, target, n):
        loss_sum = 0

        lamb_min = 0.0001

        nmax = 50000

        v_n = (n - nmax/2)/(nmax/10)

        lamb =  lamb_min + (1- lamb_min)/(1 + np.exp(-v_n))

        #NOTE - having trouble with transfer of this computation to GPU - backward() not working here
        input = input.float()
        dims = target.shape
        #target = torch.LongTensor(dims[0],self.n_classes+1,dims[2],dims[3],dims[4]).zero_()
        #faster to do this on the GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #target = target.scatter(1, targetLabels.long(), 1)
        #target = target[:,1:,:,:,:] # throwout class 0 (background)
        target = target.float()
        target = target.to(device)

        for i in range(input.shape[0]):
            input_element = input[i,:,:,:,:]
            target_element = target[i,:,:,:,:]

            pflat = input_element.contiguous().view(-1)
            tflat = target_element.contiguous().view(-1)

            loss =  lamb*(1 - tflat)*pflat + tflat*(1 - pflat)

            #if (i == 0):
            #    loss = 100*loss.sum() #weight first class the most
            #else:
            loss = loss.sum()

            loss_sum = loss_sum + loss

        loss = loss_sum/(input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3]*input.shape[4])

        return loss


#-------------------------------------------------------------------------------
class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    #@staticmethod
    def forward(self, input, target):
        # get probabilities from logits
        #input = self.normalization(input) #commented out by D.C. Elton

        if (input.size() != target.size()):
            print(input.size())
            print(target.size())

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)

# Loss Prediction Loss
#-------------------------------------------------------------------------------
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

#-------------------------------------------------------------------------------
class MarginRankingLoss_learning_loss(nn.Module):
    # from https://github.com/seominseok0429/Learning-Loss-for-Active-Learning-Pytorch (may not be finished..)
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = inputs[:inputs.size(0)//2]
        pred_lossj = inputs[inputs.size(0)//2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0)//2]
        target_lossj = target_loss[inputs.size(0)//2:]
        final_target = torch.sign(target_lossi - target_lossj)

        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')

#-------------------------------------------------------------------------------
class MarginRankingLoss_learning_loss(nn.Module):
    # from https://github.com/seominseok0429/Learning-Loss-for-Active-Learning-Pytorch (may not be finished..)
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = inputs[:inputs.size(0)//2]
        pred_lossj = inputs[inputs.size(0)//2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0)//2]
        target_lossj = target_loss[inputs.size(0)//2:]
        final_target = torch.sign(target_lossi - target_lossj)

        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')


#---------------------------------------------------------------------------------------
class DiceLoss_batchproof_learned_loss(nn.Module):
    def __init__(self, weight=None, size_average=True, return_loss_for_each_batch=False):
        super(DiceLoss_batchproof_learned_loss, self).__init__()
        self.return_loss_for_each_batch = return_loss_for_each_batch

    def forward(self, input, target, device):
        loss_sum = 0
        losses = torch.empty((input.shape[0], 1)).to(device)

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
            losses[i,0] = loss
            loss_sum = loss_sum + loss

        if (self.return_loss_for_each_batch):
            return loss_sum/input.shape[0], losses
        else:
            return loss_sum/input.shape[0]


#-----------------------------------------------------------------------------
def dice_numpy(im1, im2, smooth = 1):
    #input = torch.sigmoid(input)
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        print(im1.shape)
        print(im2.shape)
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum())/ (im1.sum() + im2.sum() + smooth)

#-------------------------------------------------------------------------------
def dice_numpy_multiclass(im1, im2, n_classes, smooth = 1, verbose=False):

    dice_scores = []

    for ic in range(1, n_classes+1):
        im1_c = np.where(im1 == ic, 1, 0)
        im2_c = np.where(im2 == ic, 1, 0)
        dice = dice_numpy(im1_c, im2_c)
        dice_scores += [dice]

    if (verbose):
        print("dice scores for all classes = ", dice_scores)

    return np.average(dice_scores), dice_scores
#-------------------------------------------------------------------------------
def write_image_3d(writerobject, imagedata, iter, step = 8, nrow=4, name = 'image'):
            writeimage = imagedata[0,:,:,:,::step]
            if writeimage.shape[1]>3:
                writeimage = writeimage[0:3, :,:,:]
            writeimage = writeimage.permute(3,0,1,2)
            #print(writeimage.shape)
            writeimage = torchvision.utils.make_grid(writeimage, nrow = nrow,normalize =True)
            writerobject.add_image('data/' + name, writeimage , iter)

#---------------------------------------------------------------------------------------
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
def dice_loss(input, target, smooth = 1.):
    #input = torch.sigmoid(input)
    iflat = input.contiguous().view(-1)
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
    def __init__(self, reweight=False, size_average=True):
        super(DiceLoss_batchproof, self).__init__()
        self.reweight = reweight

    def forward(self, input, target):
        loss_sum = 0
        total_tflat_sum = 0

        for i in range(input.shape[0]):
            input_element = input[i,:,:,:,:]
            target_element = target[i,:,:,:,:]

            smooth = 1.
            iflat = input_element.contiguous().view(-1)
            tflat = target_element.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            tflat_sum = tflat.sum()
            loss =  1 - ((2. * intersection  + smooth) / (iflat.sum() + tflat_sum  + smooth))
            #print(i, loss, "batch element loss")
            if (self.reweight):
                loss = (1/tflat_sum)*loss
                total_tflat_sum += tflat_sum

            loss_sum = loss_sum + loss

        if (self.reweight):
            return total_tflat_sum*loss_sum/input.shape[0]
        else:
            return loss_sum/input.shape[0]

#---------------------------------------------------------------------------------------
def dice_loss_mulit_class_simple(input, target):
    loss_sum = 0

    for i in range(input.shape[0]):
        input_element = input[i,:,:,:,:]
        target_element = target[i,:,:,:,:]

        smooth = 1.
        iflat = input_element.contiguous().view(-1)
        tflat = target_element.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        loss =  1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        loss_sum = loss_sum + loss

    return loss_sum/input.shape[0]

#------------------------------------------------------------------------------------------
class FocalLoss_batchproof(nn.Module):
    def __init__(self, weight=None, gamma=2, alpha=1):
        super(FocalLoss_batchproof, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def forward(self, input, targetLabels):

        targetLabels=targetLabels.type(torch.LongTensor)
        dims = targetLabels.shape

        n_classes = input.shape[1]

        target = torch.LongTensor(dims[0],n_classes+1,dims[2],dims[3],dims[4]).zero_()

        target = target.scatter(1, targetLabels, 1)

        target = target[:,1:,:,:,:] # throwout class 0 (background)

        target = target.to(self.device)
        input = input.to(self.device)

        #print(input.shape, 'loss input shape')
        #print(target.shape, 'loss target shape')

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
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

#------------------------------------------------------------------------------------------
class DiceLoss_multi(nn.Module):
    ''' Note : this is actually generalized dice loss, but we leave the name the same for historical reasons'''
    def __init__(self, weight=None, size_average=False, n_classes=1, OVERLAP_PENALTY=False):
        super(DiceLoss_multi, self).__init__()
        self.OVERLAP_PENALTY = OVERLAP_PENALTY

    def forward(self, input, target):
        #print(input.shape, 'DiceLoss_multi input shape')
        #print(targetLabels.shape, 'DiceLoss_multi target labels shape')

        n_classes = input.shape[1]
        dims = target.shape
        #targetLabels=targetLabels.type(torch.LongTensor)
        #target = torch.LongTensor(dims[0],n_classes+1,dims[2],dims[3],dims[4]).zero_()
        #target = target.scatter(1, targetLabels, 1)
        #target = target[:,1:,:,:,:] # throwout class 0 (background)

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

        if (self.OVERLAP_PENALTY):
            total_overlap = 0
            for i in range(0,n_classes):
                for j in range(i+1,n_classes):
                    this_overlap = probs[:,i,:,:,:]*probs[:,j,:,:,:]
                    this_overlap = torch.sum(this_overlap,dim=3)
                    this_overlap = torch.sum(this_overlap,dim=2)
                    this_overlap = torch.sum(this_overlap,dim=1)
                    total1 = torch.sum(den1[:,i]) #sum over batches
                    total2 = torch.sum(den1[:,j]) #sum over batches
                    overlap_fraction = this_overlap/(total1+total2)
                    total_overlap += overlap_fraction

            #dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_s
            overlap_loss = 2*torch.sum(total_overlap)/n_classes/dims[0]
            #print("overlap_loss=", overlap_loss)

        if (self.OVERLAP_PENALTY):
            dice_loss_total = 1.0 - torch.mean(dice) + 0.5*overlap_loss #sum_over_batches
        else:
            dice_loss_total = 1.0 - torch.mean(dice)

        return dice_loss_total.cuda()/input.shape[0]



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


#----------------------------------------------------------------------------------------------------------
def get_data_loaders(path, originalXY, originalZ,  batch_size, usetasks, cliplow, cliphigh, augmentversions=0, n_classes=1,
                    CVfold=0, num_train = None, deformfactor=3, nonrigid=True, testsplits=5, config=None, validationfolds=3,
                    augment3d=True):

    from model.DatasetDecathlon import DecathlonData

    print("making train data set generator\n")
    train_dataset = DecathlonData(path , originalXY, originalZ, usetasks, cliplow, cliphigh, n_augmented_versions=augmentversions, n_classes=n_classes,
                                mode='CVtrain', CVfold=CVfold, testsplits=testsplits, deformfactor=deformfactor,
                                    augment3d=True, nonrigid=nonrigid, num_train=num_train, config = config, validationfolds=validationfolds)

    print("making test data set generator\n")
    test_dataset = DecathlonData(path, originalXY, originalZ, usetasks, cliplow, cliphigh, n_augmented_versions=0, mode='CVtest', n_classes=n_classes,
                                CVfold=CVfold, testsplits=testsplits, num_train=0,config = config, validationfolds=validationfolds)

    print("making val data set generator\n")
    val_dataset = DecathlonData(path, originalXY, originalZ, usetasks, cliplow, cliphigh, n_augmented_versions=0, mode='CVval', n_classes=n_classes,
                                CVfold=CVfold, testsplits=testsplits,num_train=num_train, config = config, validationfolds=validationfolds)

    num_workers = multiprocessing.cpu_count()

    if num_workers >= 32:
        num_workers = 1  #for some reason paralell loader is slower than non-parlallel!! best to go with 1 here most likely!

    train_sampler = WeightedRandomSampler(train_dataset.weights(), num_samples=len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)#useCPUs)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,  num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,  num_workers=1)

    print('=====> #train, #val, #test = ', len(train_dataset),  len(val_dataset), len(test_dataset))
    #print(train_dataset.allfiles, train_dataset.files)
    return train_loader, validation_loader, test_loader, list(set(train_dataset.tasks))

#-------------------------------------------------------------------------------
def validate(model, validation_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_layer = nn.Sigmoid()

    with torch.no_grad():
        model.train(False)

        TP = 0
        FP = 0
        FN = 0

        for i, data in enumerate(validation_loader):
            inputs = data['data']
            labels = data['label']

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = final_layer(outputs)
            outputs = outputs.view(-1)

            predicted_label = int(np.round(outputs.detach().cpu().numpy()))
            labels = int(labels.detach().cpu().numpy())

            if ((predicted_label == 1) and (labels == 1)):
                TP += 1
            elif ((predicted_label == 0) and (labels == 1)):
                FN += 1
            elif ((predicted_label == 1) and (labels == 0)):
                FP += 1

    model.train(True)

    precision = TP/(TP+FP+.00001)
    recall = TP/(TP+FN+.00001)
    print("prec: ", precision, "recall:", recall)
    F1 = 2*precision*recall/(precision + recall + .00001)

    return F1


#-------------------------------------------------------------------------------
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
