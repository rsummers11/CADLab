import torch.nn as nn
import torch
import torch.nn.functional as F
#from profilehooks import profile
from model.utils import *
import numpy as np


class convolution_block(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding = (1,1,1)):
        super(convolution_block, self).__init__()
        self.convolution1 = nn.Conv3d(in_filters, out_filters, kernel_size = kernel_size, stride = stride, padding= padding)
        if out_filters % 16 == 0:
            self.normalization1 = nn.GroupNorm(16, out_filters)
        elif out_filters % 12 == 0:
            self.normalization1 = nn.GroupNorm(12, out_filters)
        else:
            self.normalization1 = nn.GroupNorm(out_filters, out_filters)

        self.activation1 = nn.LeakyReLU()

    def forward(self,x):
        out = self.convolution1(x)
        out = self.normalization1(out)
        out = self.activation1(out)
        return out

class localization_module(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super(localization_module, self).__init__()
        self.convolution1 = convolution_block(in_filters, out_filters)
        self.convolution2 = convolution_block(out_filters, out_filters, kernel_size=(1,1,1) , padding =0) #

    def forward(self,x):
        out = self.convolution1(x)
        out = self.convolution2(out)
        return out

class up_sampling_module(torch.nn.Module):
    def __init__(self,in_filters, out_filters, size=(2, 2, 2)):
        super(up_sampling_module, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.convolution1 = convolution_block(in_filters, out_filters, kernel_size=(1,1,1), padding =0)
    def forward(self,x):
        out = self.upsample(x)
        out = self.convolution1(out)
        return out

class context_module(torch.nn.Module):
    def __init__(self,in_filters, out_filters, dropout_rate=0.3):
        super(context_module, self).__init__()
        self.convolution1 = convolution_block(in_filters, out_filters)
        self.dropout =  nn.Dropout3d(p=dropout_rate)
        self.convolution2 = convolution_block(in_filters, out_filters)
    def forward(self,x):
        out = self.convolution1(x)
        out = self.dropout(out)
        out = self.convolution2(out)
        return out


class PyramidPool(nn.Module):
    def __init__(self, in_features, out_features, pool_factor):
        super(PyramidPool,self).__init__()
        self.net = nn.Sequential(
            nn.AvgPool3d(pool_factor, stride=pool_factor, padding=0),
            nn.Conv3d(in_features, out_features, 1, bias=False),
            nn.GroupNorm(out_features, out_features),
            nn.LeakyReLU(), nn.Dropout3d(p=0.3),
            nn.Upsample(scale_factor= pool_factor,  mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        return self.net(x)


class MultiDataModel(nn.Module):
    def __init__(self, in_channels, n_classes, n_filter_per_level =( 4,16,64,128,256), dropout_rate =0.3):
        super(MultiDataModel, self).__init__()

        self.material = False
        self.n_classes= n_classes
        self.in_channels = in_channels
        self.n_filter_per_level = n_filter_per_level
        self.dropout_rate = dropout_rate
        self.base_n_filter = 8

        self.initial_downconvolution =nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size = (7,7,7), stride = (2,2,2) , padding= (3,3,3))

        self.convolution_block1 = convolution_block(self.base_n_filter, n_filter_per_level[0])
        self.context_module1 = context_module(n_filter_per_level[0], n_filter_per_level[0], dropout_rate=self.dropout_rate)

        self.convolution_block2 = convolution_block(n_filter_per_level[0],n_filter_per_level[1], stride=(2, 2, 2))
        self.context_module2 = context_module(n_filter_per_level[1],n_filter_per_level[1], dropout_rate=self.dropout_rate)

        self.convolution_block3 = convolution_block(n_filter_per_level[1],n_filter_per_level[2], stride=(2, 2, 2))
        self.context_module3 = context_module(n_filter_per_level[2],n_filter_per_level[2], dropout_rate=self.dropout_rate)

        self.convolution_block4 = convolution_block(n_filter_per_level[2],n_filter_per_level[3], stride=(2, 2, 2))
        self.context_module4 = context_module(n_filter_per_level[3],n_filter_per_level[3], dropout_rate=self.dropout_rate)

        self.convolution_block5 = convolution_block(n_filter_per_level[3],n_filter_per_level[4], stride=(2, 2, 2))
        self.context_module5 = context_module(n_filter_per_level[4],n_filter_per_level[4], dropout_rate=self.dropout_rate)

        self.up_sampling_module4 = up_sampling_module(n_filter_per_level[4],n_filter_per_level[3])
        self.localization_module4 = localization_module( n_filter_per_level[4], n_filter_per_level[3])

        self.up_sampling_module3 = up_sampling_module(n_filter_per_level[3],n_filter_per_level[2])
        self.localization_module3 = localization_module(n_filter_per_level[2]*2, n_filter_per_level[2])

        self.up_sampling_module2 = up_sampling_module(n_filter_per_level[2],n_filter_per_level[1])
        self.localization_module2 = localization_module(n_filter_per_level[1]*2, n_filter_per_level[1])

        self.up_sampling_module1 = up_sampling_module(n_filter_per_level[1], n_filter_per_level[0])
        self.localization_module1 = localization_module(n_filter_per_level[0]*2,  n_filter_per_level[0])


        self.Up4 = nn.Upsample(scale_factor= 4,  mode='trilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor= 2,  mode='trilinear', align_corners=False)

        #self.final_layers = nn.ModuleList([nn.ConvTranspose3d(n_filter_per_level[0]*3,self.n_classes,  kernel_size = (7,7,7), stride=(2, 2, 1),  padding=(3, 3, 3), output_padding = (1,1,0)) for i in range(11)])
        #self.final_layers = nn.ModuleList([nn.Conv3d(n_filter_per_level[0],self.n_classes,  kernel_size = (3,3,3), stride=(1,1, 1), padding = (1,1,1)) for i in range(13)])
        self.final_layers = nn.ModuleList([nn.ConvTranspose3d(n_filter_per_level[0],self.n_classes,  kernel_size = (7,7,7), stride=(2, 2,2),  padding=(3, 3, 3), output_padding = (1,1,1)) for i in range(17)])

        self.final_activation = nn.Sigmoid()

        #self.PSPa = PyramidPool(n_filter_per_level[0], self.PSP_features, 2)
        #self.PSPb = PyramidPool(n_filter_per_level[0], self.PSP_features, 4)
        #self.PSPc = PyramidPool(n_filter_per_level[0], self.PSP_features, 8)
        #self.PSPd = PyramidPool(n_filter_per_level[0], self.PSP_features, 16)
        if self.material:
            self.final_layer_mat = nn.ConvTranspose3d(n_filter_per_level[0],6, kernel_size = (7,7,7), stride=(2, 2,2),  padding=(3, 3, 3), output_padding = (1,1,1))

        self.dropout =  nn.Dropout3d(p=0.3)


    def forward(self, x, task, alltasks):
        #print('task',task.shape)
        #tasfor layer in range()
        task = task.cpu().numpy()[0]
        #print('task number', task)
        out = self.initial_downconvolution(x)
        #import pdb; pdb.set_trace()
        out = self.convolution_block1(out)
        out_context = self.context_module1(out)
        out = out + out_context
        level1 = out


        out = self.convolution_block2(out)
        out_context = self.context_module2(out)
        out = out + out_context
        level2 = out

        out = self.convolution_block3(out)
        out_context = self.context_module3(out)
        out = out + out_context
        level3 = out


        out =self.convolution_block4(out)
        out_context = self.context_module4(out)
        out = out + out_context
        level4 = out

        out = self.convolution_block5(out)
        out_context = self.context_module5(out)
        out = out + out_context



        #up_sampling4
        out = self.up_sampling_module4(out) # 16
        out = torch.cat( (out,level4), dim =1) #
        out = self.localization_module4(out)

        #up_sampling3
        out = self.up_sampling_module3(out)
        out = torch.cat( (out,level3), dim =1)
        out = self.localization_module3(out)
        #segmentation3 = self.seg3(out)

        #up_sampling2
        out = self.up_sampling_module2(out)
        out = torch.cat( (out,level2), dim =1)
        out = self.localization_module2(out)
        #segmentation2 = self.seg2(out)

        #up_sampling1
        out = self.up_sampling_module1(out)
        out = torch.cat((out,level1), dim =1)
        #dense1 = out
        out = self.localization_module1(out)
        #out = torch.cat((out,dense1), dim =1)



        #PSP
        #out = torch.cat([
        #    self.dropout(out),
        #    self.PSPa(out)*0,
        #    self.PSPb(out)*0,
        #    self.PSPc(out)*0,
        #    self.PSPd(out)*0], dim=1)
        # generate multitask outptut
        #print(out.shape)
        with torch.no_grad():
            combined = x
            for i in alltasks:
                #print(i)
                #final_layer = self.final_layers[i]
                #final = final_layer(out)
                final = self.final_layers[i](out)
                final = self.final_activation(final)
                combined = torch.cat( (combined, final ), dim = 1)

        # select output
        output = dict()
        if self.material:
            mat = self.final_layer_mat(out)
            output['matmap'] = mat

        out = self.final_layers[task](out)
        seg_layer = self.final_activation(out)

        output['final_layerA'] = seg_layer
        output['complete'] = combined



        return output
