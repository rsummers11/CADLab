-- adapted from https://github.com/szagoruyko/cifar.torch

require('mobdebug').start()
require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

-- function Provider:__init(trFilesTxt, teFilesTxt, dataLoc, trBatchSize, teBatchSize, batchNum, 
--   nEpoch, imageSize, cropSize, doAug, maxLabel, max2MaxRatio)
function Provider:__init(trFiles, doAugs, dataLoc, trBatchSize, imageSize, cropSize)


  local trsize = trBatchSize

  local garbageBatchSize = 10

  if trBatchSize<0 then trsize = garbageBatchSize end


  local imsize = imageSize
  local crsize = cropSize
  local nchannels = 3


  -- load dataset

  self.trainData = {
     data = torch.Tensor(trsize, crsize*crsize*nchannels),
     size = function() return trsize end
  }
  local trainData = self.trainData
  local nidxTr = 1


  for i=1,trsize do
    im00 = image.load(os.getenv('HOME')..'/'..dataLoc..'/'..trFiles[1])
    if im00:size(2)>imsize or im00:size(3)>imsize or im00:size(2)<imsize or im00:size(3)<imsize then
      im01 = image.scale(im00, imsize, imsize)
    else
      im01 = im00
    end
    if im01:size()[1]<3 then
      im1 = torch.Tensor(nchannels, imsize, imsize)
      im1[1] = im01
      im1[2] = im01
      im1[3] = im01
    elseif im01:size()[1]>3 then
      im1 = torch.Tensor(nchannels, imsize, imsize)
      im1[1] = im01[1]
      im1[2] = im01[2]
      im1[3] = im01[3]
    else
      im1 = im01
    end
    doAug = doAugs[i]
    if doAug<1 then
      im2 = image.crop(im1, torch.round((imsize-crsize)/2), torch.round((imsize-crsize)/2),
        torch.round((imsize-crsize)/2)+crsize, torch.round((imsize-crsize)/2)+crsize)

      im = image.scale(im2, crsize, crsize):reshape(crsize*crsize*nchannels)
    else
      d1x = torch.randperm(16)[1]
      d1y = torch.randperm(16)[1]
      im2 = image.crop(im1, torch.round(d1x/2), torch.round(d1y/2), 
        torch.round(d1x/2)+crsize, torch.round(d1y/2)+crsize)

      im = image.scale(im2, crsize, crsize):reshape(crsize*crsize*nchannels)
    end
    trainData.data[i] = im
  end


  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  
  -- reshape data
  trainData.data = trainData.data:reshape(trsize,nchannels,crsize,crsize)
end


function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  -- local testData = self.testData

  -- print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  -- local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  local normalization = SpatialContrastiveNormalization__(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     -- xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization:updateOutput(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v
end
