require('mobdebug').start()
require 'nn'
require 'image'
require 'xlua'
require 'Csv'

local Provider = torch.class 'Provider'


function Provider:__init(trFilesTxt, teFilesTxt, dataLoc, trBatchSize, teBatchSize, batchNum, nEpoch, imageSize, cropSize, doAug, maxLabel, max2MaxRatio)

  torch.manualSeed(nEpoch)


  local trsize = trBatchSize
  local tesize = teBatchSize

  local garbageBatchSize = 10

  if trBatchSize<0 then trsize = garbageBatchSize end
  if teBatchSize<0 then tesize = garbageBatchSize end


  local imsize = imageSize
  local crsize = cropSize
  local nchannels = 3


  -- load dataset

  self.trainData = {
     data = torch.Tensor(trsize, crsize*crsize*nchannels),
     labels = torch.Tensor(trsize),
     files = {},
     size = function() return trsize end
  }
  local trainData = self.trainData
  local nidxTr = 1

  if trBatchSize>0 then

    local separator = ' '
    local csv = Csv(trFilesTxt, 'r', separator)
    local idxs = torch.randperm(trsize)
    local idx = 1
    local idx2 = trsize * (batchNum - 1) + 1
    local idx3 = 1
    local maxLabelSkipped = {}
    while true do
      local data = csv:read()
      if idx3 >= idx2 then   
        if not data then break end
        if nidxTr >= trsize then break end
        im00 = image.load(os.getenv('HOME')..'/'..dataLoc..'/'..data[1])
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
        lbaug = tonumber(data[3])
        if doAug=='false' or lbaug<1 then
          im2 = image.crop(im1, torch.round((imsize-crsize)/2), torch.round((imsize-crsize)/2),
            torch.round((imsize-crsize)/2)+crsize, torch.round((imsize-crsize)/2)+crsize)

          im = image.scale(im2, crsize, crsize):reshape(crsize*crsize*nchannels)
          lb = tonumber(data[2])

          if lb==maxLabel and torch.uniform()>max2MaxRatio then
            table.insert(maxLabelSkipped, {im, lb, data[1]})
          else            
            trainData.data[idxs[idx]] = im
            trainData.labels[idxs[idx]] = lb
            trainData.files[idxs[idx]] = data[1]
            idx = idx + 1
            nidxTr = nidxTr + 1
          end
        else
          d1x = torch.randperm(16)[1]
          d1y = torch.randperm(16)[1]
          im2 = image.crop(im1, torch.round(d1x/2), torch.round(d1y/2), 
            torch.round(d1x/2)+crsize, torch.round(d1y/2)+crsize)

          im = image.scale(im2, crsize, crsize):reshape(crsize*crsize*nchannels)
          lb = tonumber(data[2])
          
          trainData.data[idxs[idx]] = im
          trainData.labels[idxs[idx]] = lb
          trainData.files[idxs[idx]] = data[1]
          idx = idx + 1
          nidxTr = nidxTr + 1
        end

      else
        -- print(idx, idx2, idx3)
      end
      idx3 = idx3 + 1
    end
    if maxLabel>=0 and idx<=trsize and table.getn(maxLabelSkipped)>0 then
      local idxNew01 = torch.randperm(table.getn(maxLabelSkipped))
      for idxNew0=1,table.getn(maxLabelSkipped) do
        if idx>trsize then break end
        trainData.data[idxs[idx]] = maxLabelSkipped[idxNew01[idxNew0]][1]
        trainData.labels[idxs[idx]] = maxLabelSkipped[idxNew01[idxNew0]][2]
        trainData.files[idxs[idx]] = maxLabelSkipped[idxNew01[idxNew0]][3]
        idx = idx + 1
      end
    end
    if idx<=trsize then
      for idxNew=idx,trsize do
        trainData.data[idxs[idxNew]] = trainData.data[idxs[idxNew-idx+1]]
        trainData.labels[idxs[idxNew]] = trainData.labels[idxs[idxNew-idx+1]]
        trainData.files[idxs[idxNew]] = trainData.files[idxs[idxNew-idx+1]]
      end
      nidxTr = trsize
    end
    trainData.labels = trainData.labels + 1
  else
    --
  end


  self.testData = {
     data = torch.Tensor(tesize, crsize*crsize*nchannels),
     labels = torch.Tensor(tesize),
     files = {},
     size = function() return tesize end
  }
  local testData = self.testData
  local nidxTe = 1

  if teBatchSize>0 then

    separator = ' '
    csv = Csv(teFilesTxt, 'r', separator)
    idxs = torch.randperm(tesize)
    idx = 1 
    idx2 = tesize * (batchNum - 1) + 1
    idx3 = 1
    while true do
      local data = csv:read()
      if idx3 >= idx2 then
        if not data then break end
        if nidxTe >= tesize then break end
        im00 = image.load(os.getenv('HOME')..'/'..dataLoc..'/'..data[1])
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

        if crsize<imsize and crsize>0 then
          im2 = image.crop(im1, torch.round((imsize-crsize)/2), torch.round((imsize-crsize)/2),
            torch.round((imsize-crsize)/2)+crsize, torch.round((imsize-crsize)/2)+crsize)
        else
          im2 = im1
          crsize = imsize
        end

        im = image.scale(im2, crsize, crsize):reshape(crsize*crsize*nchannels)
        lb = tonumber(data[2])
        testData.data[idxs[idx]] = im
        testData.labels[idxs[idx]] = lb
        testData.files[idxs[idx]] = data[1]
        idx = idx + 1
        nidxTe = nidxTe + 1
      else
        -- print(idx, idx2, idx3)
      end
      idx3 = idx3 + 1
    end
    if idx<=tesize then
      for idxNew=idx,tesize do
        testData.data[idxs[idxNew]] = testData.data[idxs[idxNew-idx+1]]
        testData.labels[idxs[idxNew]] = testData.labels[idxs[idxNew-idx+1]]
        testData.files[idxs[idxNew]] = testData.files[idxs[idxNew-idx+1]]
      end
    end
    testData.labels = testData.labels + 1
  else
    --
  end


  if nidxTr and trsize>nidxTr then trsize = nidxTr end
  if nidxTe and tesize>nidxTe then tesize = nidxTe end


  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  trainData.data = trainData.data:reshape(trsize,nchannels,crsize,crsize)
  testData.data = testData.data:reshape(tesize,nchannels,crsize,crsize)
end


function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  -- print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     -- xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
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

  -- preprocess testSet
  for i = 1,testData:size() do
    -- xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
