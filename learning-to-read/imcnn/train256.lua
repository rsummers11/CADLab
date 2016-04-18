require('mobdebug').start()
require 'xlua'
require 'optim'
require 'cunn'
require 'loadcaffe'
dofile './provider256.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs/nin")      subdirectory to save logs
   -b,--batchSize             (default 50)          batch size
   -r,--learningRate          (default 0.01)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 33)          epoch step
   --model                    (default nin)     model name
   --max_epoch                (default 100)           maximum number of iterations
   --data_loc                 (default "workspace/dataset/torch")            image location
   --trainData                (default "workspace/dataset/torch/train.txt")  train.txt location
   --testData                 (default "workspace/dataset/torch/val.txt")    val.txt location
   --image_size               (default 256)         image size
   --crop_size                (default 224)         image size after crop
   --use_pretrained           (default false)   whether to use pre-trained model
   --num_labels               (default 1000)    number of labels
   --do_aug                   (default false)   do augmentation or not
   --max_label                (default -1)      label with maximum occurences to subsample
   --max_2max_ratio           (default -1)      ratio of maximum label to subsample and 2nd maximum label
   --pretrained_model_loc     (default none)    location of the pre-trained model
]]

print(opt)


function read_num_lines(txtFile)
  local cache_len = 10000
  local totCount = 0
  local f = io.open(txtFile, 'rb')
  local content = f:read(cache_len)
  repeat
    local _, count = content:gsub('\n', '\n')
    totCount = totCount + count
    content = f:read(cache_len)
  until not content
  f:close()
  return totCount
end
local trFilecount = read_num_lines(os.getenv('HOME')..'/'..opt.trainData)
local teFilecount = read_num_lines(os.getenv('HOME')..'/'..opt.testData)


do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output = input
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
-- model:add(nn.BatchFlip():float())
if opt.use_pretrained=='true' then
  if opt.pretrained_model_loc=='none' then
    model:add(loadcaffe.load('models/' .. opt.model .. '/deploy.prototxt', 'models/' .. opt.model .. 
      '/' .. opt.model .. '_imagenet.caffemodel', 'cudnn'))
    if opt.model=='nin' then
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):add(cudnn.SpatialConvolution(1024, opt.num_labels, 1, 1, 1, 1, 0, 0, 1)):cuda()
      model:get(2):add(cudnn.ReLU(true)):cuda()
      model:get(2):add(cudnn.SpatialAveragePooling(6, 6, 1, 1, 0, 0):ceil()):cuda()
      model:get(2):add(nn.View(opt.num_labels)):cuda()
    elseif opt.model=='vgg' then
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):add(nn.Linear(4096, opt.num_labels)):cuda()
    else
      print('--model option unrecognized!')
      return
    end
  else
    local model_pre = torch.load(opt.pretrained_model_loc)
    model:add(model_pre:get(2))
    if opt.model:sub(1,3)=='nin' then
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):remove(#model:get(2).modules)
      model:get(2):add(cudnn.SpatialConvolution(1024, opt.num_labels, 1, 1)):cuda()
      model:get(2):add(nn.SpatialBatchNormalization(opt.num_labels, 1e-3)):cuda()
      model:get(2):add(cudnn.ReLU(true)):cuda()
      model:get(2):add(cudnn.SpatialAveragePooling(6,6)):cuda()
      model:get(2):add(nn.View(opt.num_labels)):cuda()
    elseif opt.model:sub(1,3)=='vgg' then
      model:get(2):get(2):remove(#model:get(2):get(2).modules)
      model:get(2):get(2):add(nn.Linear(4096, opt.num_labels)):cuda()
    elseif opt.model:sub(1,3)=='goo' then
      model:get(2):get(2):remove(#model:get(2):get(2).modules)
      model:get(2):get(2):remove(#model:get(2):get(2).modules)
      model:get(2):get(2):add(nn.Linear(1024, opt.num_labels)):cuda()
      main_branch:add(nn.LogSoftMax())
    else
      --
    end
  end
elseif opt.use_pretrained=='false' then
  model:add(dofile('models/'..opt.model..'.lua'):cuda())
  if opt.model:sub(1,3)=='nin' then
    model:get(2):remove(#model:get(2).modules)
    model:get(2):remove(#model:get(2).modules)
    model:get(2):remove(#model:get(2).modules)
    model:get(2):remove(#model:get(2).modules)
    model:get(2):remove(#model:get(2).modules)
    model:get(2):add(cudnn.SpatialConvolution(1024, opt.num_labels, 1, 1)):cuda()
    model:get(2):add(nn.SpatialBatchNormalization(opt.num_labels, 1e-3)):cuda()
    model:get(2):add(cudnn.ReLU(true)):cuda()
    model:get(2):add(cudnn.SpatialAveragePooling(6,6)):cuda()
    model:get(2):add(nn.View(opt.num_labels)):cuda()
  elseif opt.model:sub(1,3)=='vgg' then
    model:get(2):get(2):remove(#model:get(2):get(2).modules)
    model:get(2):get(2):add(nn.Linear(4096, opt.num_labels)):cuda()
  elseif opt.model:sub(1,3)=='goo' then
      model:get(2):get(2):remove(#model:get(2):get(2).modules)
      model:get(2):get(2):remove(#model:get(2):get(2).modules)
      model:get(2):get(2):add(nn.Linear(1024, opt.num_labels)):cuda()
      main_branch:add(nn.LogSoftMax())
  else
    --
  end
else
  print('--use_pretrained option unrecognized!')
  return
end
model:get(2).updateGradInput = function(input) return end
print(model)

confusion = optim.ConfusionMatrix(opt.num_labels)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


if opt.use_pretrained=='false' then
  print(c.blue'==>' ..' configuring optimizer')
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
  }
elseif opt.use_pretrained=='true' then
  params_lr_m = model:get(2):clone()
  if opt.model=='nin' then
    params_lr_m = model:clone()
    params_lr = params_lr_m:getParameters()
    params_lr:fill(1);
    params_lr_m:get(2):get(#model:get(2).modules-3).weight:fill(10);
    params_lr_m:get(2):get(#model:get(2).modules-3).bias:fill(20);
  elseif opt.model=='vgg' then
    params_lr_m = model:clone()
    params_lr = params_lr_m:getParameters()
    params_lr:fill(1);
    params_lr_m:get(2):get(#model:get(2).modules).weight:fill(10);
    params_lr_m:get(2):get(#model:get(2).modules).bias:fill(20);
  elseif opt.model=='goo' then
    params_lr_m = model:clone()
    params_lr = params_lr_m:getParameters()
    params_lr:fill(1);
    params_lr_m:get(2):get(2):get(#model:get(2):get(2).modules-1).weight:fill(10);
    params_lr_m:get(2):get(2):get(#model:get(2):get(2).modules-1).bias:fill(20);
  else
    print('--model option unrecognized!')
    return
  end
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
    learningRates = params_lr,
  }
else
  print('--model option unrecognized!')
  return
end


function rProvider(trFilesTxt, teFilesTxt, dataLoc, trBatchSize, teBatchSize, t, epoch, imageSize, cropSize, doAug, maxLabel, max2MaxRatio)
  local provider = Provider(trFilesTxt, teFilesTxt, dataLoc, trBatchSize, teBatchSize, t, epoch, 
    imageSize, cropSize, doAug, maxLabel, max2MaxRatio)
  provider:normalize()
  return provider
end


function train(trFilesTxt, teFilesTxt)
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/3 end--2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local inputs = torch.FloatTensor(opt.batchSize,3,opt.crop_size,opt.crop_size)
  local targets = torch.CudaTensor(opt.batchSize)

  local numIndicesM1 = math.floor(trFilecount/opt.batchSize)
  local remainder = math.fmod(trFilecount, opt.batchSize)
  local indices = {}
  for indi=1,numIndicesM1 do
    indices[indi] = torch.range(1,opt.batchSize):long()
  end

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local trCpSize = opt.batchSize
    if math.fmod(t,opt.batchSize)==1 then
      provider = rProvider(trFilesTxt, teFilesTxt, opt.data_loc, opt.batchSize*opt.batchSize, -1, 
        math.floor(t/opt.batchSize)+1, epoch, opt.image_size, opt.crop_size, opt.do_aug, 
        opt.max_label, opt.max_2max_ratio)
      indicesU = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
      --
      inputs:copy(provider.trainData.data:index(1,indicesU[1]))
      targets:copy(provider.trainData.labels:index(1,indicesU[1]))
    elseif math.fmod(t,opt.batchSize)==0 then
      inputs:copy(provider.trainData.data:index(1,indicesU[opt.batchSize]))
      targets:copy(provider.trainData.labels:index(1,indicesU[opt.batchSize]))
    else
      if t*opt.batchSize+opt.batchSize>trFilecount then 
        trCpSize=trFilecount-t*opt.batchSize
        if trCpSize==0 then trCpSize=opt.batchSize end
        inputs = inputs:narrow(1,1,trCpSize)
        targets = targets:narrow(1,1,trCpSize)
        inputs:copy(provider.trainData.data:index(1,indicesU[math.fmod(t,opt.batchSize)]:narrow(1,1,trCpSize)))
        targets:copy(provider.trainData.labels:index(1,indicesU[math.fmod(t,opt.batchSize)]:narrow(1,1,trCpSize)))
      else
        inputs:copy(provider.trainData.data:index(1,indicesU[math.fmod(t,opt.batchSize)]))
        targets:copy(provider.trainData.labels:index(1,indicesU[math.fmod(t,opt.batchSize)]))
      end
    end
    --
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test(trFilesTxt, teFilesTxt)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 32
  local batchStep = math.ceil(teFilecount/bs)

  local inputs = torch.FloatTensor(bs,3,opt.crop_size,opt.crop_size)
  local targets = torch.CudaTensor(bs)

  for i=1,math.floor(teFilecount/bs) do
    ----
    local teCpSize = opt.batchSize
    if math.fmod(i,bs)==1 then
      local teBatchNum = (math.floor(i/bs)+1)
      if bs*bs*teBatchNum>teFilecount then teBatchNum=1 end
      provider = rProvider(trFilesTxt, teFilesTxt, opt.data_loc, bs*bs, bs*bs, teBatchNum, epoch, 
        opt.image_size, opt.crop_size, opt.do_aug, opt.max_label, opt.max_2max_ratio)
      indicesU = torch.randperm(provider.testData.data:size(1)):long():split(bs)
      --
      inputs:copy(provider.testData.data:index(1,indicesU[1]))
      targets:copy(provider.testData.labels:index(1,indicesU[1]))
    elseif math.fmod(i,bs)==0 then
      inputs:copy(provider.testData.data:index(1,indicesU[bs]))
      targets:copy(provider.testData.labels:index(1,indicesU[bs]))
    else
      if i*bs+bs>teFilecount then
        teCpSize=teFilecount-i*bs
        if teCpSize==0 then teCpSize=bs end
        inputs = inputs:narrow(1,1,teCpSize)
        targets = targets:narrow(1,1,teCpSize)
        inputs:copy(provider.testData.data:index(1,indicesU[math.fmod(i,bs)]:narrow(1,1,teCpSize)))
        targets:copy(provider.testData.labels:index(1,indicesU[math.fmod(i,bs)]:narrow(1,1,teCpSize)))
      else
        inputs:copy(provider.testData.data:index(1,indicesU[math.fmod(i,bs)]))
        targets:copy(provider.testData.labels:index(1,indicesU[math.fmod(i,bs)]))
      end      
    end
    ----
    local outputs = model:forward(inputs)
    confusion:batchAdd(outputs,targets)
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
  end

  confusion:zero()
end



trFilesTxt = os.getenv('HOME')..'/'..opt.trainData
teFilesTxt = os.getenv('HOME')..'/'..opt.testData

for i=1,opt.max_epoch do
  train(trFilesTxt, teFilesTxt)
  test(trFilesTxt, teFilesTxt)
end


