-- adapted from https://github.com/szagoruyko/cifar.torch

require 'nn'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(backend.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(backend.ReLU(true))
  return model
end

Block(3,96,11,11,4,4)
Block(96,96,1,1)
Block(96,96,1,1)
model:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
-- model:add(nn.Dropout())
Block(96,256,5,5,1,1,2,2)
Block(256,256,1,1)
Block(256,256,1,1)
model:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
-- model:add(nn.Dropout())
Block(256,384,3,3,1,1,1,1)
Block(384,384,1,1)
Block(384,384,1,1)
model:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
-- model:add(nn.Dropout())
Block(384,1024,3,3,1,1,1,1)
Block(1024,1024,1,1)
Block(1024,1000,1,1)
model:add(backend.SpatialAveragePooling(6,6))
model:add(nn.View(1000))

for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  v.weight:normal(0,0.05)
  v.bias:zero()
end

--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

return model
