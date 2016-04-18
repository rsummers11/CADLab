require('mobdebug').start()
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'cunn'
require 'cudnn'

require 'util.OneHot'
require 'util.misc'

require 'image'
dofile 'util/improvider.lua'
require '../imcnn/Csv.lua'
require 'util.SpatialContrastiveNormalization__'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
-- cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-model','cv/iter0/lm_gru_nin_128_epoch23.39_3.0613.t7','model checkpoint to use for sampling')
cmd:option('-imfiles','../src/chestx/data/iter0_im_te.csv','text file with input images and labels')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',5,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-immodel','nin','name of the imcnn model')
cmd:option('-immodelloc','../imcnn/logs/nin_default_lr0.1_bnonly_ddropout_nodropout/model.net','path to the imcnn model')
cmd:option('-pcaloc','../src/chestx/data/iter0_imcaps_trval/pca_nin.t7','location of the pca file')
cmd:option('-data_loc','workspace/learning_to_read/data/chestx/ims','location of the image data')
cmd:option('-labels_file','../src/chestx/data/iter0_label_names.csv','text file with label names')
cmd:text()

-- parse input params
opt = cmd:parse(arg)


local imsize = 256
local crsize = 224
local nchannels = 3
local batchsize = 10


-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- prepare image cnn model
local immodel = torch.load(opt.immodelloc)
if opt.immodel:sub(1,3)=='nin' then
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):remove(#immodel:get(2).modules)
    immodel:get(2):add(cudnn.SpatialAveragePooling(6,6)):cuda()
elseif opt.immodel:sub(1,3)=='vgg' then
    immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
    immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
    if #opt.immodel<10 then
        immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
    end
elseif opt.immodel:sub(1,3)=='goo' then
    immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
    immodel:get(2):get(2):remove(#immodel:get(2):get(2).modules)
else
    print('--immodel option unrecognized!')
    return
end
immodelraw = torch.load(opt.immodelloc)


function load_im_batches(imfiles, doaugs, dataloc, batchsize, imagesize, cropsize)
    local provider = Provider(imfiles, doaugs, dataloc, batchsize, imagesize, cropsize)
    provider:normalize()
    local ims = provider.trainData.data
    return ims
end


function get_init_state(imbatch)
  
    local outputs = immodel:forward(imbatch)
    outputs = outputs:view(batchsize,-1)
    local outputsd = outputs:double()

    local imbatch_proj = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if outputs:size(2) == checkpoint.opt.rnn_size then
        imbatch_proj = outputsd
    else
        v = torch.load(opt.pcaloc)
        imbatch_proj = outputsd*(v:narrow(2,1,checkpoint.opt.rnn_size))
    end

    local current_state = {}
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        -- local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
        local h_init = imbatch_proj
        if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(current_state, h_init:clone())
        if checkpoint.opt.model == 'lstm' then
            table.insert(current_state, h_init:clone())
        end
    end

    return current_state
end


-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end


-- prepare to load test data
local separator = ' '
local csv = Csv(opt.imfiles, 'r', separator)
local data = csv:readall()
local csvl = Csv(opt.labels_file, 'r', separator)
local data_labels = csvl:readall()


wfile = io.open(opt.model..'_pre.txt', 'w')
io.output(wfile)


for iternum=1,10 do


    dataidxs = torch.randperm(#data)
    for batchi=1,math.floor(#data/batchsize) do
        local btIdx1 = (batchi-1)*batchsize + 1
        local dataidxsi = dataidxs:narrow(1,btIdx1,batchsize)
        local fnames = {}
        local doaugs = {}
        local labels = {}
        for batchi2=1,batchsize do
            fnames[batchi2] = data[dataidxsi[batchi2]][1]
            doaugs[batchi2] = tonumber(data[dataidxsi[batchi2]][2])
            labels[batchi2] = tonumber(data[dataidxsi[batchi2]][3])
        end


        -- initialize the rnn state to all zeros
        -- gprint('creating an ' .. checkpoint.opt.model .. '...')
        local current_state_bs
        local imbatch = load_im_batches(fnames, doaugs, opt.data_loc, batchsize, imsize, crsize)
        current_state_bs = get_init_state(imbatch)
        state_size = #current_state_bs

        -- image predictions
        impredicts = immodelraw:forward(imbatch)


        for batchj=1,batchsize do
          
            io.write(fnames[batchj]..'|')

            current_state_bj = {}
            for csi=1,state_size do
                table.insert(current_state_bj, current_state_bs[csi][batchj]:resize(1,checkpoint.opt.rnn_size))
            end

            -- now, predicted label text is the seed text
            local maxi, labeli = torch.max(impredicts[batchj],1)
            local labelstr = string.gsub(data_labels[labeli[1]][2], "%s+", "")
            local seed_text = labelstr

            -- do a few seeded timesteps
            -- local seed_text = opt.primetext
            if string.len(seed_text) > 0 then
                -- gprint('seeding with ' .. seed_text)
                -- gprint('--------------------------')
                for c in seed_text:gmatch'[^%s]+' do--'.' do
                    if not vocab[c] then
                        -- fill with uniform probabilities over characters (? hmm)
                        -- gprint('missing seed text, using uniform probability over first character')
                        -- gprint('--------------------------')
                        prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
                        if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
                        if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
                    else
                        prev_char = torch.Tensor{vocab[c]}
                        io.write(ivocab[prev_char[1]]..' ')
                        if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
                        if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
                        local lst = protos.rnn:forward{prev_char, unpack(current_state_bj)}
                        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
                        current_state = {}
                        for i=1,state_size do table.insert(current_state, lst[i]) end
                        prediction = lst[#lst] -- last element holds the log probabilities
                    end
                end
            else
                -- fill with uniform probabilities over characters (? hmm)
                gprint('missing seed text, using uniform probability over first character')
                gprint('--------------------------')
                prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
                if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
                if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
            end

            -- start sampling/argmaxing
            for i=1, opt.length do

                -- log probabilities from the previous timestep
                if opt.sample == 0 then
                    -- use argmax
                    local _, prev_char_ = prediction:max(2)
                    prev_char = prev_char_:resize(1)
                else
                    -- use sampling
                    prediction:div(opt.temperature) -- scale by temperature
                    local probs = torch.exp(prediction):squeeze()
                    probs:div(torch.sum(probs)) -- renormalize so probs sum to one
                    prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
                end

                -- forward the rnn for next character
                local lst = protos.rnn:forward{prev_char, unpack(current_state_bj)}
                current_state = {}
                for i=1,state_size do table.insert(current_state, lst[i]) end
                prediction = lst[#lst] -- last element holds the log probabilities

                io.write(ivocab[prev_char[1]]..' ')
            end
            io.write('\n') io.flush()

            --accumulate
            --torch.mean(accumulated,1)
            --m4 = nn.HardTanh():cuda()
            --((m4:forward(lst[4]*1000)+1)/2):int()


        end

    end

end

io.close(wfile)
