
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local CharSplitLMMinibatchLoader = {}
CharSplitLMMinibatchLoader.__index = CharSplitLMMinibatchLoader

function CharSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, nEpoch, iterNum)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, CharSplitLMMinibatchLoader)

    local input_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'input.txt')
    local vocab_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'vocab.t7')
    local tensor_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'data.t7')
    local fnames_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'fnames.t7')
    local labels_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'labels.t7')
    local doaugs_file = path.join(data_dir, 'iter'..tostring(iterNum)..'_imcaps_trval', 'epoch'..tostring(math.floor(nEpoch)), 'doaugs.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        CharSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, fnames_file, labels_file, doaugs_file)
    end

    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)
    local fnamesl = torch.load(fnames_file)
    local labelsl = torch.load(labels_file)
    local doaugsl = torch.load(doaugs_file)
    local fnames = {}
    local labels = {}
    local doaugs = {}

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
        numFnames = data:size(1)/5
        for fnamesi=1,numFnames do 
            fnames[fnamesi] = fnamesl[fnamesi]
            labels[fnamesi] = labelsl[fnamesi]
            doaugs[fnamesi] = doaugsl[fnamesi]
        end
    else
        fnames = fnamesl
        labels = labelsl
        doaugs = doaugsl
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    self.fname_batches = {}
    self.label_batches = {}
    self.doaug_batches = {}
    self.all_fnames = {}
    self.all_doaugs = {}
    local fnamesi2 = 1
    local fnamesi3 = 1
    local fname_batch = {}
    local label_batch = {}
    local doaug_batch = {}
    for fnamesi=1,numFnames do
        if (fnamesi%50==1 and fnamesi>1) or fnamesi==numFnames then
            self.fname_batches[fnamesi3] = fname_batch
            self.label_batches[fnamesi3] = label_batch
            self.doaug_batches[fnamesi3] = doaug_batch
            fnamesi3 = fnamesi3 + 1
            
            fnamesi2 = 1
            fname_batch[fnamesi2] = fnames[fnamesi]
            label_batch[fnamesi2] = labels[fnamesi]
            doaug_batch[fnamesi2] = doaugs[fnamesi]
            fnamesi2 = fnamesi2 + 1

            fname_batch = {}
            label_batch = {}
            doaug_batch = {}
        else
            fname_batch[fnamesi2] = fnames[fnamesi]
            label_batch[fnamesi2] = labels[fnamesi]
            doaug_batch[fnamesi2] = doaugs[fnamesi]
            fnamesi2 = fnamesi2 + 1
        end
        self.all_fnames[fnamesi] = fnames[fnamesi]
        self.all_doaugs[fnamesi] = doaugs[fnamesi]
    end

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function CharSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function CharSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix], self.fname_batches[ix], self.label_batches[ix], self.doaug_batches[ix], 
        self.all_fnames, self.all_doaugs
end

-- *** STATIC method ***
function CharSplitLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, out_fnames_file, out_labels_file, out_doaugs_file)
    local timer = torch.Timer()

    print('loading text file...')
    local cache_len = 10000
    local rawdata
    local tot_len = 0
    f = io.open(in_textfile, "r")

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all characters to a set
    local unordered = {}
    for line in f:lines() do
        for word in line:gmatch'[^%s]+' do
            if string.sub(word, string.len(word)-3, string.len(word)) == '.png' then
                --
            elseif tonumber(word) then
                --
            else
                if not unordered[word] then unordered[word] = true end
                tot_len = tot_len + 1
            end
        end
    end
    f:close()
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
    local imfiles = {}
    local imlabels = {}
    local imdoaugs = {}
    f = io.open(in_textfile, "r")
    local imfilesi = 1
    local datai = 1
    local justReadFname = false
    for line in f:lines() do
        justReadFname = false
        for word in line:gmatch'[^%s]+' do
            if string.sub(word, string.len(word)-3, string.len(word)) == '.png' then
                imfiles[imfilesi] = word                
                justReadFname = true
            elseif tonumber(word) then
                if justReadFname then
                    imlabels[imfilesi] = tonumber(word)
                    justReadFname = false
                else
                    imdoaugs[imfilesi] = tonumber(word)
                    imfilesi = imfilesi + 1
                end
            else
                data[datai] = vocab_mapping[word]
                datai = datai+1
            end
        end
    end
    f:close()

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
    print('saving ' .. out_fnames_file)
    torch.save(out_fnames_file, imfiles)
    print('saving ' .. out_labels_file)
    torch.save(out_labels_file, imlabels)
    print('saving ' .. out_doaugs_file)
    torch.save(out_doaugs_file, imdoaugs)
end

return CharSplitLMMinibatchLoader

