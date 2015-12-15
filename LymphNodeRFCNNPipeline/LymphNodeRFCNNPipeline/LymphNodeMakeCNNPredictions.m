function [OUT_BATCH_FOLDER, PREDICTIONS_TEXT_FILE] = LymphNodeMakeCNNPredictions(FolderName,opt)

    %% PARAMS
    if ~isfield(opt,'ConvNetSrcFolder')
        opt.ConvNetSrcFolder = '';
    end     
    if ~isfield(opt,'CNNmodel')
        opt.CNNmodel = '';
    end     
    if ~isfield(opt,'NumberBatches')
        opt.NumberBatches = 20;
    end         
    if ~isfield(opt,'UseMultiview')
        opt.UseMultiview = true;
    end        
    if ~isfield(opt,'ImageSize')
        opt.ImageSize = 32;
    end       
    if ~isfield(opt,'ImageSearchString')
        opt.ImageSearchString = '_AxCoSa.png';
    end         
    if ~isfield(opt,'ImageChannels')
        opt.ImageChannels = 3;
    end  
    if ~isfield(opt,'BatchStartIndex')
        opt.BatchStartIndex = 1;
    end     
    
    parseParams;
    
    %% RUN
    curr_dir = pwd;    
    cd(ConvNetSrcFolder); % python scripts need to run in ConvNetSrcFolder
    
    % make batches
    [outpath, outname] = fileparts(FolderName);
    SAVE_PATH = [outpath,filesep,outname,'_CNNpredictions'];
	if exist(SAVE_PATH,'dir')
		warning(' Removing previous results in %s ...', SAVE_PATH);
		rmdir(SAVE_PATH,'s'); 
    end	    
	mkdir(SAVE_PATH);	
	
   % predPath = [SAVE_PATH,filesep,'CNNpreds'];
   % mkdir(predPath);
    
    PREDICTIONS_BATCH_FILE=[SAVE_PATH,filesep,outname,'_predictions'];
    PREDICTIONS_TEXT_FILE=[SAVE_PATH,filesep,outname,'_predictions.txt'];
    
    INPUT_FOLDER=FolderName;
    OUT_BATCH_FOLDER=SAVE_PATH;
    START_IDX=num2str(BatchStartIndex);
    NUMBER_BATCHES=num2str(NumberBatches);
    SEARCH_STR=ImageSearchString;
    IMG_SIZE=num2str(32);
    IMG_CHANNELS=num2str(ImageChannels);

    command = ['python ./lymph-nodes/make_general_batches.py ',INPUT_FOLDER,' ',OUT_BATCH_FOLDER,' ',START_IDX,' ',NUMBER_BATCHES,' ',SEARCH_STR,' ',IMG_SIZE,' ',IMG_CHANNELS];
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end
    
    %% predict multiview
    INPUT_NN = CNNmodel;
    if exist(INPUT_NN,'file') % using checkpoint file
        INPUT_MODEL_DIR = fileparts(fileparts(INPUT_NN));
    else % is already convnet dir
        INPUT_MODEL_DIR = fileparts(INPUT_NN);
    end
    DATA_SET_PATH = OUT_BATCH_FOLDER;
    TEST_RANGE = [START_IDX,'-',NUMBER_BATCHES];
    if UseMultiview
        USE_MULTIVIEW = '1';
    else
        USE_MULTIVIEW = '0';
    end
    
    % RECONFIGURE
    disp('RECONFIGURE')
    command = ['python ./convnet.py -f ',INPUT_NN,' --logreg-name=logprob --multiview-test=',USE_MULTIVIEW,' --train-range=',TEST_RANGE,' --test-range=',TEST_RANGE,' --test-only=1 --data-path=',DATA_SET_PATH,' --save-path=',INPUT_MODEL_DIR,' --test-freq=1'];
    disp(command);
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end

    % WRITE PREDICTIONS
    disp('WRITE PREDICTIONS')
    command = ['python ./shownet.py -f ',INPUT_NN,' --write-predictions=',PREDICTIONS_BATCH_FILE,' --multiview-test=',USE_MULTIVIEW,' --test-range=',TEST_RANGE];
    disp(command);
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end    
    
    % CONVERT PREDICTIONS TO TEXT FILE
    disp('CONVERT PREDICTIONS TO TEXT FILE')
    command = ['python ./lymph-nodes/predict_multiview.py ',PREDICTIONS_BATCH_FILE,' ',PREDICTIONS_TEXT_FILE,' ',USE_MULTIVIEW];
    disp(command);
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end  
    
    cd(curr_dir)
    
    
