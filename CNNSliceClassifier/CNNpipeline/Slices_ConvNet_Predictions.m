function [OUT_BATCH_FOLDER, NumberBatches, PREDICTIONS_TEXT_FILE] = Slices_ConvNet_Predictions(FolderName,opt)

    %% PARAMS    
    LABEL_TYPE='anatomy';
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
    
    INPUT_FOLDER=FolderName;
    OUT_BATCH_FOLDER=SAVE_PATH;
    START_IDX=num2str(BatchStartIndex);
    SEARCH_STR=ImageSearchString;
    IMG_SIZE=num2str(ImageSize);
    IMG_CHANNELS=num2str(ImageChannels);
    DO_SHUFFLE='0'; % not useful for testing
    
    NumberBatches  = 1;        
    NUMBER_BATCHES=num2str(NumberBatches);
    
    command = ['python ', ConvNetSrcFolder, '/make_data/make_general_data.py ', INPUT_FOLDER,' ', OUT_BATCH_FOLDER,' ', START_IDX,' ', NUMBER_BATCHES,' ', SEARCH_STR,' ', IMG_SIZE,' ', IMG_CHANNELS,' ', DO_SHUFFLE,' ', LABEL_TYPE,' ',existingMeanImageFilename];
    disp(command)
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end
    
    %% predict multiview
    NET = CNNmodel;
    OUTPUT_NOTE=LABEL_TYPE;
    DATA_SET_PATH = OUT_BATCH_FOLDER;
    TEST_RANGE = [START_IDX,'-',NUMBER_BATCHES]; % in case we want to add more batches later
    PREDICTIONS_DIR=[DATA_SET_PATH,'/',OUTPUT_NOTE,'_predictions'];
    %%%TEST_RANGE = NUMBER_BATCHES;
%     if exist(INPUT_NN,'file') % checkpoint file
%         INPUT_MODEL_DIR = fileparts(fileparts(INPUT_NN));
%     else % is already convnet dir
%         INPUT_MODEL_DIR = fileparts(INPUT_NN);
%     end

    % WRITE PREDICTIONS
    disp('WRITE PREDICTIONS') 
    %  --test-only 1 --multiview-test 0! write-features and multi-view
    %  cannot
    command = ['python ./convnet.py --load-file ', NET,' --data-path=',DATA_SET_PATH,' --write-feature probs --feature-path ',PREDICTIONS_DIR,' --train-range ',TEST_RANGE,' --test-range ',TEST_RANGE,' --test-only 1 --multiview-test 0'];
    disp(command);
    [status,result] = system(command,'-echo');
    if status~=0
        error(result);
    end    
    
    % each prediction batch has to be converted separately!
    for b = BatchStartIndex:NumberBatches
        TEST_RANGE=num2str(b);
        DATA_BATCH_TEXT_FILE=[DATA_SET_PATH,'/data_batch_',TEST_RANGE,'.txt'];        
        PREDICTIONS_BATCH_FILE=[PREDICTIONS_DIR,'/data_batch_',TEST_RANGE];
        PREDICTIONS_TEXT_FILE=[PREDICTIONS_BATCH_FILE,'.txt'];        
        % CONVERT PREDICTIONS TO TEXT FILE
        disp('CONVERT PREDICTIONS TO TEXT FILE')
        command = ['python ./predict_multiview.py ',PREDICTIONS_BATCH_FILE,' ', PREDICTIONS_TEXT_FILE,' ', DATA_BATCH_TEXT_FILE,' ', LABEL_TYPE];
        disp(command);
        [status,result] = system(command,'-echo');
        if status~=0
            error(result);
        end  
    end
    cd(curr_dir)
