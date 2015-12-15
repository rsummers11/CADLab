function mps_outputFolder = LymphNodeFuseCNNPredictions(CNN_image_folder,BatchFolder,PredictionsTextFile,opt)

    %% PARAMS
    if ~isfield(opt,'NumberBatches')
        opt.NumberBatches = 20;
    end         
    if ~isfield(opt,'BatchStartIndex')
        opt.BatchStartIndex = 1;
    end     
    if ~isfield(opt,'operating_probability')
        opt.operating_probability = [0.25, 0.5, 0.75, 0.85, 0.90, 0.95];
    end         
    if ~isfield(opt,'showMPRpredictions')
        opt.showMPRpredictions = false;
    end        
    if ~isfield(opt,'unique_roi_nifti_str')
        opt.unique_roi_nifti_str = '_t00000_r00000.nii.gz';
    end
    if ~isfield(opt,'itkGetRegionOfInterestCentersFromList_exe')
        error('itkGetRegionOfInterestCentersFromList_exe is not defined!');
    end
    
    parseParams;
    
    %% RUN
    batch_log_file=combineBatchLogs(BatchFolder,[BatchStartIndex BatchStartIndex+NumberBatches-1]); 
    prediction_file=PredictionsTextFile;
    corrdinates_path = CNN_image_folder; 
    
    [outpath, outname] = fileparts(BatchFolder);
    mps_outputFolder = [outpath,filesep,outname,'_CNN_CADe_Results'];
	if exist(mps_outputFolder,'dir')
		warning(' Removing previous results in %s ...', mps_outputFolder);		
		rmdir(mps_outputFolder,'s'); 
	end	
    mkdir(mps_outputFolder);
    
    
    export_suffix = '_CNN_CADe';
    
    N_samples_to_test = NaN; % use all available samples    
    %% RUN

    [Patient_Unique_Roi_Probs,...
                Patient_Unique_Roi_Names,...
                Patient_N_predictions,...
                Patient_Unique_Roi_Lables,...
                Patient_Unique_Predictions] = LymphNodeFuseCNNMultiviewPredictions(itkGetRegionOfInterestCentersFromList_exe,batch_log_file,prediction_file,N_samples_to_test,corrdinates_path,unique_roi_nifti_str,...
        showMPRpredictions,mps_outputFolder,operating_probability,export_suffix);
    
    export_to_workspace;
    

