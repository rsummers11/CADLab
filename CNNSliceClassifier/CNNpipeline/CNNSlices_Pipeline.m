function output_prob_volume = CNNSlices_Pipeline(imagename,output_dir,optSlices,optConvNet)

if ispc
    error('This code requires a LINUX system (e.g. Ubuntu)!')
end

%% RUN
tic

%% Extract slice images
slices_dir = [output_dir,filesep,'slices'];
slices_outprefix = extractImageSlices(imagename,slices_dir,optSlices);

%% make batches and compute ConvNet predictions
[BatchFolder, NumberBatches, PredictionsTextFile] = Slices_ConvNet_Predictions(slices_dir,optConvNet);

%% convert predictions to volume
if ~(isfield(optSlices,'corrValue') && ~isempty(optSlices.corrValue))
    window = [optSlices.minWin, optSlices.maxWin];
    get_prediction_probs(imagename,PredictionsTextFile,window)
else
    get_prediction_probs(imagename,PredictionsTextFile)
end

%% Done
disp('************************************************************************')
disp('************************************************************************')
toc
%disp(['ConvNet predictions saved at: ', output_prob_volume])
disp('************************************************************************')
disp('************************************************************************')


