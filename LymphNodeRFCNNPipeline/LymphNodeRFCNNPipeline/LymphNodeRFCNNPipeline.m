function mps_outputFolder = LymphNodeRFCNNPipeline(FileName,optRF,optCNNimages,optCNNpredictions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%          RUN CADe                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t1 = tic;

% RF candidates
addpath('../LymphNodeRandForestCandidates/')
centroids_RF_physicalPointsFilePrefix = AbdominalLymphNodeCAD(FileName,optRF);

% Generate CNN image patches
CNN_image_folder = LymphNodeGenerateCNNImagePatches(FileName,centroids_RF_physicalPointsFilePrefix,optCNNimages);

% Make CNN predictions
[BatchFolder, PredictionsTextFile] = LymphNodeMakeCNNPredictions(CNN_image_folder,optCNNpredictions);

% Fuse CNN predictions to final CADe marks
mps_outputFolder = LymphNodeFuseCNNPredictions(CNN_image_folder,BatchFolder,PredictionsTextFile,optCNNpredictions);

disp('****************************************************************************************************************************************')
disp(['   RESULTS (can be displayed with MITK): ',mps_outputFolder])
disp('****************************************************************************************************************************************')
disp(' Total time:')
disptime(toc(t1));
