function       [Patient_Unique_Roi_Probs,...
                Patient_Unique_Roi_Names,...
                Patient_N_predictions,...
                Patient_Unique_Roi_Lables,...
                Patient_Unique_Predictions] = ...
                    LymphNodeFuseCNNMultiviewPredictions(itkGetRegionOfInterestCentersFromList_exe,batch_log_file,prediction_file,N_samples_to_test,corrdinates_path,unique_roi_nifti_str,...
                    showMPRpredictions,mps_outputFolder,operating_probability,export_suffix)
    
    if exist('mps_outputFolder','var') && exist('operating_probability','var') && exist('export_suffix','var')
        export_CAD_marks = true;
    else
        error(' Not all required variables are set to export CADe marks!')
    end
    
    % for saving figures
    [prediction_base, prediction_model_name, ~] = fileparts(prediction_file);
    output_fig_prefix = [prediction_base,filesep,prediction_model_name];    

    % GET PREDICTIONS FOR EACH SAMPLE
    [Patient_IDs,...
    Patient_Rois,...
    Patient_Predictions,...
    Patient_Labels] = getPredictionsAndLabels(batch_log_file,prediction_file);    
    
    pred_to_test = true;    
    
    % GET PROBS AT DIFFERENT SAMPLE RATES
    N_sample_tests = numel(N_samples_to_test);  
    Patient_Unique_Roi_Probs = cell(N_sample_tests,1);
    Patient_Unique_Roi_Names = cell(N_sample_tests,1);
    Patient_N_predictions = cell(N_sample_tests,1);
    Patient_Unique_Roi_Lables = cell(N_sample_tests,1);
    Patient_Unique_Predictions = cell(N_sample_tests,1);

    % get Probabilities for different N_samples_to_test
    for i = 1:N_sample_tests
        fprintf('compute PROBABILITIES for sampling %d of %d: N = %d\n',i,N_sample_tests,N_samples_to_test(i))
        [Patient_Unique_Roi_Probs{i},...
        Patient_Unique_Roi_Names{i},...
        Patient_N_predictions{i},...
        Patient_Unique_Roi_Lables{i},...
        Patient_Unique_Predictions{i}] = getProbabilities(Patient_Rois,Patient_Predictions,Patient_Labels,pred_to_test,N_samples_to_test(i));
    end
    
    % centroids are the same for all N_samples_to_test (compute only once)
    [Patient_Unique_Roi_Centroids, Patient_Unique_Roi_Centroid_Filenames] = getRoiCentroids(itkGetRegionOfInterestCentersFromList_exe,Patient_IDs,Patient_Unique_Roi_Names{1},corrdinates_path,unique_roi_nifti_str);
    
    % get TP and FP FROC curves based on ground truth
    for i = 1:N_sample_tests
        fprintf(' Exporting CADe marks to %s\n',mps_outputFolder);
        exportCADe(Patient_IDs,Patient_Unique_Roi_Centroids,Patient_Unique_Roi_Probs{end},mps_outputFolder,operating_probability,export_suffix);            
    end
     
    if showMPRpredictions
       Patient_Unique_Roi_Centroid_Filenames 
    end
    
end % main()

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    FUNCTIONS                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% exportCADe()
function exportCADe(Patient_IDs,Patient_Unique_Roi_Centroids,Patient_Unique_Roi_Probs,mps_outputFolder,operating_probability,export_suffix)
	if exist(mps_outputFolder,'dir')
		warning(' Removing previous results in %s ...', mps_outputFolder);	
		rmdir(mps_outputFolder,'s'); 
	end
    mkdir(mps_outputFolder);
    
    N_patients = numel(Patient_IDs); 
    for i = 1:N_patients
        curr_Centroids = Patient_Unique_Roi_Centroids{i};
        curr_Probs = Patient_Unique_Roi_Probs{i};
        curr_Patient_ID = Patient_IDs{i};

        fprintf('  exporting CADe marks for patient %d of %d: %s at %g probability...\n',i,N_patients,curr_Patient_ID,operating_probability);

        % save all CADe marks
        curr_outputFolder = [mps_outputFolder,filesep,curr_Patient_ID];
        mkdir(curr_outputFolder);
        write2mps(curr_Centroids,[curr_outputFolder,export_suffix,'_all.mps']);
        writeMatrixToFile(curr_Centroids,[curr_outputFolder,export_suffix,'.txt']);        
        % and all CNN probs
        writeMatrixToFile(curr_Probs,[curr_outputFolder,export_suffix,'_CNNprobs.txt']);                
        
        for o = 1:numel(operating_probability)
            % save CADe marks greater operating_probability
            CADe = curr_Centroids(curr_Probs>operating_probability(o),:);

            curr_outputFolder = [mps_outputFolder,filesep,curr_Patient_ID];
            if ~isdir(curr_outputFolder)
                mkdir(curr_outputFolder);
            end
            write2mps(CADe,        [curr_outputFolder,export_suffix,'_greater',num2str(operating_probability(o)),'.mps']);
            writeMatrixToFile(CADe,[curr_outputFolder,export_suffix,'_greater',num2str(operating_probability(o)),'.txt']);
        end
    end
end

%% getTrueAndFalsePredictionsBasedOnDistance
function [Pos_Probs, Neg_Probs, Pos_Filenames, Neg_Filenames, PosNeg] = getTrueAndFalsePredictionsBasedOnDistance(Patient_IDs,Patient_Unique_Roi_Probs,Patient_Unique_Roi_Centroids,...
                                                                                    ground_truth_folder,ground_truth_suffix,ground_truth_threshold,object_level,...
                                                                                    Patient_Unique_Roi_Centroid_Filenames)
    if exist('Patient_Unique_Roi_Centroid_Filenames','var')
        assignFilenamesToProbs = true;
    else
        assignFilenamesToProbs = false;
    end 
                                            
    if ~exist('object_level','var')
        object_level = false;
    end
    
    if object_level
        disp('computing TP and FP at OBJECT level...')
    else
        disp('computing TP and FP at CANDIDATE level...')
    end

    Pos_Probs = [];
    Neg_Probs = [];
    Pos_Filenames = [];
    Neg_Filenames = [];
    N_patients = numel(Patient_IDs); 
    PosNeg = cell(1,N_patients);
    for i = 1:N_patients
        fprintf('get TP and FP probabilites for patient %d of %d...\n',i, N_patients);
        
        current_Patient_ID = Patient_IDs{i};
        curr_Probs = Patient_Unique_Roi_Probs{i};
        curr_CADe_Centroids = Patient_Unique_Roi_Centroids{i};
        if assignFilenamesToProbs
            curr_Filenames = Patient_Unique_Roi_Centroid_Filenames{i};
        end
        
        % get ground truth centroids 
        if iscell(ground_truth_folder)
            for c = 1:numel(ground_truth_folder)
                if isdir(ground_truth_folder{c})
                    if iscell(ground_truth_suffix)
                        grTr_suffix = ground_truth_suffix{c};
                    else
                        grTr_suffix = ground_truth_suffix;
                    end
                    rGroundTruthName = rdir([ground_truth_folder{c},filesep,current_Patient_ID,filesep,current_Patient_ID,grTr_suffix]); 
                    if ~isempty(rGroundTruthName)
                        break;
                    end
                else
                   error(' No such dir: %s !', ground_truth_folder{c}) 
                end
            end
        else
            rGroundTruthName = rdir([ground_truth_folder,filesep,current_Patient_ID,filesep,current_Patient_ID,ground_truth_suffix]);
        end
        if numel(rGroundTruthName) ~= 1
            error('Duplicate or no ground truth file for patient %s',current_Patient_ID)
        else
            if ~isEmptyFile(rGroundTruthName.name)
                GT = dlmread(rGroundTruthName.name);
            else
                GT = [];
            end
        end
            
        N_CADe = size(curr_CADe_Centroids,1);
        curr_pos_neg = zeros(N_CADe,1);
        if ~isempty(GT)
            N_GT = size(GT,1);
            D = NaN*ones(N_CADe,N_GT);
            for g = 1:N_GT
                D(:,g) = getEuclideanDistance(curr_CADe_Centroids,GT(g,:));
            end
            [dist_to_closest_GT, closest_LN] = min(D,[],2); % get closest GT for each CADe mark

            % TP and FP probs at candidate level
            pos_idx = dist_to_closest_GT <= ground_truth_threshold;
            neg_idx = dist_to_closest_GT  > ground_truth_threshold;
            curr_Pos_Probs = curr_Probs( pos_idx );
            closest_Pos_LN = closest_LN( pos_idx );
            curr_pos_neg( pos_idx ) = true;
            curr_pos_neg( neg_idx ) = false;
            if sum(pos_idx) == 0
                warning('getTrueAndFalsePredictionsBasedOnDistance:warning',' There are no positive detection for patient: %s !\n',current_Patient_ID);
            end
            curr_Neg_Probs = curr_Probs( neg_idx );

            if assignFilenamesToProbs
                curr_Pos_Filenames = curr_Filenames( pos_idx );
                curr_Neg_Filenames = curr_Filenames( neg_idx );
            end

            if object_level
                % TP and FP probs at candidate level
                % merge TP candidates at object level
                merged_Pos_Probs = [];
                merged_Pos_Filenames = [];
                for g = 1:N_GT
                    %only use the maximum prob. candidate at each object
                    [max_prob, max_id] = max( curr_Pos_Probs(closest_Pos_LN==g) );
                    merged_Pos_Probs = [merged_Pos_Probs; max_prob];
                    if assignFilenamesToProbs
                       merged_Pos_Filenames = [merged_Pos_Filenames; curr_Pos_Filenames(max_id)]; 
                    end
                end
                Pos_Probs = [Pos_Probs; merged_Pos_Probs];    
                if assignFilenamesToProbs
                    Pos_Filenames = [Pos_Filenames; merged_Pos_Filenames];
                end                
                
                if object_level
                    N_TP = numel(merged_Pos_Probs);
                    N_FN = N_GT - N_TP; % number of false negatives, i.e. missed lesions
                    if N_FN<0 || isnan(N_FN)
                        disp(N_FN);
                        error(' something went wrong when computing false negatives!\n');
                    end                         
                    N_FP = sum(neg_idx);                            
                end
            else
                % use all TP probabilities
                Pos_Probs = [Pos_Probs; curr_Pos_Probs];   
                if assignFilenamesToProbs
                    Pos_Filenames = [Pos_Filenames; curr_Pos_Filenames];
                end
            end
        else % just use Neg_Probs
            if object_level
                N_FN = N_GT; % (all lesions missed) number of false negatives, i.e. missed lesions
            end
                        
            curr_Neg_Probs = curr_Probs;
            curr_pos_neg(:) = false;
            
            warning(' No GT for patient %s (just Neg_Probs)!', current_Patient_ID);
        end
        if object_level
            if N_FN>0
                fprintf(' There are %d False Negatives!\n', N_FN);
            end       
            Pos_Probs = [Pos_Probs; NaN*ones(N_FN,1)]; % add false negatives as NaN probabilites for later FROC computation
            Pos_Filenames = [Pos_Filenames; cell(N_FN,1)];
        end
        
        Neg_Probs = [Neg_Probs; curr_Neg_Probs];
        if assignFilenamesToProbs
            Neg_Filenames = [Neg_Filenames; curr_Neg_Filenames];
        end        
        PosNeg{i} = curr_pos_neg; 
    end
    
end % getTrueAndFalsePredictionsBasedOnDistance

%% getTrueAndFalsePredictionsBasedOnLabels
function [Pos_Probs, Neg_Probs] = getTrueAndFalsePredictionsBasedOnLabels(Patient_IDs,Patient_Unique_Roi_Probs,Patient_Unique_Roi_Lables)

    pos_label = true;
    neg_label = false;

    Pos_Probs = [];
    Neg_Probs = [];
    N_patients = numel(Patient_IDs); 
    for i = 1:N_patients
        fprintf('get TP and FP probabilites for patient %d of %d...\n',i, N_patients);
        
        current_Patient_ID = Patient_IDs{i};
        curr_Probs = Patient_Unique_Roi_Probs{i};
        curr_Labels = Patient_Unique_Roi_Lables{i};
        
        Pos_Probs = [Pos_Probs; curr_Probs( curr_Labels==pos_label )];
        Neg_Probs = [Neg_Probs; curr_Probs( curr_Labels==neg_label )];
    end
    
end % getTrueAndFalsePredictionsBasedOnLabels

%% getPredictionsAndLabels
function [Patient_IDs,...
          Patient_Rois,...
          Patient_Predictions,...
          Patient_Labels] = getPredictionsAndLabels(batch_log_file,prediction_file)
    
    tic
    
    pos_groundtruth_label='pos'; % true
    neg_groundtruth_label='neg'; % false
    prob_idx = 2; %[0, 1] labelling in CNN
    
    pos_pred_label='true_lymphnode'; % true
    neg_pred_label='false_lymphnode'; % false

    Patient_IDs = cell(1);
    Patient_Rois = cell(1); 
    Patient_Predictions = cell(1);
    Patient_Labels = cell(1);
    %Patient_Centroids = cell(1);

    fid_log = fopen(batch_log_file,'r');
    fid_pred = fopen(prediction_file,'r');

    roi_count = 0;
    while true
        tline_log = fgetl(fid_log);    
        if isempty(strfind(tline_log,'#')) % ignore number of files
            tline_pred = fgetl(fid_pred);             
            if isempty(strfind(tline_pred,'[')) % ignore lines without probs: e.g. 'id,label,pred'
                tline_pred = fgetl(fid_pred);
            end
            if ~ischar(tline_log) || ~ischar(tline_pred) % not sure why this is necessary...
                break
            end
            %disp(tline_log)
            %disp(tline_pred)

            % get prediction
            curr_prob = str2num(tline_pred(strfind(tline_pred,'[ ')+1:strfind(tline_pred,']')-1)); % str2double does not work for more than 1 number!
            if any(isnan(curr_prob)) || isempty(curr_prob)
                error(' Could not get probability from %s: %g !',tline_pred,curr_prob);
            end
            if abs(sum(curr_prob)-1.0)>1e-6
                error(' Probabilities do not add to 1.0 at %s !',tline_pred);
            end
            if strfind(tline_pred,pos_pred_label)
                curr_pred = curr_prob(prob_idx); % only use probability of being a true lymph node!
            elseif strfind(tline_pred,neg_pred_label)
                curr_pred = curr_prob(prob_idx);
            else
                error(['Unknown prediction label in line: ',tline_pred])
            end
            % get label from batch log
            [roi_path, roi_name] = fileparts(tline_log);
            if ~isempty(strfind(roi_name,pos_groundtruth_label))
                curr_label = 1;
            elseif ~isempty(strfind(roi_name,neg_groundtruth_label))
                curr_label = 0;
            else
                curr_label = -1;
                %warning(['Unknown ground truth label in line: ',tline_log])
            end    

            % get patient ID
            curr_patient_id = roi_name(1:strfind(roi_name,'_')-1);

            % get ROI centeroid coordinate
            roi_str_idx = strfind(roi_name,'ROI');
            roi_name_begin = roi_name(1:roi_str_idx+2);
            roi_name_rest = roi_name(roi_str_idx+3:end);
            roi_id_str = roi_name_rest(1:find(roi_name_rest=='_',1,'first')-1);
            curr_roi_id = str2double(roi_id_str);        
            if isnan(curr_roi_id)
                error(['curr_roi_id is NaN: ',roi_name]);
            end        
            %curr_roi_name_and_id = [roi_name_begin,roi_id_str];
            curr_roi_name_and_id = [roi_path,filesep,roi_name_begin,roi_id_str]; % ONLY FOR BALANCED DATA
            % add new ROI
            roi_count = roi_count + 1;
            if mod(roi_count,5000) == 0
                fprintf('%d ROIs processed...\n',roi_count);
                toc
                tic
            end

            if roi_count==1
                Patient_IDs{1} = curr_patient_id;
            end
            if ~isempty(Patient_IDs{1}) && ~isempty(curr_patient_id) && ischar(curr_patient_id)
                Lia = ismember(Patient_IDs,curr_patient_id);
            else
                error([' error adding new patient: ',curr_patient_id])
            end
            if ~any(Lia) % new patient
                Patient_IDs{numel(Patient_IDs)+1} = curr_patient_id;
                cell_roi_name = cell(1);
                cell_roi_name{1} = curr_roi_name_and_id;            
                Patient_Rois{numel(Patient_Rois)+1} = cell_roi_name;
                Patient_Predictions{numel(Patient_Predictions)+1} = curr_pred;
                Patient_Labels{numel(Patient_Labels)+1} = curr_label;
                %Patient_Centroids{numel(Patient_Centroids)+1} = curr_Centroid;
            else % add to existing patient
                % ROI numbers
                if ~isempty(Patient_Rois{Lia})
                    currROIs = Patient_Rois{Lia};
                    currROIs{numel(currROIs)+1} = curr_roi_name_and_id;
                    Patient_Rois{Lia} = currROIs;        
                else
                    cell_roi_name = cell(1);
                    cell_roi_name{1} = curr_roi_name_and_id;
                    Patient_Rois{Lia} = cell_roi_name;        
                end
                % Predictions
                currPredictions = Patient_Predictions{Lia};
                currPredictions = [currPredictions; curr_pred];
                Patient_Predictions{Lia} = currPredictions;
                % Ground truth labels
                currLabels = Patient_Labels{Lia};
                currLabels = [currLabels; curr_label];
                Patient_Labels{Lia} = currLabels;
                % Centroid Coordinates
                %currCentroids = Patient_Centroids{Lia};
                %currCentroids = [currCentroids; curr_Centroid];
                %Patient_Centroids{Lia} = currCentroids;
            end
        end
    end
    
    fclose(fid_log);
    fclose(fid_pred);    
end % getPredictionsAndLabels

%% getProbabilities
function [Patient_Unique_Roi_Probs,...
          Patient_Unique_Roi_Names,...
          Patient_N_predictions,...
          Patient_Unique_Roi_Lables,...
          Patient_Unique_Predictions] = getProbabilities(Patient_Rois,Patient_Predictions,Patient_Labels,pred_to_test,N_samples_to_test)
      
    % Find unique ROIs
    N_patients = numel(Patient_Rois);
    fprintf('found %d patients.\n',N_patients);
    
    Patient_Unique_Predictions  = cell(1,N_patients);
    Patient_Unique_Roi_Probs = cell(1,N_patients);
    Patient_Unique_Roi_Names = cell(1,N_patients);
    Patient_Unique_Roi_Lables = cell(1,N_patients);
    Patient_N_predictions  = cell(1,N_patients);
    
    for i = 1:N_patients
        fprintf('get probabilites for patient %d of %d...\n',i, N_patients);
        curr_Rois = Patient_Rois{i};
        curr_Predictions = Patient_Predictions{i};
        curr_Labels = Patient_Labels{i};
        
        % rename ROIs with multipe scales (and rename unspecific pos tag
        % ('_pos_','_posmanu_' and '_neg_','_negauto_' is the same)
        for c = 1:numel(curr_Rois)
            roi = curr_Rois{c};
            roi_str_idx1 = strfind(roi,'_s'); %s
            roi_str_idx2 = strfind(roi,'mm_')+2;
            curr_Rois{c} = [roi(1:roi_str_idx1),roi(roi_str_idx2:end)];
            curr_Rois{c} = strrep(curr_Rois{c},'pos_','posmanu_');
            curr_Rois{c} = strrep(curr_Rois{c},'neg_','negauto_');  
        end
        
        [curr_unique_Rois, ~, unique_roi_idx] = unique(curr_Rois);
        
        N_unique_Rois = numel(curr_unique_Rois);
        curr_unique_Probs = NaN*ones(N_unique_Rois,1);
        curr_unique_Labels = NaN*ones(N_unique_Rois,1);
        curr_N_predictions = zeros(N_unique_Rois,1);
        curr_random_predictions = cell(N_unique_Rois,1);
        for r = 1:N_unique_Rois
            % gold standard label mean
            curr_unique_Labels(r) = mean( curr_Labels(unique_roi_idx==r) );
            if ( curr_unique_Labels(r)-round(curr_unique_Labels(r)) ) ~= 0.0
                error(' The label should be the same for all samples of a unique ROI!')
            end
            % get CNN prediction probability
            predictions = curr_Predictions( unique_roi_idx==r );
            curr_random_predictions{r} = predictions;
            N_predictions = numel(predictions);
            curr_N_predictions(r) = N_predictions;
            
            if isnan(N_samples_to_test) || (N_predictions <= N_samples_to_test)
                %only from binary labels (does not work anymore): curr_unique_Probs(r) = sum( predictions == pred_to_test )/N_predictions; % prediction probability of being 'pred_to_test'
                %curr_unique_Probs(r) = sum( predictions>0.5 == pred_to_test )/N_predictions; % same as old binary average
                curr_unique_Probs(r) = mean( predictions );
                %curr_unique_Probs(r) = median( predictions ); % 1% better sensistivity in Abd :-) compared to binary averaging, 2% better than mean!
                %curr_unique_Probs(r) = max( predictions ); % does not seem to work as well as mean!
%DEBUG: use ground truth as sanity test:
%curr_unique_Probs(r) = curr_unique_Labels(r);
%END DEBUG
            else 
                 rand_perm_idx = randperm(N_predictions);
                 rand_idx = rand_perm_idx(1:N_samples_to_test);
                 %only from binary labels (does not work anymore): curr_unique_Probs(r) = sum( predictions(rand_idx) == pred_to_test )/N_samples_to_test; % prediction probability of being 'pred_to_test' using N_samples_to_test
                 %curr_unique_Probs(r) = sum( predictions(rand_idx)>0.5 == pred_to_test)/N_predictions; % same as old binary average
                 curr_unique_Probs(r) = mean( predictions(rand_idx) ); 
                 %curr_unique_Probs(r) = median( predictions(rand_idx) ); 
                 %curr_unique_Probs(r) = max( predictions(rand_idx) ); 
%DEBUG: use ground truth as sanity test:
%curr_unique_Probs(r) = curr_unique_Labels(r);
%END DEBUG
            end            
        end
        
        if ( sum(curr_N_predictions - mean(curr_N_predictions)) ) ~= 0.0
            warning('getPredictions:warning',' The number of samples differes between unique ROIs!')
            disp(' Unique numbers of predictions:')
            disp(unique(curr_N_predictions))
        end
    
        % add to patient cells
        Patient_Unique_Predictions{i} = curr_random_predictions;
        Patient_Unique_Roi_Probs{i} = curr_unique_Probs;
        Patient_Unique_Roi_Names{i} = curr_unique_Rois;
        Patient_Unique_Roi_Lables{i} = curr_unique_Labels;
        Patient_N_predictions{i}  = curr_N_predictions;            
        
    end % N_patients
        
    toc
end % getProbabilities

%% getRoiCentroids
function [Patient_Unique_Roi_Centroids, Patient_Unique_Roi_Centroid_Filenames] = getRoiCentroids(itkGetRegionOfInterestCentersFromList_exe,Patient_IDs,Patient_Unique_Roi_Names,corrdinates_path,unique_roi_nifti_str)
 
    N_patients = numel(Patient_IDs);
    fprintf('found %d patients.\n',N_patients);
   
    Patient_Unique_Roi_Centroids = cell(1,N_patients);
    Patient_Unique_Roi_Centroid_Filenames = cell(1,N_patients);
        
    hw_roi = waitbar(0,'looking for centroid coordinates...');
    for i = 1:N_patients        
        % get centroid file names
        tic
        curr_Patient_ID = Patient_IDs{i};
        curr_unique_Rois = Patient_Unique_Roi_Names{i};
        N_unique_Rois = numel(curr_unique_Rois);
        
        FileNames = cell(N_unique_Rois,1);
        
        if N_unique_Rois==0
            error('  There are no ROIs!')
        end
        % get Roi centroids
        fprintf('  find centroids for %d ROIs for patient %d of %d...\n', N_unique_Rois, i, N_patients);
        for v = 1:N_unique_Rois
            curr_Roi_name = curr_unique_Rois{v};
            roi_names_file = [corrdinates_path,filesep,sprintf('%s_Roi_filenames.txt',curr_Patient_ID)];
            [~, curr_Roi_name_name] = fileparts(curr_Roi_name);
            curr_Roi_volume_search_str = [corrdinates_path,filesep,strrep(curr_Roi_name_name,'__','*'),unique_roi_nifti_str]; 
            curr_ROI_volume_filename = rdir(curr_Roi_volume_search_str);         

            if isempty(curr_ROI_volume_filename)
                error(' Could not find volume for ROI %s\n!',curr_Roi_name);
            end
            % add to file of roi names
            s = blanks(500); % use empty string to force equally sized cells 
            s(1:numel(curr_ROI_volume_filename(1).name)) = curr_ROI_volume_filename(1).name;
            FileNames{v,1} = s;
            
            waitbar(v/N_unique_Rois,hw_roi,sprintf('looking for centroid coordinate %d of %d',v,N_unique_Rois));
        end
        Patient_Unique_Roi_Centroid_Filenames{i} = FileNames;
        dlmwrite(roi_names_file,FileNames,''); % very fast
        fprintf(' ... wrote %d Roi Filenames to %s .\n', N_unique_Rois, roi_names_file);
        
        % get coordinates
        fprintf(' get for %d centroid coordinates from filenames for patient %d of %d.\n',N_unique_Rois,i,N_patients);
        % get ROI centroid using ITK
        centroids_fileprefix = [strrep(roi_names_file,'.txt',''),'_centroids_mm'];
        command = [itkGetRegionOfInterestCentersFromList_exe,' ',roi_names_file,' ',centroids_fileprefix,' 0'];
        [status,result] = dos(command);
        if status~=0
           error(result)
        end  
        fidC = fopen([centroids_fileprefix,'.txt'],'r'); 
        curr_unique_Centroids = textscan(fidC,'%f %f %f');
        if numel(curr_unique_Centroids{1}) ~= N_unique_Rois
            error('  Number of centroids coords does not equal number of ROIs!');
        end
        fclose(fidC);
        
        % add to patient cells
        Patient_Unique_Roi_Centroids{i} = [curr_unique_Centroids{1}, curr_unique_Centroids{2}, curr_unique_Centroids{3}];
       
        toc
    end
    close(hw_roi);

end % getRoiCentroids

%% showRandomMPRpredictions
function showRandomMPRpredictions(Probs,Filenames,N_rows,N_cols,title_str)

    N = N_rows*N_cols;
    N_probs = numel(Probs);
    N_Filenames = numel(Filenames);
    if N_probs ~= N_Filenames
        error('showRandomMPRpredictions: N_probs ~= N_Filenames !')
    end
    if N > N_probs
        N = N_probs;
        warning('showRandomMPRpredictions: N > N_probs: set to be N = N_probs!');
    end

    rand_idx = randperm(N_probs);
    rand_idx = rand_idx(1:N);
    
    fig = figure;
    for s = 1:N;
        idx = rand_idx(s);
        curr_filepath = Filenames{idx};
        [~, curr_name] = fileparts(curr_filepath);
        curr_scale = (curr_name(strfind(curr_name,'_s')+2:strfind(curr_name,'mm')+1));
        [V, ~, vdim] = read_nifti_volume(curr_filepath);
        subplot(N_rows,N_cols,s)
        fprintf('show MPR of %s ...\n',curr_name);
        showMPR(V,vdim);
        hold on
        text(0,0,0,curr_scale);
        title([title_str,': p = ',num2str(Probs(idx),'%10.2f')]);
    end
    shg

end % showRandomMPRpredictions
