function [physicalPointsFilePrefix, centroids_RF_mm, probability_volume_RF, SR, outputPrefix] = AbdominalLymphNodeCAD(FileName,opt)

    %% PARAMS
    if ~isfield(opt,'itkConvertIndexToPhysicalPointsExe')
        opt.itkConvertIndexToPhysicalPointsExe = '';
    end
    
    if ~isfield(opt,'save_flag')
        opt.save_flag = true;
    end
    if ~isfield(opt,'t_prob')
        opt.t_prob = 0.25; % only keep probabilities >= t_prob
    end    
    if ~isfield(opt,'randForestFilename')
        opt.randForestFilename = 'D:\HolgerRoth\projects\lymph_node_detection\LymphNodeRandForestCandidates\trained_Forests\RF_regression.mat';
    end        
    if ~isfield(opt,'randForestModelName')
        opt.randForestModelName = 'model_reg_f5';
    end    
        
    %SEARCH REGION PARAMETERS
    if ~isfield(opt,'t_low')
        opt.t_low = 700; 
    end
    if ~isfield(opt,'t_high')
        opt.t_high = 1250;
    end        
    if ~isfield(opt,'t_low_atery')
        opt.t_low_atery = 1175;
    end
    if ~isfield(opt,'t_high_atery')
        opt.t_high_atery = 1275;   
    end
    if ~isfield(opt,'skip_voxels')
        opt.skip_voxels = 1;
    end
    if ~isfield(opt,'rErodeDilate')
        opt.rErodeDilate = 1;
    end
    
    %FEATURE PARAMETERS
    if ~isfield(opt,'n')
        opt.n = 5; % filter size
    end
    
    if ~isfield(opt,'N_feature_display')
        opt.N_feature_display = 2000;    
    end
    
    parseParams;
    
    %% RUN
    padding = ceil(2*n);
    
    fprintf('Running AbdominalLymphNodeCAD\n')
    
    FileName = strrep(FileName,'"','');
    FileName = strrep(FileName,' ','');
    [outpath, outname, inext] = fileparts(FileName);
    s1 = strfind(outname,'.');
    if ~isempty(s1)
        outname = outname(1:s1-1);
    end
    resultfolder = [outpath,filesep,outname,'_',randForestModelName,'_results'];
    mkdir(resultfolder);
    outputPrefix = [resultfolder,filesep,outname];    
    
    time1 = tic;
    
    % LOAD TRAINED RANDOM FOREST MODEL
    model = load(randForestFilename,randForestModelName);
    model = getfield(model,randForestModelName);

    % LOAD CT SCAN
    %    [FileName,PathName] = uigetfile('*.nii,*.nii.gz','Select input image');
    if ~isempty(strfind(FileName,'.nii'))
        [V, vsiz, vdim, ~, vhdr] = read_nifti_volume(FileName); 
    elseif strcmpi(inext,'.mat')
        V = load(FileName);
        vdim = V.image.spacing;
        V = V.image.value;
        vsiz = size(V);
        V = uint16(V); % Needs to be uint16 (as Kevin trained on that)
    else
        error(' Extension not supported for %s (%s)!',FileName, inext);
    end
    if ~exist('vhdr')
        vhdr = [];
    end

    corrVal = 1024;
    if min(V(:)) < 0
        warning(' This code assume int16 image values as input, not HU values! -> try corrections');
        V = uint16(V + corrVal); % Needs to be uint16 (as Kevin trained on that)
    end
    if min(V(:)) < 0
        error(' This code assume int16 image values as input, not HU values! Correction unsuccessful!');
    end    
    spacing = vdim;
    NX = vsiz(1);
    NY = vsiz(2);
    NZ = vsiz(3);
    N = NX*NY*NZ;

    %SET SEARCH REGION
    %[SR, limits] = getAterialRegion(V, t_low_atery, t_high_atery, t_low, t_high, rDilate);
    
    sourceImageFile = [SearchRegionAtlas_dir,'/Abd1_body.nii.gz'];
    sourceAtlasFile = [SearchRegionAtlas_dir,'/Abd1_SearchRegionClosed03.nii.gz'];
    [SR, limits] = getSearchRegionByAtlasRegistration(NiftyRegAppsDir,sourceImageFile,sourceAtlasFile,FileName,...
                                                        V,t_low, t_high, rErodeDilate);
    SR(V>t_low_atery & V<t_high_atery) = 0; % exclude atery
    x1 = limits(1,1);
    x2 = limits(1,2);
    y1 = limits(2,1);
    y2 = limits(2,2);
    z1 = limits(3,1);
    z2 = limits(3,2);                                                        
                                                    
    N_SR = sum(SR(:)>0);                                                    
    fprintf(' There are %d voxels in search region:\n %g percent of %d voxels total.\n',...
        N_SR, 100*N_SR/N, N);
    
    % show Search Region
    segmentationoverlay(V,SR);

    if save_flag
        write_nifti_volume(uint8(SR),vdim,[outputPrefix,'_SearchRegion.nii.gz'],vhdr);  % assume empty header! (no offsett ot detections)
    end    
    
    %% COMPUTE FEATURES
    if N_SR>0
        N_features = 12;

        nhood = ones(n,n,n);
        hg = fspecial3('gaussian',[n,n,1.5*n]);

        % Intesity
        if x1>padding && x2<=(NX-padding)
            xpadding = padding;
        else
            xpadding = 0;
        end
        if y1>padding && y2<=(NY-padding)
            ypadding = padding;
        else
            ypadding = 0;
        end   
        if z1>padding && z2<=(NZ-padding)
            zpadding = padding;
        else
            zpadding = 0;
        end   
        xidx = x1-xpadding:x2+xpadding;
        yidx = y1-ypadding:y2+ypadding;
        zidx = z1-zpadding:z2+zpadding;
        %I is padded a little so the filters dont have errors at top and bottom slice
        I = V(xidx,yidx,zidx);
          
        HoldFeatures = NaN*ones(vsiz);
        pixel_list = find(SR==1);
        pixel_features = NaN*ones(N_features,N_SR);
        
        disp(' intensity ...')
        pixel_features(1,:) = V(pixel_list)';

        disp(' intensity smoothed...')
        tic
        feat_idx = 2;
        HoldFeatures(xidx,yidx,zidx) = imfilter(I, hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % Entropy
        disp(' entropy...')
        tic    
        feat_idx = 3;
        HoldFeatures(xidx,yidx,zidx) = imfilter(entropyfilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % Range
        disp(' intensity range...')
        tic    
        feat_idx = 4;
        HoldFeatures(xidx,yidx,zidx) = imfilter(rangefilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % Std. dev.
        disp(' intensity range...')
        tic        
        feat_idx = 5;
        HoldFeatures(xidx,yidx,zidx) = imfilter(stdfilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % fft
        disp(' intensity fft...')
        tic       
        feat_idx = 6;
        HoldFeatures(xidx,yidx,zidx) = imfilter(fftn(I),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % blobness
        HoldFeatures(xidx,yidx,zidx) = double(blobness(I,spacing));

        feat_idx = 7;        
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';

        disp(' blobness smoothed...')
        tic    
        feat_idx = 8;
        HoldFeatures(xidx,yidx,zidx) = imfilter(I, hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % Entropy blobness
        disp(' blobness entropy...')
        tic        
        feat_idx = 9;
        HoldFeatures(xidx,yidx,zidx) = imfilter(entropyfilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % Range blobness
        disp(' blobness range...')
        tic         
        feat_idx = 10;
        HoldFeatures(xidx,yidx,zidx) = imfilter(rangefilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        clear Iran Irang
        toc

        % Std. dev. blobness
        disp(' blobness std. dev....')
        tic         
        feat_idx = 11;
        HoldFeatures(xidx,yidx,zidx) = imfilter(stdfilt(I,nhood),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc

        % fft blobness
        disp(' blobness fft...')
        tic      
        feat_idx = 12;
        HoldFeatures(xidx,yidx,zidx) = imfilter(fftn(I),hg);
        pixel_features(feat_idx,:) = HoldFeatures(pixel_list)';
        if sum(pixel_features(feat_idx,:))==0 || any(isnan(pixel_features(feat_idx,:)))
           %warning(' Feature %d was probably not computed correctly!', feat_idx)
        end 
        toc
        
        % display normed features
        normed_pixel_features = abs(pixel_features);
        max_pixel_features = max(normed_pixel_features,[],2);
        for i = 1:N_features
           normed_pixel_features(i,:) = normed_pixel_features(i,:)/max_pixel_features(i); 
        end
        
		try
        figure
        if N_feature_display < size(normed_pixel_features,2)
            imagesc(normed_pixel_features(:,1:N_feature_display)')
        else
            imagesc(normed_pixel_features')
        end
		end
        colormap jet;
        colorbar;
        title(sprintf('%d normed feature vectors',N_features),'FontSize',15);
        shg
    else
        error(' The search region is zero!')
    end

    %% SAVE FEATURES
    if save_flag
        save([outputPrefix,'_',num2str(N_features),'features.mat'],'pixel_features','-v7.3');
    end
    
	%% VOXEL PREDICTIONS   
    %Predict LN Probability for all pixels in SR
    tic
    disp('get prediction scores...')
    prediction_scores = regRF_predict(pixel_features',model);
    fprintf('Predictions score mean +- std = %g +- %g, range: %g to %g\n',...
        mean(prediction_scores), std(prediction_scores), min(prediction_scores), max(prediction_scores));
    toc
    
    %Create Volume
    probability_volume_RF = NaN*ones(vsiz);
    probability_volume_RF(pixel_list) = prediction_scores;
    probability_volume_RF = probability_volume_RF + 1.0; % scale probs. between 0.0 and 1.0
    probability_volume_RF(isnan(probability_volume_RF)) = -1.0; % set outside SR to -1.0
        
	if save_flag
        saveVarStr = 'probability_volume_RF';
        save([outputPrefix,'_',saveVarStr,'.mat'],saveVarStr,'-v7.3');        
    end
	fprintf('Finished Pixel Predictions for %s.\n', FileName)
	
	%% FIND LN CANDIDATES 
    if skip_voxels<=1    
        disp(' find local maximas...')
        % the 3D neighbors of each voxel
        maxNeighbors = ones(3,3,3);
        maxNeighbors(2,2,2) = 0;
    	tic
        a = probability_volume_RF(x1:x2,y1:y2,z1:z2);
        hg_prob = fspecial3('gaussian',[2*n,2*n,2*n]);
        a = imfilter(a, hg_prob,'symmetric'); % smooth probabilities
        a(a<t_prob) = -1.0;
        [Minima,Maxima] = findExtrema(a,maxNeighbors);
        Maxima(~SR(x1:x2,y1:y2,z1:z2)) = 0; % make sure no maxima is found outside SR!
        [xmax,ymax,zmax] = findnd(Maxima==1);
        centroids_RF_vx = [x1+xmax-2,y1+ymax-2,z1+zmax-2]; % add cropped region offset (c-style indexing)
        N_centroids = size(centroids_RF_vx,1);
        
        indicesFilename = [outputPrefix,'_centroids_RF_vx.dat'];
        physicalPointsFilePrefix = [outputPrefix,'_centroids_RF_mm'];
        writeMatrixToFile(centroids_RF_vx,indicesFilename);     
        command = [itkConvertIndexToPhysicalPointsExe,' ',FileName,' ',indicesFilename,' ',physicalPointsFilePrefix];
        [status, result] = dos(command,'-echo');        
        if status~=0
            error(result);
        end
        toc
    
        centroids_RF_mm = dlmread([physicalPointsFilePrefix,'.txt']);
        
		try
        figure
        subplot(1,2,1)
        showMPR(a,[1 1 1],[0.0, 1.0])
        colormap jet
        hold on
        grid on
        title('Smoothed RF probabilities','FontSize',15)
        subplot(1,2,2) 
        V_SR = V(x1:x2,y1:y2,z1:z2);
        showMPR(V_SR,[1 1 1],[t_low, t_high])
        hold on
        plot3(xmax,ymax,zmax,'or')
        title('Detections','FontSize',15)
        shg
		end
    else
        error(' No centroids computed as skip_voxels = %d!',skip_voxels);
        centroids_RF_vx = [];
        centroids_RF_mm = [];
        N_centroids = 0;     
        physicalPointsFilePrefix = [];
    end
	
	if save_flag
        saveVarStr = 'centroids_RF_vx';
        save([outputPrefix,'_',saveVarStr,'.mat'],saveVarStr,'-v7.3');
    end
	fprintf('Found %d CADe lymph node centroids for %s.\n', N_centroids, FileName)    
    
    disp('  total feature computation time:')
    disptime(toc(time1))

    %% save result
	if save_flag
        if isempty(vhdr)
            write_nifti_volume(V,vdim,[outputPrefix,'_image.nii.gz'],vhdr);  % assume empty header! (no offsett ot detections)
        end
        write_nifti_volume(probability_volume_RF,vdim,[outputPrefix,'_probabilities_RF.nii.gz'],vhdr);  

        
        %writeMatrixToFile(centroids_RF_mm,[outputPrefix,'_centroids_RF_mm.dat']);
        %write2mps(centroids_RF_mm,[outputPrefix,'_centroids_RF_mm.mps']);
    end    
    
