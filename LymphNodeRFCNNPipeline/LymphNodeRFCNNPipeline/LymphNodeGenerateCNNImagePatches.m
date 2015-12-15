function output_folder = LymphNodeGenerateCNNImagePatches(FileName,physicalPointsFilePrefix,opt)

%% PARAMS
    if ~isfield(opt,'itkResampleRegionOfInterestExe')
        opt.itkResampleRegionOfInterestExe = '';
    end        
    if ~isfield(opt,'pointList_suffix')
        opt.pointList_suffix = '_physicalPoints.txt';
    end        
    if ~isfield(opt,'outputLabel')
        opt.outputLabel = 'CADe';
    end
    if ~isfield(opt,'t_lower')
        opt.t_lower='-500';%[HU]
    end        
    if ~isfield(opt,'t_upper')
        opt.t_upper='500';%[HU] 
    end             
    if ~isfield(opt,'displacementFactorRandomTranslations')
        opt.displacementFactorRandomTranslations='3.0';%[mm]
    end
    if ~isfield(opt,'ratio')
        opt.ratio = 1;
    end        
    if ~isfield(opt,'numberRandomTranslations')
        opt.numberRandomTranslations=5;
    end        
    if ~isfield(opt,'numberRandomRotations')
        opt.numberRandomRotations=5;
    end        
    if ~isfield(opt,'scales')
        opt.scales = [30, 45]; %mm
    end
    if ~isfield(opt,'numberROIvoxels')
        opt.numberROIvoxels = 32; 
    end    
    if ~isfield(opt,'interpolationType')    
        opt.interpolationType='BSPLINE';
    end
    if ~isfield(opt,'transformType')    
        opt.transformType='XYZ';
    end    
    
    parseParams;
    
    %% RUN
    tic

    disp('scales:')
    disp(scales)
    
    physicalPointsFile = [physicalPointsFilePrefix,'.txt']; 
    
    FileName = strrep(FileName,'"','');
    FileName = strrep(FileName,' ','');
    [outpath, outname, inext] = fileparts(FileName);
    s1 = strfind(outname,'.');
    if ~isempty(s1)
        outname = outname(1:s1-1);
    end
    output_folder = [outpath,filesep,outname,'_CNN_images_',interpolationType];
	if exist(output_folder,'dir')
		warning(' Removing previous results in %s ...', output_folder);			
		rmdir(output_folder,'s'); 
	end	
    mkdir(output_folder);

    [curr_path, curr_name] = fileparts(FileName);

    outputFilenamePrefix0=[output_folder,filesep,outname];

    for s = 1:numel(scales)
        current_cubeSize = num2str(scales(s));                

        outputFilenamePrefix = [outputFilenamePrefix0,'_s',current_cubeSize,'mm_',outputLabel];
        fprintf(' Current cube size is %s mm.\n',current_cubeSize);

        command = [itkResampleRegionOfInterestExe,' ',FileName,' ',...
            outputFilenamePrefix,' ',t_lower,' ',t_upper,' ',physicalPointsFile,...
            ' ',current_cubeSize,' ',num2str(numberROIvoxels),...
            ' ',num2str(round(ratio*numberRandomTranslations)),...
            ' ',displacementFactorRandomTranslations,...
            ' ',num2str(round(ratio*numberRandomRotations)),...
            ' ',interpolationType,' ',transformType];
        disp(command)

        % run command
        [status, result] = dos(command,'-echo');
        if status~=0
            error(result)
        end
    end
   
toc

%export_to_workspace

