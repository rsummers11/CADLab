function [affineResultFilename, nonrigidResultFilename] = ...
    niftyreg_resample(NiftyRegAppsDir,sourceFile,targetFile,affineTransFilename,nonrigidTransFilename,outputPrefix)

     %Interpolation order (0, 1, 3, 4)[3] (0=NN, 1=LIN; 3=CUB, 4=SINC)
     inter=0;
        
    %% RUN    
    inter_str = [' -inter ',num2str(inter)];

    tic
    reg_resample_exe=[NiftyRegAppsDir,'/reg_resample.exe'];
    
    % check ifle formats
    [~, ~, sourceExt] = fileparts(sourceFile);
    if ~strcmpi(sourceExt,'.nii') && ~strcmpi(sourceExt,'.gz')
        old_sourceFile = sourceFile;
        sourceFile = strrep(sourceFile,sourceExt,'.nii.gz');
        if ~exist(sourceFile,'file')
            convertImage(old_sourceFile,sourceFile)
        end        
    end
    [~, ~, targetExt] = fileparts(targetFile);
    if ~strcmpi(targetExt,'.nii') && ~strcmpi(targetExt,'.gz')
        old_targetFile = targetFile;
        targetFile = strrep(targetFile,targetExt,'.nii.gz');
        if ~exist(targetFile,'file')
            convertImage(old_targetFile,targetFile)
        end        
    end       
    
    % define output filenames    
    [~, sourceName] = fileparts(sourceFile);
    if any(strfind(sourceName,'.')) % in case of .gz files
     sourceName = sourceName(1:strfind(sourceName,'.')-1);
    end
    [~, targetName] = fileparts(targetFile);
    if any(strfind(targetName,'.')) % in case of .gz files
     targetName = targetName(1:strfind(targetName,'.')-1);
    end    
    if isdir(outputPrefix)
     outputPrefix = [outputPrefix,filesep];
    else
     outputPrefix = [outputPrefix,'_'];
    end

    resultPrefix = [outputPrefix,sourceName,'_TO_',targetName];
    
    %% RESAMPLE AFFINE
    affineResultFilename = [resultPrefix,'_affine.nii.gz'];
    command = [reg_resample_exe,' -source ',sourceFile,' -target ',targetFile,' -aff ',affineTransFilename,' -result ',affineResultFilename,inter_str];
    [status, result_affine] = system(command,'-echo');
    if status ~= 0
        error(result_affine)
    end

    %% RESAMPLE AFFINE TO NONRIGID
    nonrigidResultFilename = [resultPrefix,'_nonrigid.nii.gz'];
    command = [reg_resample_exe,' -source ',affineResultFilename,' -target ',targetFile,' -cpp ',nonrigidTransFilename,' -result ',nonrigidResultFilename,inter_str];
    [status, result_nonrigid] = system(command,'-echo');
    if status ~= 0
        error(result_nonrigid)
    end    
    
    toc
    
    result_logname = [resultPrefix,'_RESAMPLE_LOG.txt'];
    result = [result_affine, result_nonrigid];
    result = strrep(result,'%','percent');
    fid_LOG = fopen(result_logname,'w');
    fprintf(fid_LOG,result);
    fclose(fid_LOG);
    
    