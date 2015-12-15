function [affineTransFilename, affineResultFilename, nonrigidTransFilename, nonrigidResultFilename] = ...
    niftyreg_aladin_plus_f3d(NiftyRegAppsDir,sourceFile,targetFile,outputPrefix,aladin_params,f3d_params)

    tic
    
    outdir = fileparts(outputPrefix);
    if ~isdir(outdir)
        mkdir(outdir)
    end
    
    reg_aladin_exe=[NiftyRegAppsDir,'/reg_aladin.exe'];
    reg_f3d_exe=[NiftyRegAppsDir,'/reg_f3d.exe'];
        
    % check file formats
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
    
    %% RUN REG_ALADIN (RIGID OR AFFINE REGISTRATION)
    affineTransFilename = [resultPrefix,'_affine.txt'];
    affineResultFilename = [resultPrefix,'_affine.nii.gz'];
    command = [reg_aladin_exe,' -source ',sourceFile,' -target ',targetFile,' -aff ',affineTransFilename,' -result ',affineResultFilename,' ',aladin_params];
    command = strrep(command,'\','/');
    disp(command)
   [status, result_affine] = dos(command,'-echo');
   if status ~= 0
       error(result_affine)
   end

    %% RUN REG_F3D (FAST-FREE-FORM-DEFORMATION)
    nonrigidResultFilename = [resultPrefix,'_nonrigid.nii.gz'];
    nonrigidTransFilename = [resultPrefix,'_nonrigid_CPP.nii'];
    command = [reg_f3d_exe,' -source ',affineResultFilename,' -target ',targetFile,' -cpp ',nonrigidTransFilename,' -result ',nonrigidResultFilename,' ',f3d_params];
    command = strrep(command,'\','/');
    disp(command)
    [status, result_nonrigid] = dos(command,'-echo');
    if status ~= 0
        error(result_nonrigid)
    end    
    
    toc
    
    result_logname = [resultPrefix,'_NIFTYREG_LOG.txt'];
    result = [result_affine, result_nonrigid];
    result = strrep(result,'%','percent');
    fid_LOG = fopen(result_logname,'w');
    fprintf(fid_LOG,result);
    fclose(fid_LOG);
    
    
    
    