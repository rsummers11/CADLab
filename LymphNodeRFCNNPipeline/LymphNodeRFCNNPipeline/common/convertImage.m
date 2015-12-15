function convertImage(inFile,outFile)
    %outDir: if [], same as directory of input images
    
    if ispc
        itkReadWriteImage_exe = '"D:\HolgerRoth\projectsNIH_builds\spine_cpp\vs10_x64r\Release\itkReadWriteImage.exe"';
    elseif isunix
        itkReadWriteImage_exe = '/home/rothhr/Code/projectsNIH_builds/nihLymphNodeApps/release/itkReadWriteImage';
    end

   
    command = [itkReadWriteImage_exe,' ',inFile,' ',outFile];
    %% run command
    disp(command);
    [status, result] = system(command,'-echo');
    if status ~= 0
        error(result);
    end 
    