function dicom2volume(dicom2volume_exe,dicom_dir,output_file)    
    
    
    mkdir(fileparts(output_file));
    [status, result] = dos([dicom2volume_exe,' ',dicom_dir,' ',output_file],'-echo');
    if status~=0
        error(result)
    end
