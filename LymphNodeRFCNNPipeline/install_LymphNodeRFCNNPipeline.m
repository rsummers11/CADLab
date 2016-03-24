
pyconvnet_dir='pyconvnet';

%%%%%%%%%%%%%%%%%%% INTSALLL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add matlab sub folders to path
disp(' adding to path:')
addpath(genpath('LymphNodeRFCNNPipeline'));

%% pyconvnet files
if ispc    
    itk_dir = [pwd,filesep,'pyconvnet-0.1.1-win64/pyconvnet'];    
    nifty_prebuild_apps_dir=[pwd,filesep,'niftyreg-git_prebuild_vs10_x64r'];

    %% unzip convnet libraries and apps
%   pyconvnet_zip = 'pyconvnet-0.1.1-win64.zip';    
%     disp(' installing convnet library')
%     filenames = unzip(pyconvnet_zip);
%     convnet_dll = [];
%     for i = 1:numel(filenames)
%         if ~isempty(strfind(filenames{i},'.dll'))
%             convnet_dll = filenames{i};
%         end
%     end
%     if isempty(convnet_dll)
%         error(' could not find pyconvnet dll in %s!',pyconvnet_zip);
%     end
%     %% install pyconvnet lib
%     [itk_dir, convnet_dll_name] = fileparts(convnet_dll);
%     copyfile(convnet_dll,[pyconvnet_dir,filesep,convnet_dll_name,'.pyd'])
else % linux
    nifty_prebuild_apps_dir=[pwd,filesep,'nifty_reg-1.3.9_linux_x64r/reg-apps'];    
    itk_dir = [pwd,filesep,'pyconvnet-0.1.1-linux_x64r'];
    
    %% install pyconvnet lib
%     convnet_lib = rdir([linux_build_dir,'/**/*.so']);
%     convnet_lib = convnet_lib.name;
%     [itk_dir, convnet_lib_name] = fileparts(convnet_lib);
%     itk_dir = fileparts(itk_dir); % root folder of linux build
%     copyfile(convnet_lib,[pyconvnet_dir,filesep,'pyconvnet.so'])    
end
