pyconvnet_zip = 'pyconvnet-0.1.1-win64.zip';
convnent_dir='pyconvnet';
nifty_prebuild_apps_dir=[pwd,filesep,'niftyreg-git_prebuild_vs10_x64r'];

%%%%%%%%%%%%%%%%%%% INTSALLL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add matlab sub folders to path
disp(' adding to path:')
subdirs = genpath('LymphNodeRFCNNPipeline');
seps = [0 strfind(subdirs,';')]; % start from beginning
for i = 1:numel(seps)-1
   subdir = subdirs(seps(i)+1:seps(i+1)-1);
   disp([pwd,filesep,subdir])
   addpath([pwd,filesep,subdir]);
end

%% unzip convnet libraries and apps
disp(' installing convnet library')
filenames = unzip(pyconvnet_zip);
convnet_dll = [];
for i = 1:numel(filenames)
    if ~isempty(strfind(filenames{i},'.dll'))
        convnet_dll = filenames{i};
    end
end
if isempty(convnet_dll)
    error(' could not find pyconvnet dll in %s!',pyconvnet_zip);
end
%% install pyconvnet lib
[itk_dir, convnet_dll_name] = fileparts(convnet_dll);
copyfile(convnet_dll,[convnent_dir,filesep,convnet_dll_name,'.pyd'])
