%% INPUT
inputDir = ''; % Please specify!
outputDir = ''; % Please specify!

%% PARAMS
  itkRandomlyDeformImages2D_exe = 'itkApps_build/release/itkRandomlyDeformImages2D';
  
  searchString = '.png';
  outSize = 256;
  Ntranslations = 3;
  Nrotations = 3;
  Nnonrigiddeforms = 3;
  translation_max = 10.0; % [mm]
  rotation_max = 2.5; % [mm]
  Nnonrigid_points = 5;
  siffness = 1e-6;

%V_nonrigid_deform_max = [1:1:100]; % [mm]  
V_nonrigid_deform_max = 5; % [mm]  

%% RUN
hw = waitbar(0);
for n = 1:numel(V_nonrigid_deform_max)
    waitbar(n/numel(V_nonrigid_deform_max),hw,sprintf('%d of %d', n, numel(V_nonrigid_deform_max)));

    nonrigid_deform_max = V_nonrigid_deform_max(n);
    outputPrefix = [outputDir,filesep,sprintf('def%03d_',nonrigid_deform_max)];
    
    command = [itkRandomlyDeformImages2D_exe, ' ', ...
               inputDir, ' ', ...
               searchString, ' ', ...
               outputPrefix, ' ', ...
               num2str(outSize), ' ', ...
               num2str(Ntranslations), ' ', ...
               num2str(Nrotations), ' ', ...
               num2str(Nnonrigiddeforms), ' ', ...
               num2str(translation_max), ' ', ...
               num2str(rotation_max), ' ', ...
               num2str(Nnonrigid_points), ' ', ...
               num2str(nonrigid_deform_max), ' ', ...
               num2str(siffness)];

    %disp(command)
    [status, result] = dos(command,'-echo');
    if status ~= 0 
        error(result)
    end
end
close(hw)

    