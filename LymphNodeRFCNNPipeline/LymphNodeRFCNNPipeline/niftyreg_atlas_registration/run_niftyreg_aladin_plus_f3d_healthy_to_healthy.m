%% Healthy is target (with leave-one-out healthy atlases: 4 vs 1)
atlas_rootdir  = '/home/rothhr/Data/Spine/SpineSegmentation/CompData/Healthy_Atlas';
target_rootdir = '/home/rothhr/Data/Spine/SpineSegmentation/CompData/Healthy_Atlas';
out_rootdir = '/home/rothhr/Data/Spine/SpineSegmentation/CompData/Healthy_NiftyReg_symaffine_f3dGPU';

case_numbers = [18, 19, 20, 22, 23];

%% Vertebra labels
vertebrae_labels = [];
vertebrae_labels{1}='L1';
vertebrae_labels{2}='L2';
vertebrae_labels{3}='L3';
vertebrae_labels{4}='L4';
vertebrae_labels{5}='L5';
vertebrae_labels{6}='T1';
vertebrae_labels{7}='T10';
vertebrae_labels{8}='T11';
vertebrae_labels{9}='T12';
vertebrae_labels{10}='T2';
vertebrae_labels{11}='T3';
vertebrae_labels{12}='T4';
vertebrae_labels{13}='T5';
vertebrae_labels{14}='T6';
vertebrae_labels{15}='T7';
vertebrae_labels{16}='T8';
vertebrae_labels{17}='T9';

%% RUN
case_number_idx = 1:numel(case_numbers);

t0 = tic;

%for k = 1:3
for k = 4:5
    
    target_number = case_numbers(case_number_idx==k);
    atlas_numbers = case_numbers(case_number_idx~=k);

    target_name = ['case',num2str(target_number)];

    atlas_labels = cell(numel(atlas_numbers),1);
    for aa = 1:numel(atlas_numbers)
        atlas_labels{aa}=['case',num2str(atlas_numbers(aa))];
    end

    %% RUN REGISTRATIONS
    out_dir = [out_rootdir,filesep,target_name];

    t1 = tic;
    mkdir(out_dir)
    N_vertebrae = numel(vertebrae_labels);
    N_atlases   = numel(atlas_labels);
    N_regs = N_vertebrae*N_atlases;
    reg_count = 0;
    hw = waitbar(0,'Please wait...');
    for v = 1:N_vertebrae
        for a = 1:N_atlases
            reg_count = reg_count + 1;
            progress_str = sprintf('%d. registration %d of %d: atlas %s to vertebra %s',k,reg_count,N_regs,atlas_labels{a},vertebrae_labels{v});
            disp(progress_str)
            waitbar((reg_count-0.5)/N_regs,hw,progress_str);
            % get filenames
            sourceFile = [atlas_rootdir, filesep,vertebrae_labels{v},filesep,vertebrae_labels{v},'_',atlas_labels{a},'.mhd'];
            labelSourceFile = [atlas_rootdir, filesep,vertebrae_labels{v},filesep,vertebrae_labels{v},'_',atlas_labels{a},'_label.mhd'];
            multiLabelSourceFile = [atlas_rootdir, filesep,vertebrae_labels{v},filesep,vertebrae_labels{v},'_',atlas_labels{a},'_mlabel.mhd'];
            targetFile = [target_rootdir,filesep,vertebrae_labels{v},filesep,vertebrae_labels{v},'_',target_name,'.mhd'];
            outputPrefix = [out_dir,filesep,vertebrae_labels{v}];

            if ~exist(sourceFile,'file')
                error('%s does not exist!',sourceFile);
            end
            if ~exist(labelSourceFile,'file')
                error('%s does not exist!',labelSourceFile);
            end
            if ~exist(multiLabelSourceFile,'file')
                error('%s does not exist!',multiLabelSourceFile);
            end        
            if ~exist(targetFile,'file')
                error('%s does not exist!',targetFile);
            end       

            % run registration
            mkdir(outputPrefix);

           [affineTransFilename, affineResultFilename, nonrigidTransFilename, nonrigidResultFilename] = ...
               niftyreg_aladin_plus_f3d(sourceFile,targetFile,outputPrefix);

           [affineLabelResultFilename, nonrigidLabelResultFilename] = ...
               niftyreg_resample(labelSourceFile,targetFile,affineTransFilename,nonrigidTransFilename,outputPrefix);

           [affineMultiLabelResultFilename, nonrigidMultiLabelResultFilename] = ...
               niftyreg_resample(multiLabelSourceFile,targetFile,affineTransFilename,nonrigidTransFilename,outputPrefix);        
        end
    end
    close(hw)
    toc(t1)
    disp('####################################################################')
end % k = 1:numel(case numbers)

disp('####################################################################')
disp('############################ FINISHED ##############################')
toc(t0)

%% convert image formats
%convertImageDir(out_dir,'.nii.gz',out_dir,'.mhd');

