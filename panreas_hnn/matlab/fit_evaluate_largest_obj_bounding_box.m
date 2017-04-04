%in_prob_dir = '/home/rothhr/Data/Pancreas/MICCAI2016/ExtraBodyCroppedData/img_ax_TEST_hed_pancreas_mask_iter_98000_globalweight/sigmoid-fuse_VOL';
%prob_suffix = '_maxAX-CO-SA.nii.gz';
%gt_dir = '/home/rothhr/Data/Pancreas/MICCAI2016/ExtraBodyCroppedData/label';
%outdir = '/home/rothhr/Data/Pancreas/MICCAI2016/ExtraBodyCroppedData/LargestObjectBoundingBox'

function [outname, vol_diffs, recall, Dice_coef] = fit_evaluate_largest_obj_bounding_box(in_prob_name,gt_name,outdir)

%% COMMON
PLOT = true;
%PLOT = false;

SAVE = true
p_threshs = [0.5];
r_threshs = [1];


%% RUN LOOP
cd(fileparts(mfilename('fullpath')))
addpath('..')
addpath('functions')

N = 1;
N_p_threshs = numel(p_threshs)
N_r_threshs = numel(r_threshs)
Ntotal = N*N_p_threshs*N_r_threshs

image_volumes = NaN*ones(N,N_p_threshs,N_r_threshs);
bb_volumes = NaN*ones(N,N_p_threshs,N_r_threshs);
vol_diffs = NaN*ones(N,N_p_threshs,N_r_threshs); 
Dice_coef = NaN*ones(N,N_p_threshs,N_r_threshs);
Jaccard_sim = NaN*ones(N,N_p_threshs,N_r_threshs);
precision = NaN*ones(N,N_p_threshs,N_r_threshs);
recall = NaN*ones(N,N_p_threshs,N_r_threshs);
TP = NaN*ones(N,N_p_threshs,N_r_threshs);
TPR = NaN*ones(N,N_p_threshs,N_r_threshs);
FPR = NaN*ones(N,N_p_threshs,N_r_threshs);
nr_GT_Voxels = NaN*ones(N,N_p_threshs,N_r_threshs); 
nr_GT_Voxels_within_BB = NaN*ones(N,N_p_threshs,N_r_threshs);


nr_GT_Voxels = NaN*ones(N,1);    
nr_GT_Voxels_within_BB = NaN*ones(N,1);

Patients = cell(N,1);

    
[~, patient] = fileparts(in_prob_name);
patient = clearExtension(patient);
progress = sprintf('Patient: %s', patient);
disp(repmat('=',1,100))
disp(progress)
disp(repmat('=',1,100))

i = 1;
Patients{i} = patient;

%% LOAD DATA
[P psize pdim pfilename phdr] = read_nifti_volume(in_prob_name);
if ~isempty(gt_name)
    [GT gtsize gtdim gtfilename gthdr] = read_nifti_volume(gt_name);
else
    GT = [];
    gtsize = psize;
    gtdim = pdim;
end
if ~compareImages(gtsize,gtdim,psize,pdim)
    error('GT and P do not fit!')
end
vx_vol = prod(gtdim);

%% Threshold and find largest connected components
for p = 1:N_p_threshs
    for r = 1:N_r_threshs
        OBJ = findLargestConnComponent(P>=p_threshs(p),r_threshs(r),6);
        [x, y, z] = findnd(OBJ>0);

        BB = zeros(gtsize);
        BB(min(x):max(x),min(y):max(y),min(z):max(z)) = 1;

        image_volumes(i,p,r) = prod(gtsize)*vx_vol;
        bb_volumes(i,p,r) = sum(BB(:))*vx_vol;

       
        %% CHECK IF GT IS WITHIN BOUNDING BOX
        if ~isempty(GT)
            [result_string, Dice_coef(i,p,r),Jaccard_sim(i,p,r),precision(i,p,r),recall(i,p,r),TP(i,p,r),TPR(i,p,r),FPR(i,p,r),SEG_VOL,~,seg_hdr] = ...
                pancreas_evaluation(BB,GT,0.5,false,false);            
            
            
            CHECK = BB & GT;
            nr_GT_Voxels(i,p,r) = sum(GT(:));    
            nr_GT_Voxels_within_BB(i,p,r) = sum(CHECK(:));    
        end

        %% SAVE
        outprefix = [outdir,filesep,patient];
        outname = [outprefix,'_largestObjectBB_p',num2str(p_threshs(p)),'_r',num2str(r_threshs(r)),'.nii.gz'];
        if SAVE
            write_nifti_volume(BB,pdim,outname,phdr)
        end

        %% PLOT
        if PLOT                
            [y, x, z] = findnd(true(gtsize));
            Y = [x*gtdim(1), y*gtdim(2), z*gtdim(3)]; % mmx                

            figure_fullscreen;
            hold on
            plot3(min(Y(:,1)),min(Y(:,2)),min(Y(:,3)),'o')
            plot3(max(Y(:,1)),max(Y(:,2)),max(Y(:,3)),'o')
            if ~isempty(GT)
                showSurface(GT>0,4,gtdim,1,0.5,'colon','none',true,false);
            end
            showSurface(P>=.5,4,gtdim,1,0.1,'cyan','none',true,false);

            showSurface(BB>0,4,gtdim,1,0.05,'red','none',true,false);
            htitle = title(progress);
            set(htitle,'Interpreter','None');
            axis auto
            zoom out
            if ~isempty(GT)
                hlegend = legend('bb_corner1','bb_corner2','ground truth','prediction','bounding box');
            else
                hlegend = legend('bb_corner1','bb_corner2','prediction','bounding box');
            end
            set(hlegend,'Interpreter','None');
            shg

            %% SAVE FIGURE
            saveas(gca,[outprefix,'_largestObjectBB_p',num2str(p_threshs(p)),'_r',num2str(r_threshs(r)),'.fig'])
            %saveFigure([outprefix,'_SVD.png'])        
        end
    end
end


%% show differences
if N_p_threshs==1 && N_r_threshs==1
    for i = 1:N
        vol_diff = 100-100*bb_volumes(i)/image_volumes(i);
        fprintf('%d. %s:\tabs. vol. reduction: %.2f\t|| recall: %.2f\n', i, Patients{i}, vol_diff, 100*recall(i) );    
    end
end

for p = 1:N_p_threshs
    for r = 1:N_r_threshs
        vol_diffs(:,p,r) = 100-100*bb_volumes(:,p,r)./image_volumes(:,p,r);
        disp('===================================================================')
        fprintf('   Bounding box ( p = %.1f, r = %.1f )   \n', p_threshs(p), r_threshs(r)  );
        disp('===================================================================')
        fprintf('BB mean abs. vol. reduction: %.2f, [%.2f,...,%.2f]\n', mean(vol_diffs(:,p,r)), min(vol_diffs(:,p,r)), max(vol_diffs(:,p,r)) );
        if ~isempty(GT)
            fprintf('BB mean recall: %.2f, [%.2f,...,%.2f]\n', 100*mean(recall(:,p,r)), 100*min(recall(:,p,r)), 100*max(recall(:,p,r)) );
            fprintf('BB mean Dice_coef: %.2f, [%.2f,...,%.2f]\n', 100*mean(Dice_coef(:,p,r)), 100*min(Dice_coef(:,p,r)), 100*max(Dice_coef(:,p,r)) );
        end
    end
end

save([outprefix,'_largestObjectBB_p',num2str(p_threshs(p)),'_r',num2str(r_threshs(r)),'.mat'])


