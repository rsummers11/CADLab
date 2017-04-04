in_pred_dir = '/media/rothhr/SSD/NIHpancreas/hnn_exp_fold4_predictions'
pred_search = 'stage2/**/*_meanmaxAxCoSa.nii.gz'

in_gt_dir = '/media/rothhr/SSD/NIHpancreas/hnn_exp_fold4_predictions';
gt_search = 'stage1/largestObjBB/**_Pancreas_cropped.nii.gz';

p_threshs = 0:0.05:1

%% RUN EVAL %%
addpath(genpath([pwd,filesep,'matlab']))

r = rdir([in_pred_dir,'/**/',pred_search]);


for i = 1:numel(r)
    PROB_MAP = r(i).name;
    [~, patient] = fileparts(PROB_MAP);
    patient = patient(1:find(patient=='_',1,'first')-1);
    GRTR_SEG = rdir([in_gt_dir,'/**/',patient,'/**/',gt_search]);
    GRTR_SEG = GRTR_SEG.name;
   
    [result_string{i}, Dice_coef(i,:),Jaccard_sim(i,:),precision(i,:),recall(i,:),TP(i,:),TPR(i,:),FPR(i,:),seg_vol,seg_vdim,seg_hdr]...
             = pancreas_evaluation(PROB_MAP,GRTR_SEG,p_threshs);
end

%% plot
plot(p_threshs,mean(Dice_coef,1))
grid
ylim([0 1])
xlabel('prob')
ylabel('Dice score')
shg