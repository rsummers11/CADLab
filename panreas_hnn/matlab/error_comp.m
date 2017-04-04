function [Dice,Jaccard,precision,recall,tp,rtp,rfp,rfn] = error_comp(m,o)

    %%compute Dice and Jaccard similarity measures

    % Dice =  2* (A intersect B)/ length(A) + length(B)
    %Jaccard Similarity = length(A interect B)/ length(A+B)

    if any(size(m)-size(o))
        error('  error_comp: images are not the same size!');
    end

    % make sure images are binary
    m = m>0;
    o = o>0;
    
    % outlier cases
    if sum(m(:))==0 && sum(o(:))==0 % ground truth and object both zero
        warning('Ground truth and object are zero!')
        Dice = 1.0;
        Jaccard = 1.0;
        precision = 1.0;
        recall = 1.0;
        tp = numel(m); % even though thats not really the definition....
        rtp = 1.0;
        rfp = 0.0;
        rfn = 0.0;
        return
    elseif sum(m(:))==0 && any(o(:))
        warning('Ground truth is zero (but object has values)!')
        % But still compute measures!
    end
    
%     N_vx = numel(gt_vol);
%     
%     GT_idx = find(gt_vol(:)>0);
%     SEG_idx = find(seg_vol(:)>0);
% 
%     TP = length(intersect(GT_idx,SEG_idx));
% 
%     N_GT = length(GT_idx); %GT number of pixels
%     N_SEG = length(SEG_idx); %Number of segmented voxels
%     TN = length(find(gt_vol(:)~=1)); %True Negatives
% 
%     Dice_coef = 2*TP/(N_GT+N_SEG+eps);
%     Jaccard_sim = TP/(N_GT+N_SEG-TP +eps);
% 
    
    
% function [Jaccard,Dice,rfp,rfn]=sevaluate(m,o)
%Copyright (c) 2010, M. A Balafar All rights reserved 
%from: http://www.mathworks.com/matlabcentral/fileexchange/29737-segmentation-evaluatation/content/sevaluate.m
% gets label matrix for one tissue in segmented and ground truth 
% and returns the similarity indices
% m is a tissue in gold truth (mask)
% o is the same tissue in segmented image (object)
% rfp false positive ratio
% rfn false negative ratio
m=m(:);
o=o(:);
common=sum(m & o); 
tp = common;
union=sum(m | o); 
cm=sum(m); % the number of voxels in m
co=sum(o); % the number of voxels in o
fp = co-tp;
Jaccard=common/(union+eps);
Dice=(2*common)/(cm+co+eps);
rfp=fp/(co+eps);
rfn=(cm-common)/(cm+eps);    

rtp = tp/(cm+eps);
%     FPR = (N_SEG-TP)/(N_SEG+eps);

precision = tp/(co+eps);
recall = tp/(cm+eps);
