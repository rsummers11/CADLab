function [result_string, Dice_coef,Jaccard_sim,precision,recall,TP,TPR,FPR,seg_vol,seg_vdim,seg_hdr]...
             = pancreas_evaluation(PROB_MAP,GRTR_SEG,p_threshs,LARGEST_OBJ_ONLY,FILL_AXIAL_HOLES,VERBOSE)

    plot_results = false;
    
    if ~exist('LARGEST_OBJ_ONLY','var')
        LARGEST_OBJ_ONLY = false;
    end
    if ~exist('FILL_AXIAL_HOLES','var')
        FILL_AXIAL_HOLES = false;
    end
    if ~exist('VERBOSE','var')
        VERBOSE = true;
    end    

    if ischar(PROB_MAP)
        PROB_MAP = cleanFileName(PROB_MAP);
        PROB_MAP = read_nifti_volume(PROB_MAP);
    end
    if ischar(GRTR_SEG)
        GRTR_SEG = cleanFileName(GRTR_SEG);
        [GRTR_SEG, ~, seg_vdim, ~, seg_hdr] = read_nifti_volume(GRTR_SEG);
    else
        %seg_vdim = [1.0 1.0 1.0];
        seg_vdim = [];
        seg_hdr = [];
    end     
    
    if any(isnan(p_threshs))
        p_threshs = 0;
    end
    
    % Scale PROB_MAP if nescessary
    PROB_MAP = double(PROB_MAP);
    if max(PROB_MAP(:))>1
        warning('Prob map > 1.0 -> divide by 255... (max value: %g)', max(PROB_MAP(:)));
        PROB_MAP = PROB_MAP./255.0;
    end
    
    N_p_threshs = numel(p_threshs);
    %fprintf(' evaluating probability map at %d probs...\n', N_p_threshs)
    
    Dice_coef   = NaN*ones(1,N_p_threshs);
    Jaccard_sim = NaN*ones(1,N_p_threshs);
    precision   = NaN*ones(1,N_p_threshs);
    recall      = NaN*ones(1,N_p_threshs);
    TP = NaN*ones(1,N_p_threshs);
    TPR = NaN*ones(1,N_p_threshs);
    FPR = NaN*ones(1,N_p_threshs);
    for i = 1:N_p_threshs
        if VERBOSE
            fprintf('    p = %g (%d of %d)...\n', p_threshs(i),i,N_p_threshs);
        end
        seg_vol = PROB_MAP>=p_threshs(i);        
        if FILL_AXIAL_HOLES
            disp(' fill axial holes...')
            for z = 1:size(seg_vol,3)
                seg_vol(:,:,z) = imfill(seg_vol(:,:,z),'holes');
            end
            seg_vol = closeColon(seg_vol,5);
        end
        if LARGEST_OBJ_ONLY
            % clear volume boundary
            seg_vol(:,:,1) = 0; seg_vol(:,:,end) = 0;
            seg_vol(:,1,:) = 0; seg_vol(:,end,:) = 0;
            seg_vol(1,:,:) = 0; seg_vol(end,:,:) = 0;            
            seg_vol = findLargestConnComponent(seg_vol,1);
            seg_vol = closeColon(seg_vol,3); 
            seg_vol = imfill(seg_vol,'holes');
        end
        [Dice_coef(i),Jaccard_sim(i),precision(i),recall(i),TP(i),TPR(i),FPR(i)] = error_comp(GRTR_SEG,seg_vol);
    end
    
    %% print results
    result_string = [];
    result_string = [result_string, sprintf('max_Dice_coef, %g, ', max(Dice_coef))];
    result_string = [result_string, sprintf('max_Jaccard_sim, %g, ', max(Jaccard_sim))];
    result_string = [result_string, sprintf('max_precision, %g, ', max(precision))];
    result_string = [result_string, sprintf('max_recall, %g, ', max(recall))];
    result_string = [result_string, sprintf('mean_TP, %g, ', mean(TP))];
    result_string = [result_string, sprintf('mean_TPR, %g, ', mean(TPR))];
    result_string = [result_string, sprintf('mean_FPR, %g', mean(FPR))];
    result_string = [result_string, sprintf('\n')];
    if VERBOSE
        disp(result_string)
    end
    
    %% plot results
    if plot_results
        figure_fullscreen;
        subplot(1,3,1)
        plot(p_threshs,Dice_coef,'b-*')
        hold on
        plot(p_threshs,Jaccard_sim,'g-*')
        xlim([0 1])
        ylim([0 1])    
        legend('Dice_coef','Jaccard_sim')    
        xlabel('probability')
        ylabel('value')    
        axis square
        grid on

        subplot(1,3,2)
        plot(recall,precision,'b-*')
        hold on
        for i = 1:N_p_threshs
            text(recall(i),precision(i),sprintf('p = %g',p_threshs(i)));
        end
        xlim([0 1])
        ylim([0 1])
        xlabel('recall')
        ylabel('precision')
        axis square
        grid on

        subplot(1,3,3)
        plot(FPR,TPR,'b-*')
        hold on
        for i = 1:N_p_threshs
            text(FPR(i),TPR(i),sprintf('p = %g',p_threshs(i)));
        end
        xlim([0 1])
        ylim([0 1])
        xlabel('FPR')
        ylabel('TPR')
        axis square
        grid on

        shg
    end
    