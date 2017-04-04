function outname = convertImageSlicesToVolume_hed(inputSlicesDir,volumename,PLANE)

    %% COMMON
    outputRoot = inputSlicesDir;  

    % 
    searchString = '**\*.nii.gz';
    % 
    inSliceTypes = {};
    %inSliceTypes{end+1} = 'sigmoid-dsn1';
    %inSliceTypes{end+1} = 'sigmoid-dsn2';
    %inSliceTypes{end+1} = 'sigmoid-dsn3';
    %inSliceTypes{end+1} = 'sigmoid-dsn4';
    %inSliceTypes{end+1} = 'sigmoid-dsn5';
    inSliceTypes{end+1} = 'sigmoid-fuse'; 

    %% RUN
    for t = 1:numel(inSliceTypes)
        inSliceType = inSliceTypes{t};

        
        outputDir = [outputRoot,filesep,inSliceType,'_VOL'];
        if ~isdir(outputDir)
            mkdir(outputDir)
        end 


        fprintf('sliceType %d of %d: %t\n', t, numel(inSliceTypes), inSliceType);

        [~, patientname] = fileparts(volumename);
        patientname = clearExtension(patientname);
        [~, patient] = fileparts(patientname);

        rslices = rdir([inputSlicesDir,'/*',inSliceType,'*.png'])
        NSlices = numel(rslices);
        if NSlices==0
            error('No slices found!')
        end

        [V, vsiz, vdim, ~, vhdr] = read_nifti_volume(volumename);    
        SLICES = zeros(vsiz,'single');

        if strcmpi(PLANE,'co')
            if vsiz(2) ~= NSlices
                error('Number images and image size do not fit!')
            end            
        elseif strcmpi(PLANE,'sa')
            if vsiz(1) ~= NSlices
                error('Number images and image size do not fit!')
            end                        
        else % ax
            if vsiz(3) ~= NSlices
                error('Number slice images and image size do not fit!')
            end
        end

            %[NX, NY, NC] = size(img);
            %[XI,YI] = meshgrid(1:NX/(vsiz(1)+2):NX,1:NY/(vsiz(2)+2):NY);
            %img = interp2(single(img),XI,YI);
% 
%                 if strcmpi(PLANE,'co')
%                     SLICES(:,s,:) = img';
%                 elseif strcmpi(PLANE,'sa')
%                     SLICES(s,:,:) = fliplr(img)';                
%                 else
%                     SLICES(:,:,s) = fliplr(img');
%                 end

        warning('Ax and Co are the same but transversing in different directions through volume!')
        if strcmpi(PLANE,'co')
            for s = 1:NSlices
                img = single(imread(rslices(s).name))/255.0;                                        
                SLICES(:,s,:) = img';
            end
        elseif strcmpi(PLANE,'sa')
            for s = 1:NSlices
                img = single(imread(rslices(s).name))/255.0;  
                SLICES(s,:,:) = flipud(img');
            end
        elseif strcmpi(PLANE,'ax')
            for ss = 1:NSlices
                s = NSlices-ss+1;
                img = single(imread(rslices(s).name))/255.0;                    
                SLICES(:,:,s) = flipud(rot90(img',2));
            end
        else
            error('No such PLANE %s!',PLANE)
        end                

        %save
        outname = [outputDir,filesep,patientname,'_',inSliceType,'_VOL.nii.gz'];
        write_nifti_volume(SLICES,vdim,outname,vhdr)
    end    