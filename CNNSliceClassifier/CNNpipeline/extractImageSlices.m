function slices_outprefix = extractImageSlices(imageFilename,outputRootDir,opt)

[~, patientname] = fileparts(imageFilename);
patientname = strrep(patientname,'.nii','');
slices_outprefix = [outputRootDir,filesep,patientname];

if isfield(opt,'corrValue') && ~isempty(opt.corrValue)
    V = read_nifti_volume(imageFilename);
    if min(V(:))>=0
        warning(' patient %s is of type 2 (dual energy CT)...', imageFilename);
        opt.minWin=opt.minWin + opt.corrValue;
        opt.maxWin=opt.maxWin + opt.corrValue;
    end
end

if ~isdir(outputRootDir)
    mkdir(outputRootDir)
end

command = [opt.itkExtractImageSlices_exe, ' ', ...
           imageFilename, ' ', ...
           slices_outprefix, ' ', ...
           num2str(opt.minWin), ' ', ...
           num2str(opt.maxWin), ' ', ...
           num2str(opt.outputImageSize), ' ', ...
           opt.outExtension
           ];

%disp(command)
[status, result] = system(command,'-echo');
if status ~= 0 
    error(result)
end
    