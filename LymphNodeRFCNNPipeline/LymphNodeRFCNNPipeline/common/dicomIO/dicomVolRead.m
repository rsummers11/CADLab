function [V,vsize,vdim,info,affineparams] = dicomVolRead(pathname)

disp('This function fits with ReadAnalyze!')

%DEFAULT_SLICE_THICKNESS = 3;

if nargin < 1
  pathname = uigetdir(pwd,'Choose dicom directory') ;

  if pathname ==0
    error([ 'No directory selected.' ])
  end 
end

files = dir(pathname);
disp(['Reading from ',pathname,' ...'])
dfiles = {};
locs = [];
%loc_field: 0 = unknown, 1 = SliceLocation, 2 = ImagePositionPatient(3), 3 = filename
loc_field = 0;

h = waitbar(0,'Please wait...');

Nfiles = length(files);
slice_count = 0;
for i = 1:Nfiles;
    try
        if ~isequal(files(i).name,'DIRFILE')
            info = dicominfo([pathname filesep files(i).name]);
            if ~isequal(info.Modality,'RTSTRUCT')
                switch loc_field
                    case 0
                        if isfield(info,'SliceLocation')
                            loc_field = 1;
                            locs = [locs info.SliceLocation];
                            dfiles = {dfiles{:},[pathname filesep files(i).name]};
                        elseif isfield(info,'ImagePositionPatient')
                            loc_field = 2;
                            locs = [locs info.ImagePositionPatient(3)];
                            dfiles = {dfiles{:},[pathname filesep files(i).name]};
                        else
                            loc_field = 3;
                            warning('Dicom header did not contain slice location/position information, slices ordered by file name...');
                            locs = [locs i];
                            dfiles = {dfiles{:},[pathname filesep files(i).name]};
                        end
                    case 1
                        if isfield(info,'SliceLocation')
                            locs = [locs info.SliceLocation];
                            dfiles = {dfiles{:},[pathname filesep files(i).name]};
                        else
                            warning(['SliceLocation missing for slice ' files(i).name ', slice not loaded into volume.']);
                        end
                    case 2
                        if isfield(info,'ImagePositionPatient')
                            locs = [locs info.ImagePositionPatient(3)];
                            dfiles = {dfiles{:},[pathname filesep files(i).name]};
                        else
                            warning(['ImagePositionPatient missing for slice ' files(i).name ', slice not loaded into volume.']);
                        end
                    case 3
                        locs = [locs i];
                        dfiles = {dfiles{:},[pathname filesep files(i).name]};
                end
            end
        end
        slice_count = slice_count + 1;
        waitbar(i/Nfiles,h,sprintf('read slice %d',slice_count)) 
    catch
    end
end


if length(dfiles) == 0
    error('No dicom files found');
end

[slocs ilocs] = sort(locs);
sorted_files = {};
for i = 1:length(ilocs)
    sorted_files{i} = dfiles{ilocs(i)};
end

vsize(3) = length(dfiles);

%get info for first slice
info = dicominfo(sorted_files{1});

%get no of pixels in slice
vsize(1) = info.Width;
vsize(2) = info.Height;

disp(['No. of slices: ' num2str(vsize(3)) ', Pixels per slice: ' num2str(vsize(1)) ' x ' num2str(vsize(2))]);

%get voxel dimesions
vdim(1) = info.PixelSpacing(1);
vdim(2) = info.PixelSpacing(2);


if loc_field == 1 || loc_field == 2
    diffs = slocs(2:end)-slocs(1:end-1);
    if any(diffs~=diffs(1))
        warning('Slice spacing is not constant');
    end
    vdim(3) = diffs(1);
elseif isfield(info,'SliceThickness') && ~isempty(info.SliceThickness)
    if isfield(info,'SpacingBetweenSlices')
        vdim(3) = info.SliceThickness + info.SpacingBetweenSlices;
    else
        vdim(3) = info.SliceThickness;
    end
else
    warning(['No slice thinkness information, using default slice thickness of ' num2str(DEFAULT_SLICE_THICKNESS)]);
    vdim(3) = DEFAULT_SLICE_THICKNESS;
end
    



disp(['Voxel dimensions: ' num2str(vdim(1)) ' x ' num2str(vdim(2)) ' x ' num2str(vdim(3))]);

switch info.BitsAllocated
    case 16
        
        disp('DICOM files are 16 bit, reading as shorts...');
        
        V = zeros(vsize,'int16');
        affineparams = zeros(8,1);
        
        for z = 1:vsize(3)
            
            %disp(['Reading slice ' num2str(z) '...']);
            
            %read slice
            V(:,:,z) = int16(dicomread(sorted_files{z}))';

            info = dicominfo(sorted_files{z});
            affineparams(1) = affineparams(1) + info.ImagePositionPatient(1);
            affineparams(2) = affineparams(2) + info.ImagePositionPatient(2);
            affineparams(3) = affineparams(3) + info.ImagePositionPatient(3);
        end
        
        affineparams = affineparams./vsize(3);

    otherwise
        error(['Can only load 16 bit DICOM files, data is ' num2str(info.BitsAllocated) ' bits.']);
end

close(h)
