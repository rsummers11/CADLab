function [hdr, filetype, fileprefix, machine] = read_nifti_hdr(filename)

    filename = strrep(filename,'"','');
    filename = strrep(filename,' ','');

    [hdr, filetype, fileprefix, machine] = load_nii_hdr(filename);
    