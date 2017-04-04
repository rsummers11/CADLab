function filename = cleanFileName(filename)

    filename = strrep(filename,'"','');
    %filename = strrep(filename,' ','');
    
    filename = strrep(filename,'\',filesep);
    filename = strrep(filename,'/',filesep);
    filename = strrep(filename,'\\',filesep);
    