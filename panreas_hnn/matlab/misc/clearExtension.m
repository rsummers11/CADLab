function filename = clearExtension(filename)

    ext = 'xxx';
    while ~isempty(ext)
        [~, ~, ext] = fileparts(filename);
        if numel(ext)<=7 % normally not more than 5 extension characters
            filename = strrep(filename,ext,'');
        else
            ext = [];
        end
    end
    