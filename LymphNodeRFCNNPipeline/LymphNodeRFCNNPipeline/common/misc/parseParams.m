% parseParams (script)

    names = fieldnames(opt);
    for i = 1:numel(names)
        eval([names{i},' = opt.',names{i}]);
    end
