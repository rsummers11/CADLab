%% export_to_workspace
w = who;
for i = 1:numel(w)
    assignin('base',w{i},eval(w{i}))
end 
