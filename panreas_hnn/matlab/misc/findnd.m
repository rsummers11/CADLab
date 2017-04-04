function [idx, v, z, t] = findnd(A)

IDX = find(A);
L = size(A);
[x{1:length(L)}] = ind2sub(L,IDX);

idx = zeros(length(x{1}),length(x));
for i = 1:length(x)
    idx(:,i) = x{i};
end

if nargout==2
    v = A(IDX);
end

if nargout==3
    v = idx(:,2);
    z = idx(:,3);
    idx = idx(:,1);
end

if nargout==4
    v = idx(:,2);
    z = idx(:,3);
    t = idx(:,4);
    idx = idx(:,1);
end