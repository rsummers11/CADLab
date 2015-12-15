% calculate blobness for the whole image
function [MaximumBlob] = blobness(DCMArray, daSpacing)

method = 'li';

% sigmas = 4:2:10;% Matthew
% sigmas = 2:1:6; % recommended by Jiamin
% sigmas = 2:1:3;
% sigmas = 3:1:8;% Kevin
sigmas = [2.5:1:7.5];% Kevin

%keep only the voxels of interest
% intTh = [800 1200];
% logicV = DCMArray>intTh(1) & DCMArray<intTh(2);
% logicL = logicV(:);
% clear logicV intTh

holdBLOBS = cell(length(sigmas),1);
BlobIndex = 0;
holdLAMBS  = cell(length(sigmas),1);
fprintf('Beginning Blobness Calculations.\n')

for sigs = sigmas
    tic
    fprintf('Starting sigma:  %2.2f\n',sigs)
    sigma = 2*sigs;
    h=fspecial('gaussian',1*round(sigma+1),sigma);
    Ig = double(sigma^2*imfilter(DCMArray, h));
    
    [ Ix, Iy,  Iz] = gradient(Ig,daSpacing(1),daSpacing(2),daSpacing(3));
    [Ixx,Ixy, Ixz] = gradient(Ix,daSpacing(1),daSpacing(2),daSpacing(3));
    [  ~,Iyy, Iyz] = gradient(Iy,daSpacing(1),daSpacing(2),daSpacing(3));
    [  ~,  ~, Izz] = gradient(Iz,daSpacing(1),daSpacing(2),daSpacing(3));
    clear Ix Iy Iz Ig h
    
    % Let A be the original 3 x 3 x n array
    % Separate out the nine 1 x n row vectors a,b,c,d,e,f,g,h,k
    A = Ixx(:)'; B = Ixy(:)'; C = Ixz(:)';
    E = Iyy(:)'; F = Iyz(:)';
    K = Izz(:)';
    clear Ixx Ixy Ixz Iyx Iyy Iyz Izx Izy Izz
    
    % Find eigenvalues: det([a-x,b,c;d,e-x,f;g,h,k-x]) = 0 in each layer
    a = -1;
    b = A+E+K;
    c = -A.*E - A.*K - E.*K + B.*B + C.*C + F.*F;
    d = A.*E.*K + B.*F.*C + C.*B.*F - C.*E.*C - B.*B.*K - A.*F.*F;
    clear A B C E F K
    
    x = ((3*c./a) - (b.^2./a.^2))/3;
    y = ((2*b.^3./a.^3) - (9*b.*c./a.^2) + (27*d./a))./27;
    z = y.^2/4 + x.^3/27;
    clear a c d
    
    % I,J,k,m,n,p
    I = sqrt(y.^2/4 - z);
    J = -I.^(1/3);
    k = acos(-y./(2*I));
    m = cos(k./3);
    n = sqrt(3).*sin(k./3);
    p = -b./3;
    
    Eigs_New = -1.*[-J.*(m+n) + p;
        -J.*(m-n) + p;
        2*J.*m + p];
    clear J m n p
    
    B = zeros(size(DCMArray));
    
    switch lower(method)
        
        case 'li'
            
            [~,Eigs_NewInds] = sort(abs(Eigs_New),1,'descend');
            Eigs_NewInds = Eigs_NewInds + ones(3,1)*(0:3:(length(Eigs_NewInds(:))-1));
            Eigs_New = Eigs_New(Eigs_NewInds);
            
            lambda1 = reshape(Eigs_New(1,:),size(DCMArray));
            lambda2 = reshape(Eigs_New(2,:),size(DCMArray));
            lambda3 = reshape(Eigs_New(3,:),size(DCMArray));
            
            onlyThese = (lambda1<0) & (lambda2<0) & (lambda3<0);
            B(onlyThese)= (abs(lambda3(onlyThese))).^2./abs(lambda1(onlyThese));
            
            B(DCMArray==0) = 0;
            
            BlobIndex = BlobIndex + 1;
            holdBLOBS{BlobIndex} = B(:);
            
        case 'frangi'
            
            [~,Eigs_NewInds] = sort(abs(Eigs_New),'ascend');
            Eigs_NewInds = Eigs_NewInds + ones(3,1)*(0:3:(length(Eigs_NewInds(:))-1));
            Eigs_New = Eigs_New(Eigs_NewInds);
            
            lambda1 = reshape(Eigs_New(1,:),size(DCMArray));
            lambda2 = reshape(Eigs_New(2,:),size(DCMArray));
            lambda3 = reshape(Eigs_New(3,:),size(DCMArray));
            
            onlyThese = (lambda1<0) & (lambda2<0) & (lambda3<0);
            
            Ra = abs(lambda2)./abs(lambda3);
            S = sqrt(lambda1.^2+lambda2.^2+lambda3.^2);
            alpha = 1;
            gamma = 1;
            
            B = 255.*(1-exp(-Ra.^2./(2*alpha^2))).*(1-exp(-S.^2./(2*gamma^2)));
            B(~onlyThese) = 0;
            
            clear Ra S alpha gamma
            
            B(DCMArray==0) = 0;
            
            BlobIndex = BlobIndex + 1;
            holdBLOBS{BlobIndex} = B(:);
            
        case 'lambdas'
            
            BlobIndex = BlobIndex + 1;
%             holdLAMBS{BlobIndex} = Eigs_New(:,logicL);
            holdLAMBS{BlobIndex} = Eigs_New;
            
            
    end
    
    clear a b c d x y z k m n p
    clear A B C D E F G H K I J
    clear Eigs_New Eigs_NewInds
    clear lambda1 lambda2 lambda3
    clear onlyThese B
%     fprintf('Done with sigma:  %3.2f\n',toc)
    
end

switch lower(method)
    case 'frangi' 
        MaximumBlob = max([holdBLOBS{:}],[],2);
        MaximumBlob = reshape(MaximumBlob,size(DCMArray));
    case 'li'
        MaximumBlob = max([holdBLOBS{:}],[],2);
        MaximumBlob = reshape(MaximumBlob,size(DCMArray));
    case 'lambdas'
        MaximumBlob = holdLAMBS;
end

