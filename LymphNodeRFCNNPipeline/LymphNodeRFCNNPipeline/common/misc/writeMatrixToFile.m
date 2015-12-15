function writeMatrixToFile(T,fileStr)
    % WRITEMATRIXTOFILE(T,fileStr)
% 
%     fid = fopen(fileStr, 'w');
% 
%     h = waitbar(0,['writing to ',fileStr])
%     for i=1:size(T,1)
%         for j=1:size(T,2)
%             fprintf(fid, '%f ', T(i,j));
%         end
%         fprintf(fid,'\n');
%         waitbar(i/size(T,1))
%     end
%     disp( sprintf('Wrote %d lines',i) );
%     close(h)
% 
%     fclose(fid);

    t1 = tic;

    if isa(T,'integer')
        dlmwrite(fileStr, T,' ');%, 'precision', '%.6f','delimiter','\t');
    else
        %dlmwrite(fileStr, T,'precision', '%.6e','delimiter','\t');
        dlmwrite(fileStr, T,'precision', '%g','delimiter',' ');
    end
            
    fprintf('wrote %d lines in %g secs to %s.\n',size(T,1),toc(t1),fileStr)