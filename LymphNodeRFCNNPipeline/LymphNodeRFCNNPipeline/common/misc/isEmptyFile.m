function answer = isEmptyFile(filename)

fileID = fopen(filename,'r'); 

if fseek(fileID, 1, 'bof') == -1
   % empty file
   answer = true;
else
   frewind(fileID)
   % ready to read
   answer = false;
end

fclose(fileID);
