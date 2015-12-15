function write2mps(points,mps_filename)
    mitkSpecification = '0';
    mitkTimeSeries_id = 0;

    N = size(points,1);

    fid = fopen(mps_filename,'w');

    fprintf(fid,'<?xml version="1.0" encoding="ISO-8859-1"?>\n');
    fprintf(fid,'<point_set_file>\n');
    fprintf(fid,'  <file_version>0.1</file_version>\n');
    fprintf(fid,'  <point_set>\n');
    fprintf(fid,'    <time_series>\n');
    fprintf(fid,'      <time_series_id>%d</time_series_id>\n',mitkTimeSeries_id);
    for i = 1:N
        fprintf(fid,'      <point>\n');
        fprintf(fid,'        <id>%d</id>\n',i-1); % start counting from zero!
        fprintf(fid,'        <specification>%s</specification>\n',mitkSpecification);
        fprintf(fid,'        <x>%g</x>\n', points(i,1));
        fprintf(fid,'        <y>%g</y>\n', points(i,2));
        fprintf(fid,'        <z>%g</z>\n', points(i,3));
        fprintf(fid,'      </point>\n');          
    end
    fprintf(fid,'    </time_series>\n'); 
    fprintf(fid,'  </point_set>\n');
    fprintf(fid,'</point_set_file>\n');

    fclose(fid);
      