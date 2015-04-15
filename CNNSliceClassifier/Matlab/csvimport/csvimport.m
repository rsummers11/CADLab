function varargout = csvimport( fileName, varargin )
% CSVIMPORT reads the specified CSV file and stores the contents in a cell array or matrix
%
% The file can contain any combination of text & numeric values. Output data format will vary
% depending on the exact composition of the file data.
%
% CSVIMPORT( fileName ):         fileName     -  String specifying the CSV file to be read. Set to
%                                                [] to interactively select the file.
%
% CSVIMPORT( fileName, ... ) : Specify a list of options to be applied when importing the CSV file.
%                              The possible options are:
%                                delimiter     - String to be used as column delimiter. Default
%                                                value is , (comma)
%                                columns       - String or cell array of strings listing the columns
%                                                from which data is to be extracted. If omitted data
%                                                from all columns in the file is imported. If file
%                                                does not contain a header row, the columns
%                                                parameter can be a numeric array listing column
%                                                indices from which data is to be extracted.
%                                outputAsChar  - true / false value indicating whether the data
%                                                should be output as characters. If set to false the
%                                                function attempts to convert each column into a
%                                                numeric array, it outputs the column as characters
%                                                if conversion of any data element in the column
%                                                fails. Default value is false.
%                                uniformOutput - true / false value indicating whether output can be
%                                                returned without encapsulation in a cell array.
%                                                This parameter is ignored if the columns / table
%                                                cannot be converted into a matrix.
%                                noHeader      - true / false value indicating whether the CSV
%                                                file's first line contains column headings. Default
%                                                value is false.
%                                ignoreWSpace  - true / false value indicating whether to ignore
%                                                leading and trailing whitespace in the column
%                                                headers; ignored if noHeader is set to true.
%                                                Default value is false.
%
% The parameters must be specified in the form of param-value pairs, parameter names are not
% case-sensitive and partial matching is supported.
%
% [C1 C2 C3] = CSVIMPORT( fileName, 'columns', {'C1', 'C2', C3'}, ... )
%   This form returns the data from columns in output variables C1, C2 and C3 respectively, the
%   column names are case-sensitive and must match a column name in the file exactly. When fetching
%   data in column mode the number of output columns must match the number of columns to read or it
%   must be one. In the latter case the data from the columns is returned as a single cell matrix.
%
% [C1 C2 C3] = CSVIMPORT( fileName, 'columns', [1, 3, 4], ,'noHeader', true, ... )
%   This form returns the data from columns in output variables C1, C2 and C3 respectively, the
%   columns parameter must contain the column indices when the 'noHeader' option is set to true.

%
% Notes:  1. Function has not been tested on badly formatted CSV files.
%         2. Created using R2007b but has been tested on R2006b.
%
% Revisions:
%   04/28/2009: Corrected typo in an error message
%               Added igonoreWSpace option
%   08/16/2010: Replaced calls to str2num with str2double, the former uses eval leading to unwanted
%               side effects if cells contain text with function names
%

if ( nargin == 0 ) || isempty( fileName )
  [fileName filePath] = uigetfile( '*.csv', 'Select CSV file' );
  if isequal( fileName, 0 )
    return;
  end
  fileName = fullfile( filePath, fileName );
else
  if ~ischar( fileName )
    error( 'csvimport:FileNameError', 'The first argument to %s must be a valid .csv file', ...
      mfilename );
  end
end

%Setup default values
p.delimiter       = ',';
p.columns         = [];
p.outputAsChar    = false;
p.uniformOutput   = true;
p.noHeader        = false;
p.ignoreWSpace    = false;

validParams     = {     ...
  'delimiter',          ...
  'columns',            ...
  'outputAsChar',       ...
  'uniformOutput',      ...
  'noHeader',           ...
  'ignoreWSpace'        ...
  };

%Parse input arguments
if nargin > 1
  if mod( numel( varargin ), 2 ) ~= 0
    error( 'csvimport:InvalidInput', ['All input parameters after the fileName must be in the ' ...
      'form of param-value pairs'] );
  end
  params  = lower( varargin(1:2:end) );
  values  = varargin(2:2:end);

  if ~all( cellfun( @ischar, params ) )
    error( 'csvimport:InvalidInput', ['All input parameters after the fileName must be in the ' ...
      'form of param-value pairs'] );
  end

  lcValidParams   = lower( validParams );
  for ii =  1 : numel( params )
    result        = strmatch( params{ii}, lcValidParams );
    %If unknown param is entered ignore it
    if isempty( result )
      continue
    end
    %If we have multiple matches make sure we don't have a single unambiguous match before throwing
    %an error
    if numel( result ) > 1
      exresult    = strmatch( params{ii}, validParams, 'exact' );
      if ~isempty( exresult )
        result    = exresult;
      else
        %We have multiple possible matches, prompt user to provide an unambiguous match
        error( 'csvimport:InvalidInput', 'Cannot find unambiguous match for parameter ''%s''', ...
          varargin{ii*2-1} );
      end
    end
    result      = validParams{result};
    p.(result)  = values{ii};
  end
end

%Check value attributes
if isempty( p.delimiter ) || ~ischar( p.delimiter )
  error( 'csvimport:InvalidParamType', ['The ''delimiter'' parameter must be a non-empty ' ...
    'character array'] );
end
if isempty( p.noHeader ) || ~islogical( p.noHeader ) || ~isscalar( p.noHeader )
  error( 'csvimport:InvalidParamType', ['The ''noHeader'' parameter must be a non-empty ' ...
    'logical scalar'] );
end
if ~p.noHeader
  if ~isempty( p.columns )
    if ~ischar( p.columns ) && ~iscellstr( p.columns )
      error( 'csvimport:InvalidParamType', ['The ''columns'' parameter must be a character array ' ...
        'or a cell array of strings for CSV files containing column headers on the first line'] );
    end
    if p.ignoreWSpace
      p.columns = strtrim( p.columns );
    end
  end
else
  if ~isempty( p.columns ) && ~isnumeric( p.columns )
    error( 'csvimport:InvalidParamType', ['The ''columns'' parameter must be a numeric array ' ...
      'for CSV files containing column headers on the first line'] );
  end
end
if isempty( p.outputAsChar ) || ~islogical( p.outputAsChar ) || ~isscalar( p.outputAsChar )
  error( 'csvimport:InvalidParamType', ['The ''outputAsChar'' parameter must be a non-empty ' ...
    'logical scalar'] );
end
if isempty( p.uniformOutput ) || ~islogical( p.uniformOutput ) || ~isscalar( p.uniformOutput )
  error( 'csvimport:InvalidParamType', ['The ''uniformOutput'' parameter must be a non-empty ' ...
    'logical scalar'] );
end

%Open file
[fid msg] = fopen( fileName, 'rt' );
if fid == -1
  error( 'csvimport:FileReadError', 'Failed to open ''%s'' for reading.\nError Message: %s', ...
    fileName, msg );
end

colMode         = ~isempty( p.columns );
if ischar( p.columns )
  p.columns     = cellstr( p.columns );
end
nHeaders        = numel( p.columns );

if colMode
  if ( nargout > 1 ) && ( nargout ~= nHeaders )
    error( 'csvimport:NumOutputs', ['The number of output arguments must be 1 or equal to the ' ...
      'number of column names when fetching data for specific columns'] );
  end
end

%Read first line and determine number of columns in data
rowData         = fgetl( fid );
rowData         = regexp( rowData, p.delimiter, 'split' );
nCols           = numel( rowData );

%Check whether all specified columns are present if used in column mode and store their indices
if colMode
  if ~p.noHeader
    if p.ignoreWSpace
      rowData     = strtrim( rowData );
    end
    colIdx        = zeros( 1, nHeaders );
    for ii = 1 : nHeaders
      result      = strmatch( p.columns{ii}, rowData );
      if isempty( result )
        fclose( fid );
        error( 'csvimport:UnknownHeader', ['Cannot locate column header ''%s'' in the file ' ...
          '''%s''. Column header names are case sensitive.'], p.columns{ii}, fileName );
      elseif numel( result ) > 1
        exresult  = strmatch( p.columns{ii}, rowData, 'exact' );
        if numel( exresult ) == 1
          result  = exresult;
        else
          warning( 'csvimport:MultipleHeaderMatches', ['Column header name ''%s'' matched ' ...
            'multiple columns in the file, only the first match (C:%d) will be used.'], ...
            p.columns{ii}, result(1) );
        end
      end
      colIdx(ii)  = result(1);
    end
  else
    colIdx        = p.columns(:);
    if max( colIdx ) > nCols
      fclose( fid );
      error( 'csvimport:BadIndex', ['The specified column index ''%d'' exceeds the number of ' ...
        'columns (%d) in the file'], max( colIdx ), nCols );
    end
  end
end

%Calculate number of lines
pos             = ftell( fid );
if pos == -1
  msg = ferror( fid );
  fclose( fid );
  error( 'csvimport:FileQueryError', 'FTELL on file ''%s'' failed.\nError Message: %s', ...
    fileName, msg );
end
data            = fread( fid );
nLines          = numel( find( data == sprintf( '\n' ) ) ) + 1;
%Reposition file position indicator to beginning of second line
if fseek( fid, pos, 'bof' ) ~= 0
  msg = ferror( fid );
  fclose( fid );
  error( 'csvimport:FileSeekError', 'FSEEK on file ''%s'' failed.\nError Message: %s', ...
    fileName, msg );
end

data            = cell( nLines, nCols );
data(1,:)       = rowData;
emptyRowsIdx    = [];
%Get data for remaining rows
for ii = 2 : nLines
  rowData       = fgetl( fid );
  if isempty( rowData )
    emptyRowsIdx = [emptyRowsIdx(:); ii];
    continue
  end
  rowData       = regexp( rowData, p.delimiter, 'split' );
  nDataElems    = numel( rowData );
  if nDataElems < nCols
    warning( 'csvimport:UnevenColumns', ['Number of data elements on line %d (%d) differs from ' ...
      'that on the first line (%d). Data in this line will be padded.'], ii, nDataElems, nCols );
    rowData(nDataElems+1:nCols) = {''};
  elseif nDataElems > nCols
    warning( 'csvimport:UnevenColumns', ['Number of data elements on line %d (%d) differs from ' ...
      'that one the first line (%d). Data in this line will be truncated.'], ii, nDataElems, nCols );
    rowData     = rowData(1:nCols);
  end
  data(ii,:)    = rowData;
end
%Close file handle
fclose( fid );
data(emptyRowsIdx,:)   = [];

%Process data for final output
uniformOutputPossible  = ~p.outputAsChar;
if p.noHeader
  startRowIdx          = 1;
else
  startRowIdx          = 2;
end
if ~colMode
  if ~p.outputAsChar
    %If we're not outputting the data as characters then try to convert each column to a number
    for ii = 1 : nCols
      colData     = cellfun( @str2double, data(startRowIdx:end,ii), 'UniformOutput', false );
      %If any row contains an entry that cannot be converted to a number then return the whole
      %column as a char array
      if ~any( cellfun( @isnan, colData ) )
        if ~p.noHeader
          data(:,ii)= cat( 1, data(1,ii), colData{:} );
        else
          data(:,ii)= colData;
        end
      end
    end
  end
  varargout{1}    = data;
else
  %In column mode get rid of the headers (if present)
  data            = data(startRowIdx:end,colIdx);
  if ~p.outputAsChar
    %If we're not outputting the data as characters then try to convert each column to a number
    for ii = 1 : nHeaders
      colData     = cellfun( @str2double, data(:,ii), 'UniformOutput', false );
      %If any row contains an entry that cannot be converted to a number then return the whole
      %column as a char array
      if ~any( cellfun( @isnan, colData ) )
        data(:,ii)= colData;
      else
        %If any column cannot be converted to a number then we cannot convert the output to an array
        %or matrix i.e. uniform output is not possible
        uniformOutputPossible = false;
      end
    end
  end
  if nargout == nHeaders
    %Loop through each column and convert to matrix if possible
    for ii = 1 : nHeaders
      if p.uniformOutput && ~any( cellfun( @ischar, data(:,ii) ) )
        varargout{ii} = cell2mat( data(:,ii) );
      else
        varargout{ii} = data(:,ii);
      end
    end
  else
    %Convert entire table to matrix if possible
    if p.uniformOutput && uniformOutputPossible
      data        =  cell2mat( data );
    end
    varargout{1}  = data;
  end
end
