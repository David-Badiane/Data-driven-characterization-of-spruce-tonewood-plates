function [dataTable] = writeMat2File(data,dstFileName, variablesName, nCols, singleTitles)
% WRITEMAT2FILE
% custom function to write a csv file with given variable names (columns names)
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs:
%   data          = (nTuples x nCols) double - data to write in the file
%   dstFileName   = string - filename of the file to be written (contains tag ex. .csv)
%   variablesName = cell array - cell array with the labels of the columns
%      |---------------------------------------------------------------------|
%      | if singleTitles == 0 - variablesName has less entries than nCols    |  
%      | ex. variableName = {'f' 'g'} & singleTitles == 0 --> cols names are |
%      |     yields as columns names 'f1' 'g1' 'f2' 'g2' ..... and so on     |
%      | if singleTitles == 1 - variableName must contain nCols entries      |
%      |---------------------------------------------------------------------|
%   nCols         = int - number of columns of the data
%   singleTitles  = boolean - look at variableName description
% -------------------------------------------------------------------------
% outputs:
%   dataTable     = table - written data in the form of a table
% -------------------------------------------------------------------------

% variables
columns = length(data(1,:));
nIndexes = ceil(columns/nCols);
indexes = 1:nIndexes; % indexes array
temp = 1; % used as counter



% for cycle to write variableNames
if singleTitles == true 
    names = cell(size(variablesName));
        for ii = 1:length(variablesName)
            names{ii} = variablesName{ii};
        end
else
    names = cell(1,columns);
    for ii = 1:columns
        names{ii} = [variablesName{1+mod(ii-1, nCols)}, int2str(indexes(temp))];
        if mod(ii,nCols) == 0
            temp = temp + 1;
        end
    end
end

% obtain table and save it
dataTable = array2table(data, 'VariableNames', names);
writetable(dataTable, dstFileName);
% disp message
disp(['wrote (' num2str(size(data,1)) 'x' num2str(size(data,2)) ') array ', newline, ...
      'filename: ' dstFileName, newline, 'directory: ', pwd, newline]);
end

