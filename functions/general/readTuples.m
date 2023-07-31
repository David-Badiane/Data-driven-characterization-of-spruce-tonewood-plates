function [fileData] = readTuples(filename, rows, transpose)
%   readTuples 
%   This function allows to retrieve a nVals*nCols matrix from a txt file
%   nCols depends from the length of the file, if the matrix
%   nCols*nVals is desired, apply transposed.
%   When you call it be sure to be in the same directory where 
%   the source file is present.
%   -----------------------------------------------------------------------
 file = fopen(filename,'rt'); % open file to read text
 formatSpec = '%f';           % format of the file = floats
 fileData = cell2mat(textscan(file,formatSpec)); % scan file
 % retrieve nCols
 cols = round(length(fileData)/rows);
 fileData = reshape(fileData,[rows, cols]);
 
 fclose(file);
 if transpose 
     fileData = fileData.';
 end
end

