function [files] = fastOpen(fileNames,whatToDo, fileType)
%FASTOPEN performs fopen of a list of files
%
% fileNames (cell array) = contains all fileNames
% whatToDo  (string)     = choose between 'r' read , 'w' write, 'rt' read textscan, 'wt' write textscan
% fileType  (string)     = choose type of file (.txt, .csv, .xslx, ...)
    nFiles = length(fileNames);
    files = cell(nFiles);
    for ii = 1:nFiles
        files{ii,1} = fopen([fileNames{ii},fileType], whatToDo);
    end
end

