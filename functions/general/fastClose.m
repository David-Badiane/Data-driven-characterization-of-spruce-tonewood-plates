function [] = fastClose(files)
%FASTCLOSE - closes the files specified in the cell array
%   
% files = cell array containing all files
nFiles = length(files);
for ii = 1:nFiles
fclose(files{ii});
end

end

