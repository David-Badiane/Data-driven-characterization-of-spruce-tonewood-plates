function exportModeFromDataset(model,fileName,dirName, expression, eigenFreq)
%EXPORTFROMDATASET Summary of this function goes here
%   model      = comsol model
%   fileName   = name of the without .txt
%   dirName    = name of the directory containing the file
%   expression = cell array with strings containing the expressions
%   eigenFreq  = the eigenfrequency of the eigenmode

    model.result.export('data1').set('expr', expression);
    model.result.export('data1').set('header', 'off');
    model.result.export('data1').set('filename',[dirName,'\',fileName,'.txt']);
    %model.result.export('data1').set('sort', true);
    %model.result.export('data1').setIndex('looplevelinput', 'all', 0);
    model.result.export('data1').setIndex('looplevelindices', eigenFreq, 0);
    model.result.export('data1').run;
end