function exportAllModesFromDataset(model,fileName,dirName, expression)
%EXPORTFROMDATASET 
% function to export all the modes of a comsol dataset (the plate surface)
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs:
%   model      = comsol model
%   fileName   = string - name of the without .txt
%   dirName    = string - name of the directory containing the file
%   expression = cell array - strings containing the expression of the
%                physical quantity exported - ex. solid.disp or solid.vel 
%                (displacement and velocity)
% -------------------------------------------------------------------------
% outputs:
% ~
% -------------------------------------------------------------------------
    % all comsol commands
    model.result.export('data1').setIndex('expr', expression, 0);               % set physical quantity exported
    model.result.export('data1').set('header', 'off');                          % delete header
    model.result.export('data1').set('filename',[dirName,'\',fileName,'.txt']); % set filename
    model.result.export('data1').set('sort', true);                             % sort by x
    model.result.export('data1').setIndex('looplevelinput', 'all', 0);          % set to export all modeshapes
    model.result.export('data1').run;                                           % run export
end