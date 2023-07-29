function [] = exportData(model,dset, dirName,filename,dataExpression)
%exportData --- exports data from the model
% -------------------------------------------------------------------------
% INPUTS:
%   model (Comsol model) = Comsol model from which export data
%   dset     (string)    = Dataset from which export (ex. 'dset1')
%   dirName  (string)    = Address of the directory where we export data
%   filename (string)    = Name of the file to export
%   dataExpression (cell array) = name of the data to export 
%   (ex.{'solid.disp'})
% -------------------------------------------------------------------------
% OUTPUTS:
%   ~
%--------------------------------------------------------------------------
% copyright: David Giuseppe Badiane
        model.result.export('data1').set('header', 'off');
        model.result.export('data1').set('data', dset);
        model.result.export('data1').set('expr', dataExpression);
        model.result.export('data1').set('descr', {''});
        model.result.export('data1').set('filename',[dirName,'\',filename,'.txt']);
        model.result.export('data1').run;
end

