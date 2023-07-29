function [] = setParams(model,paramsNames, paramsVals)
% SETPARAMS(model,paramsNames, paramsVals)
% This function set the params corresponding to *paramsNames* in the comsol
% model *model* to the values *paramsVals*
% -------------------------------------------------------------------------
% INPUTS
%   model = comsol.mph model
%   paramsNams = (cell array) cell array with the names of the parameters
%                             to set
%   paramsVals = array - values of the parameters
% -------------------------------------------------------------------------
% OUTPUTS:
% ~
% -------------------------------------------------------------------------
    for jj = 1:length(paramsVals)
            model.param.set(paramsNames(jj), paramsVals(jj));
    end
end