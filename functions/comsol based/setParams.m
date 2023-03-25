function [] = setParams(model,paramsNames, paramsVals)
    for jj = 1:length(paramsVals)
            model.param.set(paramsNames(jj), paramsVals(jj));
    end
end