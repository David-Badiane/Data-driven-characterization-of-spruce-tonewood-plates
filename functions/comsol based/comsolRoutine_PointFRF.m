function [Dataset_FRFs] = ...
            comsolRoutine_PointFRF(model, nTuples, referenceVals, standardDev, csvPath,...
            meshSize, fAxis, fHigh, nPts_fAxis, comsolDset, paramsIdxsModel, expression)
    
    if nargin < 10, comsolDset = 'cpt1'; end
    if nargin < 11, paramsIdxsModel = 1:12; end
    
    Dataset_FRFs = struct('inputs', [], 'FRFs', [], 'fAx', []);
    
    inputs = [];
    FRFs   = [];
    params = mphgetexpressions(model.param);
    varyingParamsNames = params(paramsIdxsModel,1);
    
    for ii = 1:nTuples
        tStart = tic;
        disp(['tuple number ', num2str(ii),  ' - start']);
        if ii == 1
            currentVals = referenceVals;
        else    
            currentVals = gaussianSample(referenceVals, standardDev);
            currentVals(11) = 50 + 50*(2*rand(1,1)-1);
            % beta uniform btw 2e-7 and 2e-5
            currentVals(12) = 2*10.^(rand(1,1)*2 - 7); 
        end
        
        disp(['current Vals = ', num2str(currentVals(:).',2)]);
        
        
        [vels, fAxisComsol] = ...
            comsolPointFRFVals(model, currentVals, csvPath, meshSize, fAxis, fHigh, nPts_fAxis,...
            comsolDset, paramsIdxsModel, 0, 0, expression);
        
        if ii == 1 
            Dataset_FRFs.fAx = fAxisComsol;
            writeMat2File(fAxisComsol, 'fAx.csv', {'f'}, 1, 0);
        end
        
        inputs = [inputs; currentVals];
        FRFs   = [FRFs; vels];   
        
        writeMat2File(FRFs, 'FRFs.csv', {'f'}, 1, 0);
        writeMat2File(inputs, 'inputs.csv', varyingParamsNames, 12, 1);
        
        Dataset_FRFs.inputs = inputs;
        Dataset_FRFs.FRFs = FRFs;
        
        disp(['finished tuple ', num2str(ii)])
    end
end

function [currentVals] = gaussianSample(referenceVals, standardDev)
    gaussRealization = randn(size(referenceVals));
    currentVals = referenceVals.*(ones(size(referenceVals)) + standardDev.*gaussRealization);
end