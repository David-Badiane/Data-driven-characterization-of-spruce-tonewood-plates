function [inputTable, ampsTable, eigTable] = genFreqAmpData(mechParams, model, paramsNames)
% GenLFData = generate loss function data
%
% Summary of this function goes here
%   Detailed explanation goes here
% 
%     mechParamsVar = zeros(nPoints, length(mechParams));    
%     for ii = 1:length(mechParams)
%         perc = 0.1;
%         step =  2*perc*mechParams(ii)/(nPoints - 1 );
%         mechParamsVar(:,ii) = mechParams(ii)*(1-perc): step : mechParams(ii)*(1+perc);
%     end
%     
    nPoints = 10;
    alphaBetaIdxs = [length(mechParams)-1, length(mechParams)];
    alphaBetaRef = mechParams(alphaBetaIdxs);
    alphaBetaVar = zeros(nPoints,2);
    
    for ii = 1:2
        perc = 0.3;
        step =  2*perc*alphaBetaRef(ii)/(nPoints - 1 );
        alphaBetaVar(:,ii) = alphaBetaRef(ii)*(1-perc): step : alphaBetaRef(ii)*(1+perc);
    end
    
    for ii = 1:length(alphaBetaRef)
        eigFilename = ['Eigenfrequencies',int2str(ii) ,'.csv'];
        ampFilename = ['Amplitudes',int2str(ii) ,'.csv'];
        
        for jj = 1:nPoints
            
            eigFreqz = table2array(readtable(eigFilename));
            Amps = table2array(readtable(ampFilename));
            
            testParams = mechParams;
            testParams(alphaBetaIdxs(ii)) = alphaBetaVar(jj,ii);
            
            [vel,eigenFreqz] = comsolRoutineFreqAmp_previous(model, testParams,paramsNames);
            vel = vel.';
            eigenFreqz = eigenFreqz.';
            
            Amps = [Amps; vel]
            eigFreqz = [eigFreqz; eigenFreqz]
                
           ampsTable =  writeMat2File(Amps,ampFilename, {'f'}, 1,false);
           eigTable =  writeMat2File(eigFreqz,eigFilename, {'f'}, 1,false);    
        end
    end
end

