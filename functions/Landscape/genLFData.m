function [inputTable, ampsTable, eigTable] = genLFData(mechParams, model, paramsNames)
% GenLFData = generate loss function data
%
% Summary of this function goes here
%   Detailed explanation goes here
% 
    nPoints = 10;
    mechParamsVar = zeros(nPoints, length(mechParams));
    
    for ii = 1:length(mechParams)
        perc = 0.1;
        step =  2*perc*mechParams(ii)/(nPoints - 1 );
        mechParamsVar(:,ii) = mechParams(ii)*(1-perc): step : mechParams(ii)*(1+perc);
    end
    
    inputTable = writeMat2File(mechParamsVar,'inputParams.csv',...
    {'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'}, 11, true);
    
    for ii = 1:length(mechParams)
        for jj = 1:length(mechParamsVar(:,1))
            
            eigFreqz = table2array(readtable('Eigenfrequencies.csv'));
            Amps = table2array(readtable('Amplitudes.csv'));
            
            testParams = mechParams;
            testParams(ii) = mechParamsVar(jj,ii);
            disp(testParams);
            disp(mechParamsVar(:,ii));
            
            [vel,eigenFreqz] = comsolRoutine(model, testParams,paramsNames);
            vel = vel.';
            eigenFreqz = eigenFreqz.';
            
            Amps = [Amps; vel]
            eigFreqz = [eigFreqz; eigenFreqz]
                
           ampsTable =  writeMat2File(Amps,'Amplitudes.csv', {'f'}, 1,false);
           eigTable =  writeMat2File(eigFreqz,'Eigenfrequencies.csv', {'f'}, 1,false);    
        end
    end
end

