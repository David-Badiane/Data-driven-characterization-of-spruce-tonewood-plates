function [params,freqz,errs] = studyMinError_eig( realIndex, comsolIndex, modesPermute, modesNames, maxLoc,linearModels, inputsInfo, outputsALLInfo, f0,minimumLoc,rho,numMechParams, plotData)
%STUDYMINIMIZATIONERROR Summary of this function goes here
%   Detailed explanation goes here
nModesPermute = length(modesPermute);
nFreq =  length(comsolIndex);

errs =  cell(nModesPermute,1); 
params = cell(nModesPermute ,1); 
freqz = cell(nModesPermute, 1);
for jj = modesPermute

    comsolPermute = nchoosek(comsolIndex,jj);
    disp(length(comsolPermute(:,1)));
    realPermute = nchoosek(realIndex,jj);

    % preallocate
    errs{jj} = cell(length(comsolPermute(:,1)),jj+2);
    params{jj} = cell(length(comsolPermute(:,1)), jj + numMechParams );
    freqz{jj} = cell(length(comsolPermute(:,1))+1, jj+nFreq);

    freqz{jj}(1,1:nFreq) = num2cell(f0(realIndex));

    for ii = 1:length(comsolPermute(:,1))
        
        [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo,...
                                            f0,minimumLoc,rho, comsolPermute(ii,:), realPermute(ii,:), false);
        % error                                
        diff = (f0(realIndex) - f_out(comsolIndex))./f0(realIndex);                                
        errs{jj}{ii,1} = mean(abs(diff))*100; 
        % mech Params and eigenfrequencies
        params{jj}(ii,1:numMechParams) = num2cell(xpar);
        freqz{jj}(ii+1,1:nFreq) = num2cell(f_out(comsolIndex).');
        
        if plotData
            figure()
            plot(realIndex, f_out(comsolIndex), '-o');
            hold on 
            plot(realIndex, f0(realIndex), '-x');
            xlabel('N mode')
            ylabel(' f     [Hz]');
            legend('fMultilin', 'fexp');
        end
        
        % labeling for used modeshapes in the minimization
        for kk = 1:length(comsolPermute(ii,:))
            errs{jj}{ii,kk+1} = modesNames{maxLoc,comsolPermute(ii,kk)}; 
            params{jj}{ii, kk+ numMechParams} = modesNames{1,comsolPermute(ii,kk)};
            freqz{jj}{ii+1, kk+ nFreq} = modesNames{1,comsolPermute(ii,kk)};
        end
        
        % label if there are negative mech parameters or not
        check = find(xpar<0);
        errLength = length(errs{jj}(ii,:));
        if isempty(check)
            errs{jj}{ii, end}  = 'positive MP';        
        else
            errs{jj}{ii, end} = 'negative MP';
        end
        
        % check results at each iteration
        params{jj}(ii,:)
        freqz{jj}(ii+1,:)
        errs{jj}(ii,:)
    end
end

end

