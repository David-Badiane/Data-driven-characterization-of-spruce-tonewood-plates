function [xpar, map, f_out, amp_notScaled, fval] = minimization_FA(fNet, ampNet, Dataset_FA, f0, fAmps,rho, NpeaksAxis, plotData)
%MINIMIZATION_FA Summary of this function goes here
%   Detailed explanation goes here
    
    [L2, firstGuessLoc] = computeFirstGuessFA(Dataset_FA, fAmps, f0, false);
    inputParsStart = Dataset_FA.inputs(firstGuessLoc,2:end);
    
    fun = @(x) errorFA(x , fNet, ampNet, f0, fAmps, rho, NpeaksAxis, inputParsStart.', 0.05, plotData);
    options = optimset('fminsearch');
    options = optimset(options, 'TolFun',1e-4,'TolX',1e-4, 'MaxFunEvals',50e3,'MaxIter', 50e3,...
    'DiffMinChange', 1, 'DiffMaxChange', 200); 
   [xpar, fval, fvals, fvals_freq, fvals_amp] = steepestDescentRelativeConst...
       (fun,inputParsStart.', 1e-10, 1e-10, 0.001, 300);
    
    [xpar,fval, exitflag, output] = fminsearch(fun, inputParsStart.', options)
    xpar = [rho; xpar];
    
    f_out = fNet(xpar);
    amp_notScaled = db2mag(ampNet(xpar));
    amp_out = amp_notScaled
    
    %% SEE RESULTS AND COMPUTE MAP
    map = zeros(1, length(NpeaksAxis));    
    ratio = (fAmps(1:5))./(amp_out(1:5));
    gamma = mean(ratio);
    % Frequency/amplitude scaling
    amp_out = gamma * amp_out;
    ampsReal =  fAmps;
    eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
    amp_out = eta*amp_out;
    ampsReal = eta*ampsReal;
    
    if plotData
        figure(200)
        plot( f_out, amp_out , '.', 'markerSize', 10)
        hold on;
        xlabel('frequency');
        ylabel('amplitude');
        title(['jj = ', num2str(jj)]);
        plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '.', 'markerSize' ,10)
        xlim([f0(1)-10, f_out(end)+20]);    
    end
    
    pointsNN = [f_out, amp_out];
    pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];

    psi = 600;
    
    for kk = NpeaksAxis
        ampsDiff = pointsReal(kk,2) - pointsNN(:,2);                
        fDiffReal = (pointsReal(kk,1) - pointsNN(:,1))./(pointsReal(kk,1));
        dist = sqrt(( psi* fDiffReal).^2 + (ampsDiff).^2);
        [minDist, minLoc] = min(dist);

        lineFreqz =  [f0(kk), f_out(minLoc)];
        lineAmps = [ampsReal(kk), amp_out(minLoc)];
        if plotData
            plot(lineFreqz, lineAmps);
            pause(0.2);
        end
        map(1,kk) = minLoc;
    end


end

