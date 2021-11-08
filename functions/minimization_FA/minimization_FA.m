function [xpar, map, f_out, amp_out, fval, idxComsol] = ...
          minimization_FA(options, MLR_freq, ampNet, Dataset_FA, f0, fAmps,rho,...
          NpeaksAxis, plotData, algorithm, rayleighParams, normalization, ampScale,lb,ub,scaleBinds)
%MINIMIZATION_FA Main of the minimization algorithm for parameters identification    

    fun1 = @(x) errorFA(x , MLR_freq, ampNet, f0, fAmps, rho,NpeaksAxis, plotData(2), algorithm, rayleighParams, normalization, ampScale, scaleBinds);
    fun2 = @(x) errorFrequency(x , MLR_freq,  f0,  rho, NpeaksAxis);
    
%   [xpar1, fval, fvals, fvals_freq, fvals_amp] = steepestDescentRelativeConst...
%        (fun1,inputParsStart.', 1e-7, 1e-7, 0.01, 100);    
%   [xpar2, fval, fvals, fvals_freq, fvals_amp] = steepestDescentRelativeConst...
%        (fun2,inputParsStart.', 1e-7, 1e-7, 0.01, 100);
%   [xpar, fval] = steepestDescentRelative...
%       (fun1,inputParsStart.', 1e-10, 1e-10, 0.9, 300);
    
   [L2, firstGuessLoc] = computeFirstGuessFA(Dataset_FA, fAmps, f0, NpeaksAxis, plotData(3), normalization, ampScale);
   inputParsStart = Dataset_FA.inputs(firstGuessLoc,2:end);
   disp('first guess:');
   fun1(inputParsStart.');
   
%    inputParsStart(10) = rayleighParams(1);
%    inputParsStart(11) = rayleighParams(2);
%   
   [xpar,fval, exitflag, output] = fminsearch(fun1, inputParsStart.', options);
   opts = optimoptions(@simulannealbnd, 'InitialTemperature', 70);
   [xpar,fval,exitFlag,output] = simulannealbnd(fun1,inputParsStart.',lb(2:end).',ub(2:end).', opts);
%    [solution, lastB] = monteCarloMinimization(fun1,inputParsStart.', 50, 1e-6)
   % see the result of the minimization
   [L2, L2_freq, L2_amp, map] = errorFA(xpar , MLR_freq, ampNet, f0, fAmps, rho,NpeaksAxis, true,'fixRayleighParams', rayleighParams, normalization, ampScale, scaleBinds);
    xpar = [rho; xpar];
    
    [fNN, idxComsol] = getFrequencyEstimation(f0, MLR_freq, xpar, NpeaksAxis, plotData(1));
    f_out =  predictEigenfrequencies(MLR_freq.linMdls , xpar.', length(MLR_freq.linMdls)).';
%     f_out = MLR_freq(xpar); f_out = f_out(:).';
    amp_out = db2mag(ampNet(xpar));
end

