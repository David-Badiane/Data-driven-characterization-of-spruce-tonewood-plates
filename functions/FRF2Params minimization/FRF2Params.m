function [mechParameters, maps, f_out, amp_out] = ...
          FRF2Params(options, fNet, aNet, f0, fAmps,...
          NpeaksAxis, plotData, fixParamsVals, fixParamsIdxs,inputParsStart)
% errorFA - objective function of the minimization - predicts frequency and amplitude of FRF peaks for given material
% properties and computes the loss function; allows to evaluate multiple FRFs computed on multiple points
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
% options       = struct with options for minimization algorithm
% fNet          = neural network object - neural network predicting eigenfrequencies
% aNet          = neural network object - neural network predicting amplitudes
% f0            = nPts x 1 cell - in each cell we have the frequencies of
%                 the peaks of a single FRF
% fAmps         = nPts x 1 cell - in each cell we have the amplitudes of
%                 the peaks of a single FRF
% rho           = 1x1 double - density of the plate
% NpeaksAxis    = nPeaks x 1 double - axis with the FRF peaks considered 
%                                  in the minimization
% plotData      = boolean to decide whether to plot or not
% fixParamsVals = values for the material properties that are not optimized,
%                 (at least density and geometry)
% fixParamsIdxs = indexes of the fixed params in the mechParams array
% inputParsStart = first guess of material properties 
% -------------------------------------------------------------------------
% outputs: 
% L2            = 1x1 double - loss function value
% maps          = real FRF - estimation associations
% -------------------------------------------------------------------------
    
   nPts = length(f0);
   figN = 200;
   parsIdxs = setdiff(1:length(inputParsStart), fixParamsIdxs);
   parameters = inputParsStart(parsIdxs);
   
   % loss function to be minimized is fun
   fun = @(x) errorFA(x , fNet, aNet, f0, fAmps, NpeaksAxis, plotData(2), fixParamsVals, fixParamsIdxs, nPts, figN);
     
 
    if plotData(3) % if plot first guess flag is on, do it
      [L2, maps] = errorFA(parameters ,fNet, aNet, f0, fAmps,NpeaksAxis,...
                                           plotData(3), fixParamsVals, fixParamsIdxs, nPts, figN);  
    end    

    % call Neldear-Mead minimization algorithm
    [xpar,fval, exitflag, output] = fminsearch(fun, parameters.', options);
    output % display minimization output
    
    % see the result of the minimization
    [L2, maps] = errorFA(xpar ,fNet, aNet, f0, fAmps,NpeaksAxis,...
       plotData(1), fixParamsVals, fixParamsIdxs, nPts, figN);
    disp(['L2 opt = ', num2str(L2, 2)]);
    
    % put together estimated params and fixed params
    mechParameters = ones(15,1);
    mechParameters(fixParamsIdxs) = fixParamsVals;
    mechParameters(setdiff(1:15,fixParamsIdxs)) = xpar; 
    
    % make final figures, compute map in frequency only space - to compare to freq/amp map
    f_out = fNet(mechParameters);
    amp_out = db2mag(aNet(mechParameters));
end

