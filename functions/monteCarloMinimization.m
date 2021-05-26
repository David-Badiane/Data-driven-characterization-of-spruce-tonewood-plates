 function [solution] = monteCarloMinimization(fun,firstGuess, startStepSize)
%MONTECARLOMINIMIZATION Summary of this function goes here
%   Detailed explanation goes here
% startStepSize = 1000;
% fun = @(xx) ackley(xx);

algorithmStep = 1;
fCount = 0;
iCount = 0;
iRoc = 0;
iStep = 0;
nRuns =0;
n = length(firstGuess);
exitFlag = false;



while ~exitFlag 
    
    stepSize = zeros(1,21);
for ii = 1:21
   % disp(stepSize);
    
    if ii >= 1 && ii <= 4
        stepSize(ii) = startStepSize;
    else
        if ii >= 5 && ~(stepSize(ii-1)> sqrt(2)*0.8 && stepSize(ii-1) <1.2*sqrt(2)) && ii <= 17
            stepSize(ii) = 1/2*stepSize(ii-1);
            idxSqrt2 = ii;      
        end

        if (stepSize(ii-1)> sqrt(2)*0.8 && stepSize(ii-1) <1.2*sqrt(2)) && ii <= idxSqrt2 + 2
            stepSize(ii) = stepSize(ii-1);
        end

        if ii> idxSqrt2 +2
           stepSize(ii) = 1/2*stepSize(ii-1);
        end
    
    if ii >17 && ii<21
        stepSize(ii) = 1/10*stepSize(ii-1);
    end
    end
end

switch algorithmStep
    % 1) INIT
    case 1
    x = firstGuess;
    delta = rand(size(x));
    b = x + delta;
    fb = 1e6;
    algorithmStep = 2;
    
    % 2) Evaluate f(x)
    case 2
    fx = feval(fun, x);
    algorithmStep = 3;
    
    % 3) Compare f(x) and f(b)
    case 3
    if fx < fb
       b = x;
       fb = fx;
       fCount = 0;
       algorithmStep = 5;
    else
        disp([fx, fb]);
        fCount = fCount + 1;
        algorithmStep = 4;
    end
    
    % 4) Check fCount, iStep, iRoc
    case 4
        disp([fCount, iStep, iRoc]);

        
        if fCount > 20 && iRoc <3
                iStep = iStep +1;
                fCount = 0;
                algorithmStep = 5;
        end
        
        if iStep > n
            iStep = 0;
            iRoc = iRoc +1;
            fCount = 0;
            algorithmStep = 5;
        end
        algorithmStep = 5;
        
        if fCount > 20 && iRoc == 3
            algorithmStep = 7;
        end 
  
    % 5) Choose new coordinates for the next step
    case 5
        if iStep == 0 && iRoc <3
           % simultaneous change of variables x 
           rx = rand(size(b))*stepSize(fCount+1);
           cx = rand(size(b))*2 - 1;
           x = b  + rx.*cx
        end
        if iStep > 0 && iRoc <3
           % change only x(iStep)
           rx = rand(1,1)*stepSize(fCount+1);
           cx = rand(1,1)*2-1;
           x(iStep) = b(iStep) + rx*cx
        end
        algorithmStep = 6;
    
    % 6) Recalculation of f(x)
    case 6
        algorithmStep = 2;
        
    % 7) Termination
    case 7
        if nRuns > Runs
            % b that are in a certain threshold of precision
            solution = mean(b);
            exitFlag = true;
        else
            nRuns = nRuns +1;
            algorithmStep = 2;
            fCount =0;
            iRoc = 0;
            iStep = 0;
            startStepSize = startStepSize/2;
        end
    end
 end

