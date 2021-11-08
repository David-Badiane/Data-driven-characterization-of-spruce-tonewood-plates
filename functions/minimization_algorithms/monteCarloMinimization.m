function [solution, lastB] = monteCarloMinimization(fun,firstGuess, startStepSize, tolFx)
%MONTECARLOMINIMIZATION Summary of this function goes here
%   Detailed explanation goes here
% startStepSize = 1000;
% fun = @(xx) ackley(xx);
firstGuess= firstGuess(:);
algorithmStep = 1;
fCount = 0;
iCount = 0;
iRoc = 0;
iStep = 0;
nRuns =0;
n = length(firstGuess);
exitFlag = false;
Runs = 1;
bb = [];

disp(['init': num2str(fun(firstGuess))])

while ~exitFlag 
    
    stepSize = zeros(1,21);
for ii = 1:21
    
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
    b = x ;%+ delta;
    fb = 1e6;
    algorithmStep = 2;
    
    % 2) Evaluate f(x)
    case 2
    fx = fun(x);
    algorithmStep = 3;
    
    % 3) Compare f(x) and f(b)
    case 3
    if fx < fb
        if fb - fx < tolFx
            solution = mean(bb);
            lastB = b;
            exitFlag = true;
        end
        
       b = x;
       bb = [bb; b.']; 
       fb = fx;
       fCount = 0;
       algorithmStep = 5;
       
    else
        disp(['fx:     ', num2str(fx), ' fb:     ' , num2str(fb)]);
        fCount = fCount + 1;
        algorithmStep = 4;
    end
    
    % 4) Check fCount, iStep, iRoc
    case 4
        disp(['fCount: ', num2str(fCount),' iStep:  ', num2str(iStep), ' iRoc:   ', num2str(iRoc)]);
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
           x = b  + rx.*cx;
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
            solution = mean(bb);
            lastB = b;
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

