function [err] = errorMultiLin(mechParams , f0,linearModels, nModes , rho, referenceVals, indexComsol, indexReal, constraintWeight)
%ERROR Summary of this function goes here
%   Detailed explanation goes here
    mechParameters = [rho; mechParams];
    [f_ML] = eigfreqzFromCoefficients(mechParameters, linearModels, nModes);
    index = indexComsol;
    f_check = f_ML(index); 

    index =indexReal;
     %err = norm((f0(index) - f_check),2)/norm(f0(index),2);
     
     err = 0;
     indexes = [3,5,6,7,8,9];    
     for ii = 1:length(indexes)
%         bound =  abs(mechParams(ii) - referenceVals(ii));
        err = err + constraintWeight*(mechParams(indexes(ii)) - referenceVals(indexes(ii)))^2; 
        
%         if mechParams(ii)>referenceVals(ii)
%         err = err + 0.0005*bound;
%         else 
%         err = err + 0.0005*bound;
%         end
     end
%      
     %err = err + norm((weights.'.*(f0(index) - f_check))./f0(index),2)^2;
     for ii = 1:length(f_check)  
            err = err + ((f0(index(ii)) - f_check(ii))/f0(index(ii)))^2;      
     end
%      disp(err);
end

