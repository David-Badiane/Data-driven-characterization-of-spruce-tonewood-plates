function [currentVals] = createAndSetParams(model, referenceVals,percDevs, pastVals, names, isPlate)
%CREATEANDSETPARAMS Summary of this function goes here
%   Detailed explanation goes here
    currentVals = zeros(size(referenceVals));
    token = false;
    while token == false

        for ii = (1:length(referenceVals))
            
            val = normrnd(referenceVals(ii), percDevs(ii)*referenceVals(ii));
            if isPlate
                if ii == 4
                    val = currentVals(3);
                end
            end
            currentVals(ii) = val;
            model.param.set(names(ii), val);
        end
        
        tmp = 0;
        avDist = zeros(size(pastVals(1,:)));
        for ii = 1:length(pastVals(1,:))
            pastTuple = pastVals(:,ii);
            avDist(ii) = sum(abs((currentVals - pastTuple)./currentVals))/length(currentVals);
            if(avDist(ii)<0.05)
               tmp = 20; 
               %disp(avDist(ii));
            end
        end

        if(tmp < 10)
            token = true;
        end
    end
end

