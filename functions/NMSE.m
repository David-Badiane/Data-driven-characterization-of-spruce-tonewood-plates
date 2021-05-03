function [NMSE] = NMSE(measSig,simSig,ii)
%NMSE calculates the normalised mean square error
%   
% simSig  (array) = simulated signal
% measSig (array) = measured signal

% simSig = simSig/max(simSig);
% measSig = measSig/max(measSig);
%NMSE = norm(measSig - simSig, 2)/norm(measSig,2) * 100;
% NMSE = sum((measSig - simSig)./measSig);
avgSimSig  = mean(simSig); 
avgMeasSig = mean(measSig);
diff = (measSig/avgMeasSig - simSig/avgSimSig);

NMSE = 1/length(measSig) * sum((diff).^2)*100;

figure(100)
subplot(5,5,ii)
hist(diff,20);
title(['f', num2str(ii)]);
xlabel(' Y - Y*')
ylabel('N ')

end

