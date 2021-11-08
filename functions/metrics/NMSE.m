function [NMSE] = NMSE(measSig,simSig)
%NMSE calculates the normalised mean square error
%   
% simSig  (array) = simulated signal
% measSig (array) = measured signal



NMSE = norm( measSig- simSig)^2/norm(measSig - mean(measSig))^2;


% figure(100)
% subplot(5,5,ii)
% hist(diff,20);
% title(outNames{ii});
% xlabel(' Y - Y*')
% ylabel('N ')

end

