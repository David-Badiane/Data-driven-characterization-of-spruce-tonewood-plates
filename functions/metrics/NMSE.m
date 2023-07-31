function [NMSE] = NMSE(x,y)
% calculates the normalised mean square error between two signals 
% ---------------------------------------------------------------------
% INPUTS: 
% x = signal, in our case measured signal
% y = signal, in our case simulated signal
% ---------------------------------------------------------------------
% NMSE = normalized mean squared error (NMSE) between x and y
NMSE = norm(x- y)^2 / norm(x)^2;
end

