function [NMSE] = NMSE(x,y)
%NMSE calculates the normalised mean square error between two signals 
NMSE = norm(x- y)^2 / norm(x - mean(x))^2;
end

