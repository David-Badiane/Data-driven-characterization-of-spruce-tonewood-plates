function [NCC] = NCC(x,y)
%NCC this function computes the normalized cross correlation function
% between two signals 
% inputs  
%   x  (array) = simulated signal
%   y (array)  = measured signal
% outputs
%   NCC (float) = normalized cross correlation coefficient

if length(x(:,1)) == 1
    x = x.';
end
if length(y(:,1)) == 1
    y = y.';
end

NCC  = ( y'*x)/(norm(x,2)*norm(y,2));
end

