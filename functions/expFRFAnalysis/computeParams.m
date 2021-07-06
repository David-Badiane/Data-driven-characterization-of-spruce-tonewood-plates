function [mechParams, normParams] = computeParams(f0,rho, geom)
%COMPUTEPARAMS Summary of this function goes here
%   Detailed explanation goes here
D1 = 0.08006*rho*(geom(1)^4*f0(2)^2)/(geom(3)^2);
EL = 12*D1;
D3 = 0.08006*rho*(geom(2)^4*f0(3)^2)/(geom(3)^2);
ER = 12*D3;
D4 = 0.274* rho * (geom(1)^2*geom(2)^2*f0(1)^2)/(geom(3)^2);
GLR = 3*D4;
mechParams = [EL, ER, GLR];
normParams = mechParams/EL;
end

