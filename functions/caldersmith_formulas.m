function [mechParams, normParams] = caldersmith_formulas(f0,rho, geom, poissonRatio) 
% compute elastic constants of the plate with  caldersmith formulas
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
%   f0 = 3x1 double - characteristic eigenfrequencies of the plate 
%                     f0(1) = f11;  f0(2) = f02;  f0(3) = f20;
%   rho = 1x1 double density of the plate
%   geom = 3x1 double - geometry of the plate
%                       geom(1) = Length; geom(2) = Width; geom(3) = Thickness; 
%   poissonRatio = 1x1 double between 0 and 1 
% -------------------------------------------------------------------------
% outputs:
%   mechParams = 1x3 double --> [EL, ER, GLR] 
%                                mechParams(1) = longitudinal Young's modulus (EL); 
%                                mechParams(2) = radial Young's modulus (ER); 
%                                mechParams(3) = longitudinal to radial Shear modulus (GLR);
%                                normParams = 1x3 double --> mechParams/EL
% -------------------------------------------------------------------------

% if not specified, don't count Poisson's ratio
if nargin<4
    poissonRatio = 0;
end

% Caldersmith formulas
% E_L
D1 = 0.08006*rho*(geom(1)^4*f0(2)^2)/(geom(3)^2);
EL = 12*(1-poissonRatio.^2)*D1;
% E_R
D3 = 0.08006*rho*(geom(2)^4*f0(3)^2)/(geom(3)^2);
ER = 12*(1-poissonRatio.^2)*D3;
% G_LR
D4 = 0.274* rho * (geom(1)^2*geom(2)^2*f0(1)^2)/(geom(3)^2);
GLR = 3*D4;

mechParams = [EL, ER, GLR];
normParams = mechParams/EL;
end

