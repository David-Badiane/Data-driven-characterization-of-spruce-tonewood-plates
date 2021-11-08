function [mechParamsMultilin] = computeParamsMultilin_plate(f0,rho, coeffs, algorithm)

intercepts = coeffs(:,1);
densities = coeffs(:,2);
GLRcoeff = coeffs(1,6);
ELcoeff  = coeffs(2,3);
ERcoeff  = coeffs(3,4);
GLTcoeffs = coeffs(:,8);

if strcmp(algorithm, 'solve')
    fiiCoeffs = coeffs(4,:);

    syms EL ER GLR GLT
    
    eq1 = 1/GLRcoeff*(-intercepts(1)- densities(1)*rho - GLTcoeffs(1)*GLT +f0(1)) == GLR;
    eq2 = 1/ELcoeff*(-intercepts(2)- densities(2)*rho - GLTcoeffs(2)*GLT +f0(2)) == EL;
    eq3 = 1/ERcoeff*(-intercepts(3)- densities(3)*rho - GLTcoeffs(3)*GLT +f0(3)) == ER;
    eq4 = 1/fiiCoeffs(8)*(-intercepts(4)- densities(4)*rho - fiiCoeffs(3)*EL -...
        fiiCoeffs(4)*ER- fiiCoeffs(6)+f0(4)) == GLT;
    [A,B] = equationsToMatrix([eq1, eq2, eq3,eq4], [EL, ER, GLR,GLT]);
    X = linsolve(A,B);
    mechParamsMultilin = eval(X(1:3));
    mechParamsMultilin = mechParamsMultilin(:).';
end

if strcmp(algorithm, 'coeffs')
    GLR = -intercepts(1)/GLRcoeff - densities(1)/GLRcoeff*rho + GLRcoeff^-1*f0(1);
    EL =  -intercepts(2)/ELcoeff - densities(2)/ELcoeff*rho + ELcoeff^-1*f0(2);
    ER =  -intercepts(3)/ERcoeff - densities(3)/ERcoeff*rho + ERcoeff^-1*f0(3);
    mechParamsMultilin = [EL, ER, GLR];
end
if strcmp(algorithm, 'raw')
    GLR = - 1.85e9 + 2.93e7*f0(1)  + 2.35e6*rho;
    EL = - 2.9e10 + 2.74e8*f0(2)  + 3.56e7*rho  ;  % EL = intercept  + w_1*rho + w_2*
    ER = - 2.15e9 + 1.29e7*f0(3)  + 2.71e6*rho ;
    mechParamsMultilin = [EL, ER, GLR];
end
end