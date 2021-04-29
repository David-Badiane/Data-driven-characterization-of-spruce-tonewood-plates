function [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo, f0,minLoc,rho, indexComsol, indexReal, plotImg)
%MINIMIZEERROR Summary of this function goes here
%   Detailed explanation goes here


nModes = length(outputsALLInfo(1,:));
inputParsStart = inputsInfo(minLoc,2:end);


constraintWeight = 0.1;
fun = @(x) errorMultiLin(x, f0,linearModels, nModes,rho,inputParsStart',indexComsol, indexReal, constraintWeight);

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-15,'TolX',1e-15, 'MaxFunEvals',10e3,'MaxIter', 10e3,...
    'DiffMinChange', 1, 'DiffMaxChange', 200); 

% minimization
%[xpar,fval,exitflag,output] = fminsearchbnd(fun,inputParsStart',2*referenceVals(2:end)',1/3*referenceVals(2:end)',options)
[xpar,fval, exitflag, output] = fminsearch(fun, inputParsStart', options)
%optParamsTable  = writeMat2File(xpar','optInputs.csv', varyingParamsNames, 10,true);   
xpar = [rho; xpar];
[f_out] = eigfreqzFromCoefficients(xpar, linearModels, nModes);
f_disp = [f_out(indexComsol)];
if plotImg 
figure()
plot(1:length(f_disp), f_disp, '-o');
hold on 
plot(1:length(f0(indexReal)), f0(indexReal), '-x');
xlabel('N mode')
ylabel(' f     [Hz]');
xlim([1,length(f_disp)]);
legend('fMultilin', 'fexp');
end
%fComsol = [73.595; 101.31; 169.97; 178.77; 223.83; 280.05; 353.43; 355.03; 463.07; 511.75];
%fComsolDisp = [fComsol(2), fComsol(3), fComsol(5), fComsol(6), fComsol(10)];
%plot(1:length(fComsolDisp), fComsolDisp, '-o')


end

