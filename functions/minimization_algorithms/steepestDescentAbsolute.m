function [xpar, fval] = steepestDescentAbsolute(f,x0, TolX, TolFun, alpha0, MaxIter)
%STEEPESDESCENT Summary of this function goes here
%   Detailed explanation goes here

% test function foo = @(x) (1-x(1)).^2 + (x(2) - x(1).^2).^2

if nargin < 6, MaxIter = 100; end %maximum # of iteration
if nargin < 5, alpha0 = 10; end %initial step size
if nargin < 4, TolFun = 1e-8; end %|f(x)| < TolFun wanted
if nargin < 3, TolX = 1e-6; end %|x(k)- x(k - 1)|<TolX wanted

x = x0; % initial guess
fx0 = feval(f,x0); % evaluate function at initial guess
fx = fx0; % init value of the function
alpha = alpha0; % init step size
kmax1 = 50; % number of iterations for line search
warning = 0; %the # of vain wanderings to find the optimum step size
fvals = []; % values of the function preallocation

for k = 1: MaxIter
    % alpha = alpha0;
    g = grad(f,x); % compute gradient 
    g = g/norm(g); %gradient as a row vector
    disp(num2str(g));
    alpha = alpha*2; %for trial move in negative gradient direction
    fx1 = feval(f,x - alpha*2*g);   
    
    for k1 = 1:kmax1    %find the optimum step size(alpha) by line search
    fx2 = fx1; 
    fx1 = feval(f,x-alpha*g);
    disp(['step: ', num2str(alpha),' fx0: ', num2str(fx0), ' fx1: ', num2str(fx1),' fx2: ', num2str(fx2),' fx: ', num2str(fx)] );

    if fx0 > fx1+TolFun && fx1 < fx2 - TolFun %fx0 > fx1 < fx2
        den = 4*fx1 - 2*fx0 - 2*fx2; 
        num = den - fx0 + fx2; %Eq.(7.1.5)
        alpha = alpha*num/den;
        x = x - alpha*g;
        fx = feval(f,x); %Eq.(7.1.9)
        disp(['step: ', num2str(alpha),' fx0: ', num2str(fx0), ' fx1: ', num2str(fx1),' fx2: ', num2str(fx2),' fx: ', num2str(fx)] );
        break;
    
    else
        alpha = alpha*0.85;
        disp(num2str(x));
        disp(['step: ', num2str(alpha),' fx0: ', num2str(fx0), ' fx1: ', num2str(fx1),' fx2: ', num2str(fx2)] );
        if warning >= 2||(abs(fx - fx0) < TolFun), searchStepSize = false; end
    end
    end
    if k1 >= kmax1
        warning = warning + 1; %failed to find optimum step size
    else
        warning = 0;
    end

figure (100)
hold on 
plot(k, fx, '-o');
pause(0.1);

fvals(k) = fx;

if warning >= 2||(norm(x - x0) < TolX&&abs(fx - fx0) < TolFun), break; end
x0 = x; fx0 = fx;
end
xpar = x; fval = fx;
if k == MaxIter
    fprintf('Just best in %d iterations',MaxIter), end
end


function g = grad(f,x, h) 
%grad.m  to get the gradient of f(x) at x.

   if nargin<3
    h=.0000001;
   end
   deltas = diag(h*ones(size(x)));
   h2= 2*h;
  
    g = zeros(size(x));
    for n=1:length(x)
        delta = deltas(:,n);
            g(n)= (feval(f,x+delta) - feval(f,x-delta))/h2;
    end
end
