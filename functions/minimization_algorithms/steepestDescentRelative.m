function [xpar, fval] = steepestDescentRelative(f,x0, TolX, TolFun, alpha0, MaxIter)
%STEEPESDESCENT Summary of this function goes here
%   f = function to optimize
%   x0 = initial guess
%   tolX = xTolerance
%   tolFun = f tolerance
%   alpha0 = initial step size ( relative )

% test function foo = @(x) (1-x(1)).^2 + (x(2) - x(1).^2).^2

if nargin < 6, MaxIter = 100; end %maximum # of iteration
if nargin < 5, alpha0 = 10; end %initial step size
if nargin < 4, TolFun = 1e-8; end %|f(x)| < TolFun wanted
if nargin < 3, TolX = 1e-6; end %|x(k)- x(k - 1)|<TolX wanted

x = x0; % initial guess
fx0 = feval(f,x0); % evaluate function at initial guess
fx = fx0; % init value of the function
alpha = alpha0; % init step size
kmax1 = 100; % number of iterations for line search
warning = 0; %the # of vain wanderings to find the optimum step size
fvals = []; % values of the function preallocation

    figure (100)
    hold on 
    plot(0, fx, '-o');
    pause(0.1);

    for k = 1: MaxIter
    alpha = alpha0;
    g = grad(f,x); % compute gradient 
    disp(num2str(g));
    fx1 = feval(f,x - 0.5*alpha*g.*x);

    
    for k1 = 1:kmax1    %find the optimum step size(alpha) by line search
        fx2 = fx1; 
        fx1 = feval(f,x - alpha*g.*x);
        
        disp(['step: ', num2str(alpha),' fx0: ', num2str(fx0), ' fx1: ', num2str(fx1),' fx2: ', num2str(fx2),' fx: ', num2str(fx)] );

        if fx0 > fx1+TolFun && fx1 < fx2 - TolFun %fx0 > fx1 < fx2
            alphaCheck = zeros(kmax1,1);
            fvalCheck = zeros(kmax1,1);
            for k2 = 1:kmax1*2
                alphaCheck(k2) = alpha*0.99^(k2-1);
                fvalCheck(k2) = feval(f,x - alphaCheck(k2)*g.*x); %Eq.(7.1.9)
                disp(['step: ', num2str(alphaCheck(k2)),' fx0: ', num2str(fx0), ' fx1: ', num2str(fvalCheck(k2)),' fx2: ', num2str(fx2)] );                
            end
            [minVal, minLoc] = min(fvalCheck);
            alpha = alphaCheck(minLoc);
            x = x - alpha*g.*x;
            fx = feval(f,x); %Eq.(7.1.9)
            disp(['chosen step: ', num2str(alpha), ' fx0: ', num2str(fx0), ' fx: ', num2str(fx)] );                

            break;
        else
            alpha = alpha*0.9;
            disp(['step: ', num2str(alpha),' fx0: ', num2str(fx0), ' fx1: ', num2str(fx1),' fx2: ', num2str(fx2)] );
            if warning >= 2||(abs(fx - fx0) < TolFun), searchStepSize = false; end
        end
    
    end
    if k1 >= kmax1
        warning = warning + 1; %failed to find optimum step size
    else
        warning = 0;
    end
    
%     figure (100)
%     hold on 
%     plot(k, fx, '-o');
%     pause(0.1);

    fvals(k) = fx;
    if warning >= 2||(norm(x - x0) < TolX&&abs(fx - fx0) < TolFun)
        break;
    end
    x0 = x; fx0 = fx;
    end
    
xpar = x; fval = fx;
if k == MaxIter
    fprintf('Just best in %d iterations',MaxIter), end
end


function g = grad(f,x,  h) 
%grad.m  to get the gradient of f(x) at x.

if nargin<3
h = 0.01;
end
deltas = diag(h*x);
h2 = 2*h.*x;
  
g = zeros(size(x));
for n=1:length(x)
    delta = deltas(:,n);
    if n == length(x)
        pause(1)
    end
    diff = feval(f,x+delta) - feval(f,x-delta);
    g(n)= (diff);%/h2(n);
    disp(g(n));
end
disp(num2str(g));
end
