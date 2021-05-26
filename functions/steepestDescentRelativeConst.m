function [xpar, fval, fvals, fvals_freq, fvals_amp] = steepestDescentRelativeConst(f,x0, TolX, TolFun, alpha0, MaxIter)
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
fvals_freq = [];
fvals_amp = [];

for k = 1: MaxIter
% alpha = alpha0;
g = grad(f,x, true); % compute gradient 
g = g/norm(g); %gradient as a row vector
disp(num2str(g));
x = x - alpha*g.*x;
[fx, fx_freq, fx_amp] = feval(f,x);

% figure (100)
% hold on 
% plot(k, fx, '-o');
% pause(0.1);

fvals(k) = fx;
fvals_freq(k) = fx_freq;
fvals_amp(k) = fx_amp;

if warning >= 2||(norm(x - x0) < TolX&&abs(fx - fx0) < TolFun)
    break;
end
x0 = x; fx0 = fx;
end
xpar = x; fval = fx;
if k == MaxIter
    fprintf([num2str(MaxIter),' iterations'])
end

figure()
axis = 1:length(fvals);
subplot(131)
plot(axis, fvals, '-x')
xlabel('step N ')
ylabel('L_2')
subplot 132
plot(axis, fvals_freq, '-x')
xlabel('step N ')
ylabel('L_2_{freq}')
subplot 133
plot(axis, fvals_amp, '-x')
xlabel('step N ')
ylabel('L_2_{amp}')

end





function g = grad(f,x, relative, h) 
%grad.m  to get the gradient of f(x) at x.

   if relative 
       if nargin<4
        h = 0.0001;
       end
       deltas = diag(h*x);
       h2= 2*h.*x;
   else
       if nargin<4
        h=1e-7;
       end
       deltas = diag(h*ones(size(x)));
       h2= 2*h;
   end
  
g = zeros(size(x));
for n=1:length(x)
    delta = deltas(:,n);
    if relative
        g(n)= (feval(f,x+delta) - feval(f,x-delta))/h2(n);
    else
        g(n)= (feval(f,x+delta) - feval(f,x-delta))/h2;
    end
end
end
