function [xpar, fval, fvals, fvals_freq, fvals_amp] = steepestDescentRelativeConst(f,x0, TolX, TolFun, alpha0, MaxIter)
%STEEPESDESCENT Summary of this function goes here
%   f = objective function
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
[fx0,a] = feval(f,x0); % evaluate function at initial guess
nPoints = length(a);
fx = fx0; % init value of the function
alpha = alpha0; % init step size
kmax1 = 50; % number of iterations for line search
warning = 0; %the # of vain wanderings to find the optimum step size
fvals = zeros(MaxIter,1); % values of the function preallocation

fvals_freq = zeros(MaxIter,nPoints);
fvals_amp = zeros(MaxIter,nPoints);
h = 0.01;

for k = 1: MaxIter
    alpha = alpha*0.99;
    h = h*0.99;
    
    g = grad(f,x, h); % compute gradient 
    %g = g/max(abs(g));   % to normalize gradient
  
    x = x - alpha*g.*x;
    [fx, fx_freq] = feval(f,x);


    disp(['chosen step: ', num2str(alpha), ' fx0(previous): ', num2str(fx0), ' fx: (present) ', num2str(fx)] );                

    fvals(k) = fx;
    fvals_freq(k,:) = fx_freq;

    if warning >= 2 || (norm(x - x0)/norm(x) < TolX)
        disp([num2str(k),' iterations'])
        break;
    end
    figure (100)
    hold on 
    plot(k, fx, '-o');
    pause(0.1);

    x0 = x; fx0 = fx;
end

xpar = x; fval = fx;
if k == MaxIter
    disp([num2str(MaxIter),' iterations'])
end

figure()
axis = 1:length(fvals);
subplot(121)
plot(axis, fvals, '-o', 'markersize', 5, 'lineWidth', 1.1)
xlabel('step N ')   
ylabel('L_2')
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.TickDir = 'out';
ax.FontSize = 15;

subplot 122
plot(axis, fvals_freq, '-o', 'markersize', 5, 'lineWidth', 1.1);
xlabel('step N ')
ylabel('L_2_{freq}')
legend('f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8');
colorAxis1 = linspace(0.3,1,length(fvals_freq(1,:)) ).';
colorAxis2 = linspace(0.5,1,length(fvals_freq(1,:)) ).';
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.TickDir = 'out';
ax.FontSize = 15;
mycolor = [ zeros(size(colorAxis1)) colorAxis1 colorAxis2  ];
set(gca, 'ColorOrder', mycolor);

% subplot 133
% p3 = plot(axis, fvals_amp, '-x')
% legend('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a9');
% xlabel('step N ')
% ylabel('L_2_{amp}')



end


function g = grad(f,x,h) 
%grad.m  to get the gradient of f(x) at x.

if nargin<3
h = 0.01;
end
deltas = diag(h*x);
h2 = 2*h.*x;
  
g = zeros(size(x));
for n=1:length(x)
    delta = deltas(:,n);
    diff = feval(f,x+delta) - feval(f,x-delta);
    g(n)= (diff);%/h2(n);
end
end
