function [res] = inInterval(value, interval)
% check if the value(num) is in an interval ( 1D Array of length 2)
if value> interval(1) && value< interval(2)
   res = true;
else
    res = false;
end
end

