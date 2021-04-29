function id = HLSM (vpar, freq)
%FUNHJKI Summary of this function goes here
%   Detailed explanation goes here
 freq = 2*pi*freq;
 m = vpar(:,1);
 c = vpar(:,2);
 k = vpar(:,3);
 A = vpar(:,4);
 B = vpar(:,5);
 C = vpar(:,6);
 D = vpar(:,7);
 E = vpar(:,8);
 F = vpar(:,9);
 id = zeros(length(freq),1);

for ii = 1:length(freq) 
id(ii) = (A+1i*B)/(-(freq(ii)^2)*m + 1i* c * freq(ii) + k )...
    +(C + 1i*D) + ((E+ 1i*F) / (freq(ii)^2));
end
end

