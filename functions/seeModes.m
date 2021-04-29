function [mins] = seeModes( minPeakVal, modeNumber, tuples)
%SEEMODES Summary of this function goes here
%   Detailed explanation goes here
mins = cell(length(tuples),4);

for ii = 1:length(tuples)
meshData = table2array(readtable(['mesh',int2str(tuples(ii)),'.csv']));
modesData = table2array(readtable(['modeshapes',int2str(tuples(ii)),'.csv']));

idxX = find(meshData(:,2) == 0);                        % y=0 ---> x axis
edgeX = lowpass(modesData(idxX,modeNumber), 0.65);
[minValsX,minLocsX] = findpeaks(1./edgeX, 'MinPeakWidth', 2, 'MinPeakProminence', minPeakVal);

xPoints =1:length(edgeX);
mins{ii,1} = minValsX;
mins{ii,2} = minLocsX/xPoints(end);

idxY = find(meshData(:,1) == 0);                        % x= 0 --->  y axis
edgeY = lowpass(modesData(idxY,modeNumber),0.65);
[minValsY,minLocsY] = findpeaks(1./edgeY, 'MinPeakWidth', 1, 'MinPeakProminence', 1e7);
yPoints =1:length(edgeY);
mins{ii,3} = minValsY;
mins{ii,4} = minLocsY/yPoints(end);

figure(4)
subplot(4,2,2*ii)
plot(xPoints, edgeX, xPoints(minLocsX),edgeX(minLocsX),'r*');
xlabel('N element in the edge')
ylabel('solid.disp')
title(['mode', num2str(modeNumber), '  y = 0   tuple ', num2str(tuples(ii))])


subplot(4,2,2*ii-1)
plot(yPoints , edgeY, yPoints(minLocsY), edgeY(minLocsY),'r*')
xlabel('N element in the edge')
ylabel('solid.disp')
title(['mode', num2str(modeNumber), '  x = 0   tuple ', num2str(tuples(ii))])
end

end

