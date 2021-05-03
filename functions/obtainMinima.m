function [mins] = obtainMinima(meshData,modesData, minPeakVal, plotData, nModes)
%OBTAINMINIMA Summary of this function goes here
%   Detailed explanation goes here
nModes = min(size(modesData));
mins = cell(nModes,4);

for ii = 1:nModes   
    idxX = find(meshData(:,2) == 0);                        % y=0 ---> x axis
    edgeX = lowpass(modesData(idxX,ii), 0.65);
    [minValsX,minLocsX] = findpeaks(1./edgeX, 'MinPeakWidth', 2, 'MinPeakProminence', minPeakVal);
    xPoints =1:length(edgeX);
    mins{ii,1} = minValsX;
    mins{ii,2} = minLocsX/xPoints(end);

    idxY = find(meshData(:,1) == 0);                        % x= 0 --->  y axis
    edgeY = lowpass(modesData(idxY,ii),0.65);
    [minValsY,minLocsY] = findpeaks(1./edgeY, 'MinPeakWidth', 1, 'MinPeakProminence', 1e7);
    yPoints =1:length(edgeY);
    mins{ii,3} = minValsY;
    mins{ii,4} = minLocsY/yPoints(end);


    if plotData
%         figure(3)
%         subplot(5,5,ii)
%         plot(xPoints, edgeX, xPoints(minLocsX),edgeX(minLocsX),'r*');
%         xlabel('Num el')
%         ylabel('disp')
%         title(['mode', num2str(ii), '  y = 0'])
% 
%         figure(4)
%         subplot(5,5,ii)
%         plot(yPoints , edgeY, yPoints(minLocsY), edgeY(minLocsY),'r*')
%         xlabel('Num el')
%         ylabel('disp')
%         title(['mode', num2str(ii), '  x = 0'])
%           if ii ==2
%               
% %               posX = minLocsX(1)/xPoints(end);
% 
% %               if (posX < 0.17 || posX > 0.23)||(isempty(posY))
%                 if mins{ii,2}>0.25
%                   mins{ii,2} = nan;
%                   figure(3)
%                   plot3(meshData(:,1), meshData(:,2), modesData(:,ii) ,'.', 'MarkerSize', 10);
% 
%                    subplot(1,2,1)
% %                   plot(yPoints , edgeY, yPoints(minLocsY), edgeY(minLocsY),'r*');
% %                   subplot(1,2,2)
% %                   plot(xPoints, edgeX, xPoints(minLocsX),edgeX(minLocsX),'r*');
%               end
%           end
    end
end
end

 