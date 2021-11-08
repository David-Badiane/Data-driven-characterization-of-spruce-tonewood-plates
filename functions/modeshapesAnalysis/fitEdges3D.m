function [outPts, edgesXY, edgesYZ] = fitEdges3D(inPtsX, inPtsZ)
nEdges = 4;
figure(10); clf reset; hold on ;
axPts = 1:length(inPtsX(:,1));
mark = {'.' 'x' 'v' 'd'};
outPts = [];
edgesXY = struct('E1', [],'E2', [],'E3', [],'E4', [], 'E5', [],'E6', [],'E7', [],'E8', []);
edgesYZ = struct('E1', [],'E2', [],'E3', [],'E4', [], 'E5', [],'E6', [],'E7', [],'E8', []); 

nSamplesLin = 200000;

% -------------- X ------------------------
for ii = [1 2 3 0]
seeOut = axPts(mod(axPts,4) == ii);

xPts = linspace(min(inPtsX(seeOut,1))-10, 10+max(inPtsX(seeOut,1)),nSamplesLin);
yPts = polyval(polyfit(inPtsX(seeOut,1),inPtsX(seeOut,2),1).', xPts); 
zPts = polyval(polyfit(inPtsX(seeOut,1),inPtsX(seeOut,3),1).', xPts);

% fit
% figure(10);
% plot3(inPts(seeOut,1),inPts(seeOut,2),inPts(seeOut,3), mark{ii+1});
% plot3(xPts,yPts,zPts, 'lineWidth', 1.8);
outPts = [outPts; xPts(:) yPts(:) zPts(:)];
% result
figure(11); hold on;
plot3(xPts,yPts,zPts, '.', 'MarkerSize',2);
    if ii == 1, edgesXY.E3 = polyfit(inPtsX(seeOut,1),inPtsX(seeOut,2),1).'; end
    if ii == 2, edgesXY.E7 = polyfit(inPtsX(seeOut,1),inPtsX(seeOut,2),1).'; end
    if ii == 3, edgesXY.E5 = polyfit(inPtsX(seeOut,1),inPtsX(seeOut,2),1).'; end
    if ii == 0, edgesXY.E1 = polyfit(inPtsX(seeOut,1),inPtsX(seeOut,2),1).'; end
end


% add other edges

% --------------- Z ------------------
axPts = 1:length(inPtsZ(:,1));
edgeOrientation = {'up' 'down' 'up' 'down'};
orderIdxs = [1 2 0 3];
count = 1;

for ii = orderIdxs
seeOut = axPts(mod(axPts,4) == ii);

xx = inPtsZ(seeOut,1); 
yy = inPtsZ(seeOut,2); 
zz = inPtsZ(seeOut,3);

if ismember(ii, [1 0])
    zPtsMin = 0.3*max(zz);
    zPtsMax = 0.7*max(zz);
    zIdxs = find(zz > zPtsMin & zz < zPtsMax);
    zPts = linspace(min(zz),20+ max(zz),nSamplesLin);
    yPts = polyval(polyfit(inPtsZ(seeOut(zIdxs),3),inPtsZ(seeOut(zIdxs),2),1).', zPts); 
    xPts = polyval(polyfit(inPtsZ(seeOut(zIdxs),3),inPtsZ(seeOut(zIdxs),1),1).', zPts);
    xPoly = inPtsZ(seeOut(zIdxs),1);
    yPoly = inPtsZ(seeOut(zIdxs),2);
    zPoly = inPtsZ(seeOut(zIdxs),3);

else
    zPtsMin = min(zz);
    zIdxs = find(zz == zPtsMin);
    
    seeOutOtherEdge = axPts(mod(axPts,4) == orderIdxs(count-1));
    zzz = inPtsZ(seeOutOtherEdge,3);
    zPtsMin = min(zzz);
    zIdxsOtherEdge =  find(zzz==zPtsMin);
    
    zPoly = [inPtsZ(seeOut(zIdxs),3); inPtsZ(seeOutOtherEdge(zIdxsOtherEdge),3)];
    yPoly = [inPtsZ(seeOut(zIdxs),2); inPtsZ(seeOutOtherEdge(zIdxsOtherEdge),2)];
    xPoly = [inPtsZ(seeOut(zIdxs),1); inPtsZ(seeOutOtherEdge(zIdxsOtherEdge),1)];
    
    zPts = zPoly;
    yPts = linspace(min(yPoly)-20, 1.2*max(yPoly),2); 
    xPts = polyval(polyfit(yPoly,xPoly,1).', yPts);
end

% fit
figure(10); hold on;
plot3(xPoly,yPoly,zPoly, mark{ii+1});
plot3(xPts,yPts,zPts, 'lineWidth', 1.8);
outPts = [outPts; xPts(:) yPts(:) zPts(:)];
count = count + 1;
% result
figure(11); hold on;
plot3(xPts,yPts,zPts, 'lineWidth', 1.8);

    if ii == 1, edgesXY.E6 = polyfit(xPoly,yPoly,1).'; end
    if ii == 2, edgesXY.E2 = polyfit(xPoly,yPoly,1).'; end
    if ii == 0, edgesXY.E8 = polyfit(xPoly,yPoly,1).'; end
    if ii == 3, edgesXY.E4 = polyfit(xPoly,yPoly,1).'; end
end

figure(10)
legend('1', '1_L', '2','2_L','3','3_L','4','4_L');
figure(11)
legend('1_L','2_L','3_L','4_L');
edgesXY = [edgesXY.E1, edgesXY.E2, edgesXY.E3, edgesXY.E4,...
         edgesXY.E5, edgesXY.E6, edgesXY.E7, edgesXY.E8];
     
end