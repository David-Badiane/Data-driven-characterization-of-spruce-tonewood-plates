function [outPts] = getVertexesMeshX(pts, stepX,scaleMesh)
   pts = scaleMesh*pts; 
   x = pts(:,1); y = pts(:,2); z = pts(:,3);
   maxX = max(x)
   minX = min(x);
   maxY = max(y)
   minY = min(y);
   maxZ = max(z) 
   minZ = min(z);
   
   
   xVals = minX:stepX:maxX;
   nPts = 20;
   xAx = linspace(minX, maxX, nPts);
   yAx = linspace(minY, maxY, nPts);
   zAx = linspace(minZ, maxX, nPts);
   
   figure(4);
   edges = {};
   
   plotData = false; 
   
   outPts = [];
%    maxSearch = logspace(log10(0.001), log10(0.08), length(xVals)-1);
   maxSearch = 0.0075* ones(length(xVals)-1,1);
   
   for ii = 1:length(xVals)-1
        idxsX = find(x > xVals(ii) & x < xVals(ii+1));
        xX = x(idxsX); yX = y(idxsX); zX = z(idxsX);
        
        mM = [min(xX) max(xX) min(yX) max(yX) min(zX) max(zX)];
        ptsX = [xX, yX, zX];
%         figure(5); hold on
%         plot3(xX,yX,zX, '.');
        % xy plane
        if plotData ,     figure(1); clf reset; end
        % edge 1
        p1 = fitEdge(ptsX, 'y', 'down', mM(3:4), plotData,pts, maxSearch(ii));
        % edge 2
        p2 = fitEdge(ptsX, 'z', 'up', mM(5:6), plotData,pts, maxSearch(ii));
        % edge 3
        p3 = fitEdge(ptsX, 'y', 'up', mM(3:4), plotData,pts, maxSearch(ii));
        % edge 4
        p4 = fitEdge(ptsX, 'z', 'down', mM(5:6), plotData,pts, maxSearch(ii));
        
        edges = {p1 p2 p3 p4};

        for jj =1:length(edges)  
            A = [edges{jj}.YZ(1)                    -1;...
                 edges{mod(jj,length(edges))+1}.YZ(1)  -1];
            B = [-edges{jj}.YZ(2);...
                 -edges{mod(jj,length(edges))+1}.YZ(2)];
            sol = linsolve(A,B);
            outPts = [outPts; mean(xX) sol.' ]; 
        end
        
   if ii >= floor(0.99*length(xVals))
       a = 1;
   end   
   end
   
   figure(4);
    subplot(122); hold on; box on;
    plot3(outPts(:,1), outPts(:,2), outPts(:,3), 'lineWidth', 1.1)
    plot3(outPts(:,1), outPts(:,2), outPts(:,3), '.', 'MarkerSize', 10)
    view(3);
    xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]');
    ax = gca;
    ax.FontSize = 12;
    ax.LineWidth = 0.65;
end

function [p] = fitEdge(pts, variable, edge, minMax, plotData, allPts, maxSearch)
    p = struct('xAx', [], 'XY', []);
    
    idxFit = find(strcmp(variable, {'x' 'y' 'z'}));
    idxGet = [1,3,2];
    idxGet = idxGet(idxFit);
    nPoints = 20;
    
   if idxFit == 3 & strcmp(edge, 'down')
       idxMinorEdge = find(pts(:,2)<0.015*max(pts(:,2)));
       maxV = max(pts(idxMinorEdge,3));
       idxMinorEdge = find(pts(:,2)<0.015*max(pts(:,2)));
       minV = min(pts(idxMinorEdge,3));
       nPoints = 8;
   else 
       maxV = max(pts(:,idxFit));
       minV = min(pts(:,idxFit));
   end
    axsearch = linspace(minV, maxV, nPoints);
    
    edgeIdxs = [];
    deltaAx = axsearch(2)-axsearch(1);
    for ii = 1:length(axsearch)-1
       idxs = find(pts(:,idxFit)<axsearch(ii+1) & pts(:,idxFit)>axsearch(ii));
       if strcmp(edge, 'down')
           [mV,mL] = min(pts(idxs,idxGet));
        elseif strcmp(edge, 'up')
           [mV,mL] = max(pts(idxs,idxGet));
       end  
       edgeIdxs = [edgeIdxs; idxs(mL)];
%        figure(6); hold on;
%        plot3(pts(idxs,1),pts(idxs,2),pts(idxs,3), '.', 'markerSize', 0.4)
%        plot3(pts(edgeIdxs,1),pts(edgeIdxs,2),pts(edgeIdxs,3), '.', 'markerSize', 15)
%        view(90,0);
    end
    
    p.YZ = polyfit(pts(edgeIdxs,2),pts(edgeIdxs,3),1).';
    
    if  idxFit == 3
        zStep =  max(allPts(:,3))-min(allPts(:,3));
        zAx = linspace(min(allPts(:,3))-0.3*zStep, max(allPts(:,3))+0.3*zStep,100);
        yAx = polyval(polyfit(pts(edgeIdxs,3),pts(edgeIdxs,2),1).', zAx);
        if strcmp(edge, 'down') 
            yRange = abs(min(pts(edgeIdxs,2)) - max(pts(edgeIdxs,2)));
            yAx = linspace(min(pts(edgeIdxs,2)) -0.5*yRange,...
                max(pts(edgeIdxs,2))+0.5*yRange,100);  
        end

    end
    
    if idxFit == 2
        yStep = max(allPts(:,2))-min(allPts(:,2));
        yAx = linspace(min(allPts(:,2))-0.3*yStep, max(allPts(:,2))+0.3*yStep,100); 
    end
    
    p.yAx = yAx;
    pyz = polyval(p.YZ, yAx);
    if plotData
        figure(3); 
        hold on;
        plot3(mean(pts(:,1))*ones(size(pyz)), yAx, pyz);
    end
    
        if plotData 
        figure(1); 
%         clf reset;
        hold on;
        plot3(pts(edgeIdxs, 1),pts(edgeIdxs, 2),pts(edgeIdxs, 3),'.') ;
        xlabel('x'); ylabel('y'); zlabel('z');
%         plot3(pts(:,1), pts(:,2), pts(:,3), '.', 'markerSize', 1);
        plot3(mean(pts(:,1))*ones(size(pyz)),sort(yAx), pyz, 'lineWidth', 1.3);
        view(90,0)
        end
    
end


