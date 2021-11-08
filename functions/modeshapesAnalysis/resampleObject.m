function [pts] = resampleObject(inputMatrix, pX, pY, pZ)

% x = linspace(min(pts(:,1)), max(pts(:,1)), pX);
% y = linspace(min(pts(:,2)), max(pts(:,2)), pY);
% z = linspace(min(pts(:,3)), max(pts(:,3)), pZ);
% 
% [X,Y] = meshgrid(x,y)
% resPts = [];
% for kk = 1:pZ
%     Z = [];
%     for ii = 1:length(X(:,1))
%         for jj = 1:length(X(1,:))
%             dist = abs(z(kk) - pts(:,3))
%             [minVal, minLoc] = min();
%             disp(minVal)
%             if minVal < 5
%                 Z(ii,jj) = pts(minLoc,3);
%             else
%                 Z(ii,jj) = nan;
%             end
%         end
%     end
%     resPts = [resPts; X(:) Y(:) Z(:)];
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DOWNSAMPLING this fucntion performs donwnsapling over a rectangular    %%%%%%%%%%%%
%%%              grid on a matrix                                          %%%%%%%%%%%%

%%% INPUTS                                                                 %%%%%%%%%%%%
%%% inputMatrix = matrix to downsample (2DArray)                           %%%%%%%%%%%%
%%% pY    = target number of rows (double)                              %%%%%%%%%%%%
%%% pX    = target number of columns (double)                           %%%%%%%%%%%%
%%% fileName = fileName for saving [only the name, not .csv] (double)      %%%%%%%%%%%%
%%% saveData = true if you want to save data on .csv file (boolean)        %%%%%%%%%%%%

%%% OUTPUTS                                                                %%%%%%%%%%%%
%%% outMatrix = resampled matrix or array (2DArray)                        %%%%%%%%%%%%

x = inputMatrix(:,1); y = inputMatrix(:,2); z = inputMatrix(:,3);

maxX = max(x); minX = min(x);
maxY = max(y); minY = min(y);
maxZ = max(z); minZ = min(z);

% step size
deltaX = (maxX - minX)/pX;
deltaY = (maxY - minY)/pY;
deltaZ = (maxZ - minZ)/pZ;

xRect = linspace(minX,maxX,pX);
yRect = linspace(minY,maxY,pY);
zRect = linspace(minZ,maxZ,pZ);

[X,Y,Z] = meshgrid(xRect, yRect,zRect);
xRect = X(:);
yRect = Y(:);
zRect = Z(:);
pts = [];
for ii = 1:pY*pX*pZ
   [fval, floc] = min(sqrt((x - xRect(ii)).^2 + (y - yRect(ii)).^2 + (z - zRect(ii)).^2));
   if fval >  0.5*sqrt(deltaX^2 + deltaY^2+ deltaZ^2)
       pts = [pts; xRect(ii) yRect(ii) NaN ];
   else
       pts = [pts; xRect(ii) yRect(ii) z(floc)];
   end
   
end


figure(100)
plot3(pts(:,1), pts(:,2), abs(pts(:,3)),'.');
hold on 
plot3(xRect, yRect, zeros(size(xRect)), '.', 'markerSize', 0.5);
plot3(x,y,abs(z), 'x', 'markerSize', 0.01);
pause(0.1);
hold off;


outMatrix = pts;
% if saveData
%     writeMat2File(outMatrix, [fileName,'.csv'], {'x' 'y' 'z'}, 3, true );
% end
end

