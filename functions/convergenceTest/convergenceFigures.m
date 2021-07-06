function [] = convergenceFigures(eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD )
%CONVERGENCEFIGURES Summary of this function goes here
%   Detailed explanation goes here

    % EIGENFREQUENCIES WITH VALUES FIGURE
    figure()
    hold on 
    for ii = 1:length(eigenFreqzMatrix(:,1))
        plot(1:length(eigenFreqzMatrix(1,:)), eigenFreqzMatrix(ii,:),'-o')
    end
    legend('C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++');
    xlabel('N mode'); ylabel('f   [Hz]'); title('Eigs variating mesh size');
    xlim([0.9, length(eigRate) + 0.1]);
    
    % EIGENFREQUENCY ERROR TREND FIGURE
    figure()
    hold on;
    plot(1:length(eigRate), eigRate)
    for ii = 1:length(eigRate)
       plot(ii, eigRate(ii), '.', 'markerSize' , 10);
       text(ii , eigRate(ii)-0.1 , [num2str(round(timesEig(ii))),'s']);
    end
    legend('err line', 'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++');
    xlabel('mesh size'); ylabel('mean normalised error [%]'); title('Eigs mean norm error variating mesh size');
    ylim([-0.5,1.1*max(eigRate)]);
    xlim([0.9, length(eigRate) + 0.1]);
    
    pointVelMatrix = abs(pointVelMatrix);
    % Amplitude Figure with values
    figure()
    hold on 
    for ii = 1:length(pointVelMatrix(:,1))
        FDRate(ii) = mean((abs(pointVelMatrix(ii,:)) - abs(pointVelMatrix(end,:)) )./abs( pointVelMatrix(end,:)) )*100;
        plot(1:length(pointVelMatrix(1,:)), pointVelMatrix(ii,:),'-o')
    end
    legend('C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++');
    xlabel('N mode'); ylabel('f   [Hz]'); title('amplitudes variating mesh size');
    xlim([0.9, length(FDRate) + 0.1]);

    % Amplitude Error trend 
    figure()
    hold on;
    plot(1:length(FDRate), FDRate)
    for ii = 1:length(FDRate)
        
       plot(ii, FDRate(ii), '.', 'markerSize' , 10);
       text(ii , FDRate(ii)+0.1, [num2str(round(timesFD(ii))),'s']);
    end
    legend('err line', 'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++');
    xlabel('mesh size'); ylabel('mean normalised error [%]');
    title('amplitude mean norm error variating mesh size');

    xlim([0.9, length(FDRate) + 0.1]);
    
    
end

