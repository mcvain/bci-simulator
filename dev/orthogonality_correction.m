%% Sigmoid function design + Control vector correction preview 
% Input: equation/params for sigmoid function. 
% Output: 4 floats as weights for BCI2000 output channels
close all; 

v = -1:0.01:1;

order = {'C3 LR', 'C3 DU', 'C4 LR', 'C4 DU'};

% Used in Experiment configuration A 
gain = [5 5 -5 5];
shift = [0 0 0 0];

% Used in Experiment configuration B  
gain = [2 1 -.5 10];  % centered is 5 
shift = [0 0 0 0];  % centered is 0

% Used in Experiment configuration C 
%gain = [2 9 .5 7];
%shift = [-.5 0.3 .6 .7];  

figure;

for i = 1:4 
    y = 1 ./ (1 + exp(gain(i) * (v+shift(i)))); 
    subplot(2, 2, i); plot(v, y); 
    title(order{i}); 
    xlim([-1 1]); ylim([0 1]); 
end


[cl_new, cr_new, cd_new, cu_new, x_weights, y_weights] = control_axes_corrector(gain, shift, 'plot');

function [cl_new, cr_new, cd_new, cu_new, x_weights, y_weights] = control_axes_corrector(gain, shift, plot_mode)

    % LEFT 
    v = [-0.95 0]; 
    C3LR = 1 ./ (1 + exp(gain(1) * (v+shift(1))));
    C4LR = 1 ./ (1 + exp(gain(3) * (v+shift(3))));
    cl = [C4LR(1)-C3LR(1)-(C4LR(2)-C3LR(2)), -C4LR(1)-C3LR(1) - (-C4LR(2)-C3LR(2))];
    %cl = cl / norm(cl); 
        
    % RIGHT
    v = [0.95 0];
    C3LR = 1 ./ (1 + exp(gain(1) * (v+shift(1))));
    C4LR = 1 ./ (1 + exp(gain(3) * (v+shift(3))));
    cr = [C4LR(1)-C3LR(1)-(C4LR(2)-C3LR(2)), -C4LR(1)-C3LR(1) - (-C4LR(2)-C3LR(2))];
    %cr = cr / norm(cr); 
    
    % DOWN
    v = [-0.95 0]; 
    C3DU = 1 ./ (1 + exp(gain(2) * (v+shift(2))));
    C4DU = 1 ./ (1 + exp(gain(4) * (v+shift(4))));
    cd = [C4DU(1)-C3DU(1)-(C4DU(2)-C3DU(2)), -C4DU(1)-C3DU(1) - (-C4DU(2)-C3DU(2))]; 
    %cd = cd / norm(cd); 
    
    % UP 
    v = [0.95 0]; 
    C3DU = 1 ./ (1 + exp(gain(2) * (v+shift(2))));
    C4DU = 1 ./ (1 + exp(gain(4) * (v+shift(4))));
    cu = [C4DU(1)-C3DU(1)-(C4DU(2)-C3DU(2)), -C4DU(1)-C3DU(1) - (-C4DU(2)-C3DU(2))]; 
    %cu = cu / norm(cu); 
    
    dl = [-1 0];
    dr = [1 0]; 
    du = [0 1]; 
    dd = [0 -1]; 
    
    la = atan2d(det([cl', dl']), dot(cl', dl'));
    ra = atan2d(det([cr', dr']), dot(cr', dr')); 
    ua = atan2d(det([cu', du']), dot(cu', du'));
    da = atan2d(det([cd', dd']), dot(cd', dd'));
    theta = mean([la, ra, ua, da]); 
    theta_LR = mean([la, ra]); 
    theta_DU = mean([ua, da]); 
    
    R = [cosd(theta), -sind(theta); sind(theta), cosd(theta)]; % rotation matrix
    R_LR = [cosd(theta_LR), -sind(theta_LR); sind(theta_LR), cosd(theta_LR)]; 
    R_DU = [cosd(theta_DU), -sind(theta_DU); sind(theta_DU), cosd(theta_DU)]; 
    
    cl_new = R_LR * cl';
    cr_new = R_LR * cr'; 
    cd_new = R_DU * cd'; 
    cu_new = R_DU * cu'; 
    
    % Add the number weights calculation and return those as well! 
    x_weights = [cosd(theta_LR) + sind(theta_LR), sind(theta_LR) - cosd(theta_LR)]; % [C4, C3]
    y_weights = [sind(theta_DU) - cosd(theta_DU), -cosd(theta_DU) - sind(theta_DU)]; % [C4, C3]
    
    if nargin == 3
        if plot_mode == 'plot'
            figure;
            subplot(1, 2, 1); title('Before correction'); 
            xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
            p1 = plot([0, cr(1)], [0, cr(2)]); 
            p2 = plot([0, cu(1)], [0, cu(2)]); 
            p3 = plot([0, cl(1)], [0, cl(2)]); 
            p4 = plot([0, cd(1)], [0, cd(2)]); 
            px = xline(0, '--'); py = yline(0, '--');
            legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 
            
            subplot(1, 2, 2); title('After correction'); 
            xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
            p1 = plot([0, cr_new(1)], [0, cr_new(2)]); 
            p2 = plot([0, cu_new(1)], [0, cu_new(2)]); 
            p3 = plot([0, cl_new(1)], [0, cl_new(2)]); 
            p4 = plot([0, cd_new(1)], [0, cd_new(2)]); 
            px = xline(0, '--'); py = yline(0, '--');
            legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 
        end
    end
    
end