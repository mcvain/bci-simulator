% Decoder rotation 
% Hyonyoung Shin 
% January 2022 

%% Initialization 
clear; close all; 
try
    addpath('C:\Users\helab\Documents\eeglab2021.1'); 
catch
    addpath('C:\Users\mcvai\eeglab2020_0');
end
eeglab; 
output_param_dir = 'rotated2D.prm'; 

% Parameters 
sampling_rate = 250; 
baseline_method = 3;

% Define channels from BCI2000
C3 = 7; C3_laplacians = [3, 6, 8, 11];
C4 = 9; C4_laplacians = [5, 8, 10, 13];

% Input dataset from calibration task. 
[EEG_LR, command] = pop_loadset('LR_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220120');
[EEG_UD, command] = pop_loadset('UD_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220120');

%% Processing 
i = 0; 
dataset = {EEG_LR, EEG_UD};
for axis = 1:2 
    EEG = dataset{axis};
    feedback_durations = [EEG.event(contains({EEG.event(:).type},'Feedback')).duration];
    target_durations = [EEG.event(contains({EEG.event(:).type},'TargetCode')).duration];
    
    EEG.event = EEG.event(~contains({EEG.event(:).type},'boundary'));
    EEG.event = EEG.event(~contains({EEG.event(:).type},'Feedback'));
    event_sequence = {EEG.event(:).type}; 
    target_event_idxs = find(contains(event_sequence,'TargetCode'));
    result_event_idxs = find(contains(event_sequence,'ResultCode'));
    
    % Exception handling 
    if length(target_durations) - 1 == length(feedback_durations)
        target_durations = target_durations(1:end-1); 
        target_event_idxs = target_event_idxs(1:end-1); 
    end
    
    PTC =  length(result_event_idxs) / length(target_event_idxs);
    
    TargetPresent = nan(length(target_event_idxs),1); 
    CorrectDirection = nan(length(target_event_idxs),1); 
    InputDirection = nan(length(target_event_idxs),1); 
    
    for t = 1:length(target_event_idxs)
        TargetPresent(t) = EEG.event(target_event_idxs(t)).latency;
        correct = EEG.event(target_event_idxs(t)).position; 
        if ~ismember(target_event_idxs(t)+1, result_event_idxs)
            input = 0; % invalid trial (subject did not hit a target)
        else
            input = EEG.event((target_event_idxs(t)+1)).position;
        end
        CorrectDirection(t) = correct;
        InputDirection(t) = input; 
    end
    BaselineStart = TargetPresent - 500;
    FeedbackStart = TargetPresent + 500;
    
    if length(FeedbackStart) + 1 ~= length(feedback_durations)  % super edge case handling
        FeedbackEnd = FeedbackStart + feedback_durations'; 
    else 
        FeedbackEnd = FeedbackStart + feedback_durations(1:length(FeedbackStart))'; 
    end
    TargetEnd = TargetPresent + target_durations'; 
    
    temp = [CorrectDirection, InputDirection, BaselineStart, TargetPresent, FeedbackStart, FeedbackEnd, TargetEnd];
    trial_data = array2table(temp,...
    'VariableNames', {'CorrectDirection', 'InputDirection', 'BaselineStart', 'TargetPresent', 'FeedbackStart', 'FeedbackEnd', 'TargetEnd'});

    % Direction codes verified. 
    LD_trials = find((trial_data.CorrectDirection == 2) & (trial_data.CorrectDirection == trial_data.InputDirection));
    UR_trials = find((trial_data.CorrectDirection == 1) & (trial_data.CorrectDirection == trial_data.InputDirection));
    trials_by_dir = {LD_trials, UR_trials};
    
    for dir = 1:2  % 2 directions in a given axis. 
        trials = trials_by_dir{dir}; 
        for tr = 1:length(trials)
            trial = trials(tr);
            i = i + 1; 
            eeg_baseline = EEG.data(:, trial_data.BaselineStart(trial):trial_data.TargetPresent(trial)-1);
            eeg_control = EEG.data(:, trial_data.FeedbackStart(trial):trial_data.FeedbackEnd(trial));

            if length(eeg_control) >= 120  % If using pwelch, feedback control period must be sufficiently long, but if using bandpower, this conditional is unnecessary.   
                % Spatial filter
                C3_data_control = double(eeg_control(C3,:) - 0.25 * eeg_control(C3_laplacians(1),:) - 0.25 * eeg_control(C3_laplacians(2),:) - 0.25 * eeg_control(C3_laplacians(3),:) - 0.25 * eeg_control(C3_laplacians(4),:));
                C4_data_control = double(eeg_control(C4,:) - 0.25 * eeg_control(C4_laplacians(1),:) - 0.25 * eeg_control(C4_laplacians(2),:) - 0.25 * eeg_control(C4_laplacians(3),:) - 0.25 * eeg_control(C4_laplacians(4),:));
                C3_data_baseline = double(eeg_baseline(C3,:) - 0.25 * eeg_baseline(C3_laplacians(1),:) - 0.25 * eeg_baseline(C3_laplacians(2),:) - 0.25 * eeg_baseline(C3_laplacians(3),:) - 0.25 * eeg_baseline(C3_laplacians(4),:));
                C4_data_baseline = double(eeg_baseline(C4,:) - 0.25 * eeg_baseline(C4_laplacians(1),:) - 0.25 * eeg_baseline(C4_laplacians(2),:) - 0.25 * eeg_baseline(C4_laplacians(3),:) - 0.25 * eeg_baseline(C4_laplacians(4),:));

                welch_window_size = 120;
                wws = min([welch_window_size, size(eeg_control,2)]);
                [C3_control_averaged_PSD, F] = pwelch(C3_data_control, wws, floor(wws/2), 4096, 250, 'onesided', 'power');
                [C4_control_averaged_PSD, F] = pwelch(C4_data_control, wws, floor(wws/2), 4096, 250, 'onesided', 'power'); 
                [C3_baseline_averaged_PSD, F] = pwelch(C3_data_baseline, wws, floor(wws/2), 4096, 250, 'onesided', 'power');
                [C4_baseline_averaged_PSD, F] = pwelch(C4_data_baseline, wws, floor(wws/2), 4096, 250, 'onesided', 'power');
                [~, alpha_start_index] = (min(abs(F - 8))); 
                alpha_start_index = alpha_start_index - 1;
                [~, alpha_end_index] = (min(abs(F - 14))); 
                alpha_end_index = alpha_end_index + 1; % verified to be working as intended as long as sampling freq (fs) is defined in the power esimation function e.g. pwelch, pburg. 
                C3_control_alpha = mean(C3_control_averaged_PSD(alpha_start_index:alpha_end_index));
                C4_control_alpha = mean(C4_control_averaged_PSD(alpha_start_index:alpha_end_index));
                C3_baseline_alpha = mean(C3_baseline_averaged_PSD(alpha_start_index:alpha_end_index));
                C4_baseline_alpha = mean(C4_baseline_averaged_PSD(alpha_start_index:alpha_end_index)); 

                % Method 1: Baseline at channel power level - CONDEMNED
                if baseline_method == 1
                relative_C3 = C3_control_alpha - C3_baseline_alpha; 
                relative_C4 = C4_control_alpha - C4_baseline_alpha; 
                control_data_table(i, 1) = relative_C4 - relative_C3; 
                control_data_table(i, 2) = -(relative_C4 + relative_C3); 
                end

                % Method 2: Baseline at control signal level - CONDEMNED
                if baseline_method == 2
                control_signal_LR_feedback = C4_control_alpha - C3_control_alpha; 
                control_signal_UD_feedback = - (C4_control_alpha + C3_control_alpha); 
                control_signal_LR_baseline = C4_baseline_alpha - C3_baseline_alpha; 
                control_signal_UD_baseline = - (C4_baseline_alpha + C3_baseline_alpha);
                control_data_table(i, 1) = control_signal_LR_feedback - control_signal_LR_baseline; 
                control_data_table(i, 2) = control_signal_UD_feedback - control_signal_UD_baseline;  
                end

                % Method 3: No baselining 
                if baseline_method == 3
                control_data_table(i, 1) = C4_control_alpha - C3_control_alpha; 
                control_data_table(i, 2) = - (C4_control_alpha + C3_control_alpha); 
                end

                if dir == 1 % LEFT or DOWN
                    if axis == 1 % LEFT
                        control_data_table(i, 3) = -1; 
                        control_data_table(i, 4) = nan; 
                    elseif axis == 2 % DOWN
                        control_data_table(i, 3) = nan; 
                        control_data_table(i, 4) = -1; 
                    end
                elseif dir == 2 % RIGHT or UP
                    if axis == 1 % RIGHT
                        control_data_table(i, 3) = 1; 
                        control_data_table(i, 4) = nan;
                    elseif axis == 2 % UP
                        control_data_table(i, 3) = nan; 
                        control_data_table(i, 4) = 1;
                    end
                end
            end

        end
            
    end
end

%% Rotation 
encoding_models = cell(1, 6); 
encoding_models{1} = control_data_table(:, 1);  
encoding_models{2} = control_data_table(:, 2);  
encoding_models{3} = control_data_table(:, 3);  
encoding_models{4} = control_data_table(:, 1);  
encoding_models{5} = control_data_table(:, 2);  
encoding_models{6} = control_data_table(:, 4);  

hor_x = encoding_models{1};
ver_x = encoding_models{2};
x_vel = encoding_models{3};
hor_y = encoding_models{4};
ver_y = encoding_models{5};
y_vel = encoding_models{6};

% Read off from subject-specific control pattern curves 
hor_x_avg_power = [mean(hor_x(x_vel == -1), 'omitnan'), mean(hor_x(x_vel == 0), 'omitnan'), mean(hor_x(x_vel == 1), 'omitnan')];
hor_y_avg_power = [mean(hor_y(y_vel == -1), 'omitnan'), mean(hor_y(y_vel == 0), 'omitnan'), mean(hor_y(y_vel == 1), 'omitnan')]; 
ver_x_avg_power = [mean(ver_x(x_vel == -1), 'omitnan'), mean(ver_x(x_vel == 0), 'omitnan'), mean(ver_x(x_vel == 1), 'omitnan')]; 
ver_y_avg_power = [mean(ver_y(y_vel == -1), 'omitnan'), mean(ver_y(y_vel == 0), 'omitnan'), mean(ver_y(y_vel == 1), 'omitnan')];

% Vx = 1, Vy = 0 [RIGHT]
hor_prime = hor_x_avg_power(3);
ver_prime = ver_x_avg_power(3);
control_space_vector_A = [hor_prime, ver_prime];
control_space_vector_A = control_space_vector_A / norm(control_space_vector_A); 

% Vx = 0, Vy = 1 [UP]
hor_prime = hor_y_avg_power(3);
ver_prime = ver_y_avg_power(3);
control_space_vector_B = [hor_prime, ver_prime];
control_space_vector_B = control_space_vector_B / norm(control_space_vector_B);

% Vx = -1, Vy = 0 [LEFT]
hor_prime = hor_x_avg_power(1);
ver_prime = ver_x_avg_power(1);
control_space_vector_C = [hor_prime, ver_prime];
control_space_vector_C = control_space_vector_C / norm(control_space_vector_C);

% Vx = 0, Vy = -1 [DOWN] 
hor_prime = hor_y_avg_power(1);
ver_prime = ver_y_avg_power(1);
control_space_vector_D = [hor_prime, ver_prime];
control_space_vector_D = control_space_vector_D / norm(control_space_vector_D);

% These plots are the unit control axes in the 2D BCI screen space projected
% onto the subject's horizontal and vertical control feature space. 
subplot(1, 2, 1); title('Before correction'); 
xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
p1 = plot([0, control_space_vector_A(1)], [0, control_space_vector_A(2)]); 
p2 = plot([0, control_space_vector_B(1)], [0, control_space_vector_B(2)]); 
p3 = plot([0, control_space_vector_C(1)], [0, control_space_vector_C(2)]); 
p4 = plot([0, control_space_vector_D(1)], [0, control_space_vector_D(2)]); 
xline(0, '--'); yline(0, '--'); 
legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 

subplot(1, 2, 2); title('After correction'); 
xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
[C, A, D, B, la, ra, ua, da, theta, R, x_weights, y_weights] = control_axes_corrector(control_space_vector_C, control_space_vector_A, control_space_vector_D, control_space_vector_B);
p1 = plot([0, A(1)], [0, A(2)]); 
p2 = plot([0, B(1)], [0, B(2)]); 
p3 = plot([0, C(1)], [0, C(2)]); 
p4 = plot([0, D(1)], [0, D(2)]); 
xline(0, '--'); yline(0, '--'); 
legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 

disp('Calculated the following rotation weights:'); 
disp(strcat('x_C4: ', num2str(x_weights(1)), ' x_C3:', num2str(x_weights(2))));
disp(strcat('y_C4: ', num2str(y_weights(1)), ' y_C3:', num2str(y_weights(2))));

function [cl_new, cr_new, cd_new, cu_new, la, ra, ua, da, theta, R, x_weights, y_weights] = control_axes_corrector(cl, cr, cd, cu) 
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
    
end