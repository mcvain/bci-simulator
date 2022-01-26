% Rewrite calculate_rotation with baselining wrt v = 0 which can be taken
% from trials in the other axis!! for e.g. vx=0 is from UD trials. 
% In other words, this script is a translation of
% orthogonality_correction.m but for real EEG data. 
% 
% Hyonyoung Shin 
% January 2022 

% Dependencies: 
% 1. EEGLAB toolbox. 
% 2. bandpowers.m, num2varstr.m, input_interp.m from CTAP toolbox. 

%% Initialization 
clear; close all; 
try
    addpath('C:\Users\helab\Documents\eeglab2021.1'); 
catch
    addpath('C:\Users\mcvai\eeglab2020_0');
end
eeglab; 
% output_param_dir = 'rotated2D.prm'; 

% Parameters 
sampling_rate = 250; 
baseline_method = 3;
timeout_trial_weight = 1; 
alphaband_definition = [8 13];
notch_filter = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',sampling_rate);

% Define channels from BCI2000
C3 = 7; C3_laplacians = [3, 6, 8, 11];
C4 = 9; C4_laplacians = [5, 8, 10, 13];

% Input dataset from calibration task. 
%[EEG_LR, command] = pop_loadset('LR_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220125');
%[EEG_UD, command] = pop_loadset('UD_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220125');
%[EEG_LR, command] = pop_loadset('LR_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220120');
%[EEG_UD, command] = pop_loadset('UD_merged.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\HS_20220120');
[EEG_LR, command] = pop_loadset('DF_20211105_LR.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\DF_20211105_eeglabdata');
[EEG_UD, command] = pop_loadset('DF_20211105_UD.set', 'C:\Users\mcvai\OneDrive\Desktop\BCI\BCI Rotation Experiments\DF_20211105_eeglabdata');

%% Processing 
dataset = {EEG_LR, EEG_UD};
output = cell(1,2); 

for axis = 1:2
    EEG = dataset{axis};
    feedback_durations = [EEG.event(contains({EEG.event(:).type},'Feedback')).duration];
    target_durations = [EEG.event(contains({EEG.event(:).type},'TargetCode')).duration]; 

    types = {EEG.event(:).type}; 
    idxs = find(contains(types,'ResultCode'));
    trial_input_dir = [EEG.event(idxs).position];

    idxs = find(contains(types,'TargetCode'));
    idxs_new = []; 
    for ii = [idxs]
        if ~strcmp(types{ii + 1}, 'boundary')
            idxs_new = [idxs_new, ii];
        end
    end
    idxs = idxs_new;     
    trial_correct_dir = [EEG.event(idxs).position];
    trial_start_times = [EEG.event(idxs).latency];
    trial_end_times = trial_start_times + [EEG.event(idxs).duration];
    idxs = find(contains(types,'Feedback'));
    idxs_new = []; 
    for ii = [idxs]
        if ~strcmp(types{ii - 1}, 'boundary')
            idxs_new = [idxs_new, ii];
        end
    end
    idxs = idxs_new; 

    feedback_start_times = [EEG.event(idxs).latency];
    feedback_end_times = feedback_start_times + [EEG.event(idxs).duration];
    baseline_start_times = trial_start_times - 600;   % maybe adjust to 700 

    temp_trial = 0; 
    trials_with_results = []; 
    ii = 0;
    for i = types
        ii = ii + 1;
        if ii + 1 <= length(types)
            if ~strcmp(types{ii + 1}, 'boundary') % reject bad data 
                if strcmp(cell2mat(i), 'TargetCode')
                    temp_trial = temp_trial + 1; 
                end
                if strcmp(cell2mat(i), 'ResultCode') 
                    trials_with_results = [trials_with_results, temp_trial]; 
                end
            end
        else
            if strcmp(cell2mat(i), 'TargetCode')
                temp_trial = temp_trial + 1; 
            end
            if strcmp(cell2mat(i), 'ResultCode') 
                trials_with_results = [trials_with_results, temp_trial]; 
            end
        end
    end

    trial_classes = zeros(1, length(trial_correct_dir));
    control_data_table = zeros(length(trial_correct_dir), 5); 
    output{axis} = control_data_table; 
    
    for t = 1:length(trials_with_results)
        trial_classes(trials_with_results(t)) = trial_input_dir(t);
    end
    temp = [trial_correct_dir', trial_classes', baseline_start_times', trial_start_times', feedback_start_times', feedback_end_times', trial_end_times'];
    trial_data = array2table(temp,...
        'VariableNames', {'CorrectDirection', 'InputDirection', 'BaselineStart', 'TargetPresent', 'FeedbackStart', 'FeedbackEnd', 'TargetEnd'});

    correct_trials = find(trial_data.CorrectDirection == trial_data.InputDirection);

    if axis == 1
        left_hit_trials = find((trial_data.CorrectDirection == 2) & (trial_data.CorrectDirection == trial_data.InputDirection));
        right_hit_trials = find((trial_data.CorrectDirection == 1) & (trial_data.CorrectDirection == trial_data.InputDirection));
        
        % if including just timeout trials 
        left_trials = find((trial_data.CorrectDirection == 2) & (trial_data.CorrectDirection == trial_data.InputDirection | trial_data.InputDirection == 0));
        right_trials = find((trial_data.CorrectDirection == 1) & (trial_data.CorrectDirection == trial_data.InputDirection | trial_data.InputDirection == 0));
        
        % if including both timeout and aborted trials 
        %left_trials = find((trial_data.CorrectDirection == 2));
        %right_trials = find((trial_data.CorrectDirection == 1));
        
        trials_by_dir_LR = {right_trials; left_trials}; 
        
        left_timeout_trials = setdiff(left_trials, left_hit_trials); 
        right_timeout_trials = setdiff(right_trials, right_hit_trials); 
        
        % save data 
        EEG_LR = EEG; 
        trial_data_LR = trial_data; 
    elseif axis == 2
        up_hit_trials = find((trial_data.CorrectDirection == 1) & (trial_data.CorrectDirection == trial_data.InputDirection));
        down_hit_trials = find((trial_data.CorrectDirection == 2) & (trial_data.CorrectDirection == trial_data.InputDirection));
        
        % if including just timeout trials 
        up_trials = find((trial_data.CorrectDirection == 1) & (trial_data.CorrectDirection == trial_data.InputDirection | trial_data.InputDirection == 0));
        down_trials = find((trial_data.CorrectDirection == 2) & (trial_data.CorrectDirection == trial_data.InputDirection | trial_data.InputDirection == 0));
        
        % if including both timeout and aborted trials 
        %up_trials = find((trial_data.CorrectDirection == 1));
        %down_trials = find((trial_data.CorrectDirection == 2));
        
        trials_by_dir_UD = {up_trials; down_trials}; 
        
        up_timeout_trials = setdiff(up_trials, up_hit_trials); 
        down_timeout_trials = setdiff(down_trials, down_hit_trials); 
        
        % save data 
        EEG_UD = EEG; 
        trial_data_UD = trial_data; 
    end
end

for axis = 1:2 
    if axis == 1
        EEG = EEG_LR;
        trial_data = trial_data_LR; 
    elseif axis == 2
        EEG = EEG_UD; 
        trial_data = trial_data_UD;
    end
    
for dir = 1:2 
    LR_trials = trials_by_dir_LR{dir};
    LR_neutral_trials = [up_trials; down_trials];    % for vx=0. just a list, order doesn't matter 
    UD_trials = trials_by_dir_UD{dir}; 
    UD_neutral_trials = [left_trials; right_trials]; % for vy=0. 
    
    % Calculate mean neutral(v=0) power first 
    C3_neutral_powers = []; C4_neutral_powers = [];
    hor_neutral = []; ver_neutral = []; % alternative approach 
    for tr = 1:length(LR_neutral_trials) 
        trial = LR_neutral_trials(tr); 
        eeg_control = EEG_UD.data(:, trial_data_UD.FeedbackStart(trial):trial_data_UD.FeedbackEnd(trial));
        C3_data_control = double(eeg_control(C3,:) - 0.25 * eeg_control(C3_laplacians(1),:) - 0.25 * eeg_control(C3_laplacians(2),:) - 0.25 * eeg_control(C3_laplacians(3),:) - 0.25 * eeg_control(C3_laplacians(4),:));
        C4_data_control = double(eeg_control(C4,:) - 0.25 * eeg_control(C4_laplacians(1),:) - 0.25 * eeg_control(C4_laplacians(2),:) - 0.25 * eeg_control(C4_laplacians(3),:) - 0.25 * eeg_control(C4_laplacians(4),:)); 
        C3_data_control = filtfilt(notch_filter, C3_data_control);
        C4_data_control = filtfilt(notch_filter, C4_data_control); 
        C3_data_control = highpass(C3_data_control, 2, sampling_rate); 
        C4_data_control = highpass(C4_data_control, 2, sampling_rate);
        %C3_control_alpha = bandpower(C3_data_control, sampling_rate, alphaband_definition); 
        %C4_control_alpha = bandpower(C4_data_control, sampling_rate, alphaband_definition);
        C3_control_alpha = shin_bandpower(C3_data_control,sampling_rate,alphaband_definition); 
        C4_control_alpha = shin_bandpower(C4_data_control,sampling_rate,alphaband_definition); 
        C3_neutral_powers = [C3_neutral_powers, C3_control_alpha]; 
        C4_neutral_powers = [C4_neutral_powers, C4_control_alpha]; 
        hor_neutral = [hor_neutral, C4_control_alpha - C3_control_alpha]; % alternative approach 
        ver_neutral = [ver_neutral, -C4_control_alpha-C3_control_alpha]; % alternative approach 
    end
    C3_neutral_alpha_LR = mean(C3_neutral_powers); 
    C4_neutral_alpha_LR = mean(C4_neutral_powers); 
    hor_neutral_LR = mean(hor_neutral); % alternative approach 
    ver_neutral_LR = mean(ver_neutral); % alternative approach
    
    C3_neutral_powers = []; C4_neutral_powers = []; 
    hor_neutral = []; ver_neutral = []; % alternative approach 
    for tr = 1:length(UD_neutral_trials) 
        trial = UD_neutral_trials(tr); 
        eeg_control = EEG_LR.data(:, trial_data_LR.FeedbackStart(trial):trial_data_LR.FeedbackEnd(trial));
        C3_data_control = double(eeg_control(C3,:) - 0.25 * eeg_control(C3_laplacians(1),:) - 0.25 * eeg_control(C3_laplacians(2),:) - 0.25 * eeg_control(C3_laplacians(3),:) - 0.25 * eeg_control(C3_laplacians(4),:));
        C4_data_control = double(eeg_control(C4,:) - 0.25 * eeg_control(C4_laplacians(1),:) - 0.25 * eeg_control(C4_laplacians(2),:) - 0.25 * eeg_control(C4_laplacians(3),:) - 0.25 * eeg_control(C4_laplacians(4),:));
        C3_data_control = filtfilt(notch_filter, C3_data_control);
        C4_data_control = filtfilt(notch_filter, C4_data_control);
        C3_data_control = highpass(C3_data_control, 2, sampling_rate); 
        C4_data_control = highpass(C4_data_control, 2, sampling_rate);
        %C3_control_alpha = bandpower(C3_data_control, sampling_rate, alphaband_definition); 
        %C4_control_alpha = bandpower(C4_data_control, sampling_rate, alphaband_definition);
        C3_control_alpha = shin_bandpower(C3_data_control,sampling_rate,alphaband_definition); 
        C4_control_alpha = shin_bandpower(C4_data_control,sampling_rate,alphaband_definition);
        C3_neutral_powers = [C3_neutral_powers, C3_control_alpha]; 
        C4_neutral_powers = [C4_neutral_powers, C4_control_alpha]; 
        hor_neutral = [hor_neutral, C4_control_alpha - C3_control_alpha]; % alternative approach 
        ver_neutral = [ver_neutral, -C4_control_alpha-C3_control_alpha]; % alternative approach 
    end
    C3_neutral_alpha_UD = mean(C3_neutral_powers); 
    C4_neutral_alpha_UD = mean(C4_neutral_powers);
    hor_neutral_UD = mean(hor_neutral); % alternative approach 
    ver_neutral_UD = mean(ver_neutral); % alternative approach 
    
    control_data_table = output{1}; 
    for tr = 1:length(LR_trials)
        trial = LR_trials(tr); 
        eeg_control = EEG_LR.data(:, trial_data_LR.FeedbackStart(trial):trial_data_LR.FeedbackEnd(trial));
        C3_data_control = double(eeg_control(C3,:) - 0.25 * eeg_control(C3_laplacians(1),:) - 0.25 * eeg_control(C3_laplacians(2),:) - 0.25 * eeg_control(C3_laplacians(3),:) - 0.25 * eeg_control(C3_laplacians(4),:));
        C4_data_control = double(eeg_control(C4,:) - 0.25 * eeg_control(C4_laplacians(1),:) - 0.25 * eeg_control(C4_laplacians(2),:) - 0.25 * eeg_control(C4_laplacians(3),:) - 0.25 * eeg_control(C4_laplacians(4),:)); 
        C3_data_control = filtfilt(notch_filter, C3_data_control);
        C4_data_control = filtfilt(notch_filter, C4_data_control);
        C3_data_control = highpass(C3_data_control, 2, sampling_rate); 
        C4_data_control = highpass(C4_data_control, 2, sampling_rate);
        %C3_control_alpha = bandpower(C3_data_control, sampling_rate, alphaband_definition); 
        %C4_control_alpha = bandpower(C4_data_control, sampling_rate, alphaband_definition); 
        C3_control_alpha = shin_bandpower(C3_data_control,sampling_rate,alphaband_definition); 
        C4_control_alpha = shin_bandpower(C4_data_control,sampling_rate,alphaband_definition);
        
        %relative_C3 = C3_control_alpha - C3_neutral_alpha_LR; 
        %relative_C4 = C4_control_alpha - C4_neutral_alpha_LR; 
        relative_C3 = C3_control_alpha; % alternative approach 
        relative_C4 = C4_control_alpha; % alternative approach 
        
        %control_data_table(trial, 1) = relative_C4 - relative_C3; 
        %control_data_table(trial, 2) = -(relative_C4 + relative_C3);
        control_data_table(trial, 1) = relative_C4 - relative_C3- hor_neutral_LR; % alternative approach 
        control_data_table(trial, 2) = -(relative_C4 + relative_C3)- ver_neutral_LR; % alternative approach 
        
        % CHANGE HERE IF DIRECTION CODE IS WRONG 
        if dir == 1 % RIGHT
            control_data_table(trial, 3) = 1; 
            control_data_table(trial, 4) = nan; 
        elseif dir == 2 % LEFT 
            control_data_table(trial, 3) = -1;
            control_data_table(trial, 4) = nan; 
        end
        
        if ismember(trial, [left_timeout_trials; right_timeout_trials])
            control_data_table(trial, 5) = timeout_trial_weight; 
        else
            control_data_table(trial, 5) = 1.0; 
        end
        
    end
    output{1} = control_data_table; 
    
    control_data_table = output{2}; 
    for tr = 1:length(UD_trials)
        trial = UD_trials(tr); 
        eeg_control = EEG_UD.data(:, trial_data_UD.FeedbackStart(trial):trial_data_UD.FeedbackEnd(trial));
        C3_data_control = double(eeg_control(C3,:) - 0.25 * eeg_control(C3_laplacians(1),:) - 0.25 * eeg_control(C3_laplacians(2),:) - 0.25 * eeg_control(C3_laplacians(3),:) - 0.25 * eeg_control(C3_laplacians(4),:));
        C4_data_control = double(eeg_control(C4,:) - 0.25 * eeg_control(C4_laplacians(1),:) - 0.25 * eeg_control(C4_laplacians(2),:) - 0.25 * eeg_control(C4_laplacians(3),:) - 0.25 * eeg_control(C4_laplacians(4),:)); 
        C3_data_control = filtfilt(notch_filter, C3_data_control);
        C4_data_control = filtfilt(notch_filter, C4_data_control);
        C3_data_control = highpass(C3_data_control, 2, sampling_rate); 
        C4_data_control = highpass(C4_data_control, 2, sampling_rate);
        %C3_control_alpha = bandpower(C3_data_control, sampling_rate, alphaband_definition); 
        %C4_control_alpha = bandpower(C4_data_control, sampling_rate, alphaband_definition); 
        C3_control_alpha = shin_bandpower(C3_data_control,sampling_rate,alphaband_definition); 
        C4_control_alpha = shin_bandpower(C4_data_control,sampling_rate,alphaband_definition);
        
        %relative_C3 = C3_control_alpha - C3_neutral_alpha_UD; 
        %relative_C4 = C4_control_alpha - C4_neutral_alpha_UD; 
        relative_C3 = C3_control_alpha; % alternative approach 
        relative_C4 = C4_control_alpha; % alternative approach 
        
        %control_data_table(trial, 1) = relative_C4 - relative_C3; 
        %control_data_table(trial, 2) = -(relative_C4 + relative_C3);
        control_data_table(trial, 1) = relative_C4 - relative_C3- hor_neutral_UD; % alternative approach 
        control_data_table(trial, 2) = -(relative_C4 + relative_C3)- ver_neutral_UD; % alternative approach 
        
        % CHANGE HERE IF DIRECTION CODE IS WRONG 
        if dir == 1 % UP
            control_data_table(trial, 3) = nan; 
            control_data_table(trial, 4) = 1; 
        elseif dir == 2 % DOWN
            control_data_table(trial, 3) = nan;
            control_data_table(trial, 4) = -1; 
        end
        
        if ismember(trial, [up_timeout_trials; down_timeout_trials])
            control_data_table(trial, 5) = timeout_trial_weight; 
        else
            control_data_table(trial, 5) = 1.0; 
        end
        
    end
    output{2} = control_data_table; 
end
end 
    
% Analyze 
for axis = 1:2 
    control_data_table = output{axis}; 
    if axis == 1
        left = control_data_table(control_data_table(:,3) == -1, :); 
        left = left(:,[1,2,3,5]);
        avg_baseline_hor_control_in_left_trials = sum(left(:,1).*left(:,4)) / length(left); 
        avg_baseline_ver_control_in_left_trials = sum(left(:,2).*left(:,4)) / length(left); 
        cl = [avg_baseline_hor_control_in_left_trials, avg_baseline_ver_control_in_left_trials];
        
        right = control_data_table(control_data_table(:,3) == 1, :); 
        right = right(:,[1,2,3,5]);
        avg_baseline_hor_control_in_right_trials = sum(right(:,1).*right(:,4)) / length(right); 
        avg_baseline_ver_control_in_right_trials = sum(right(:,2).*right(:,4)) / length(right); 
        cr = [avg_baseline_hor_control_in_right_trials, avg_baseline_ver_control_in_right_trials];
    elseif axis == 2
        down = control_data_table(control_data_table(:,4) == -1, :);
        down = down(:,[1,2,4,5]);
        avg_baseline_hor_control_in_down_trials = sum(down(:,1).*down(:,4)) / length(down); 
        avg_baseline_ver_control_in_down_trials = sum(down(:,2).*down(:,4)) / length(down); 
        cd = [avg_baseline_hor_control_in_down_trials, avg_baseline_ver_control_in_down_trials];
        
        up = control_data_table(control_data_table(:,4) == 1, :); 
        up = up(:,[1,2,4,5]); 
        avg_baseline_hor_control_in_up_trials = sum(up(:,1).*up(:,4)) / length(up); 
        avg_baseline_ver_control_in_up_trials = sum(up(:,2).*up(:,4)) / length(up); 
        cu = [avg_baseline_hor_control_in_up_trials, avg_baseline_ver_control_in_up_trials];
    end
end

dl = [-1 0];
dr = [1 0]; 
du = [0 1]; 
dd = [0 -1]; 
    
% Rotation #1 (original concept)
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

cl_new = (R_LR * cl')';
cr_new = (R_LR * cr')'; 
cd_new = (R_DU * cd')'; 
cu_new = (R_DU * cu')'; 

% Rotation #2 (The Balance) - translation into weights not implemented yet
la2 = atan2d(det([cl_new', dl']), dot(cl_new', dl'));
ra2 = atan2d(det([cr_new', dr']), dot(cr_new', dr')); 
ua2 = atan2d(det([cu_new', du']), dot(cu_new', du'));
da2 = atan2d(det([cd_new', dd']), dot(cd_new', dd')); 
theta_LD = mean([la2, da2]); 
theta_RU = mean([ra2, ua2]); 
R_LD = [cosd(theta_LD), -sind(theta_LD); sind(theta_LD), cosd(theta_LD)]; 
R_RU = [cosd(theta_RU), -sind(theta_RU); sind(theta_RU), cosd(theta_RU)]; 

cl_final = R_LD * cl_new';
cr_final = R_RU * cr_new'; 
cd_final = R_LD * cd_new'; 
cu_final = R_RU * cu_new'; 

% Add the number weights calculation and return those as well! 
x_weights = [cosd(theta_LR) + sind(theta_LR), sind(theta_LR) - cosd(theta_LR)]; % [C4, C3]
y_weights = [sind(theta_DU) - cosd(theta_DU), -cosd(theta_DU) - sind(theta_DU)]; % [C4, C3]

figure;
subplot(1, 3, 1); title('Before correction'); 
xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
p1 = plot([0, cr(1)], [0, cr(2)]); 
p2 = plot([0, cu(1)], [0, cu(2)]); 
p3 = plot([0, cl(1)], [0, cl(2)]); 
p4 = plot([0, cd(1)], [0, cd(2)]); 
px = xline(0, '--'); py = yline(0, '--');
legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 

subplot(1, 3, 2); title('Intermediate correction'); 
xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
p1 = plot([0, cr_new(1)], [0, cr_new(2)]); 
p2 = plot([0, cu_new(1)], [0, cu_new(2)]); 
p3 = plot([0, cl_new(1)], [0, cl_new(2)]); 
p4 = plot([0, cd_new(1)], [0, cd_new(2)]); 
px = xline(0, '--'); py = yline(0, '--');
legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 

subplot(1, 3, 3); title('After correction'); 
xlim([-1 1]); ylim([-1 1]); set(gcf, 'Position',  [100, 100, 1200, 400]); hold on;
p1 = plot([0, cr_final(1)], [0, cr_final(2)]); 
p2 = plot([0, cu_final(1)], [0, cu_final(2)]); 
p3 = plot([0, cl_final(1)], [0, cl_final(2)]); 
p4 = plot([0, cd_final(1)], [0, cd_final(2)]); 
px = xline(0, '--'); py = yline(0, '--');
legend([p1,p2,p3,p4], {'right', 'up', 'left', 'down'}); 


function pwr = shin_bandpower(eeg_time_series, fs, alphaband_definition)
    realx = eeg_time_series; 
    y = fft(realx); 
    n = length(realx);          % number of samples
    f = (0:n-1)*(fs/n);         % frequency range
    power = abs(y).^2/n;        % PSD 
    
    f_resolution = diff(f); 
    f_resolution = f_resolution(1); 
    
    pwr = bandpowers(power, f_resolution, alphaband_definition, 'valueType', 'absolute', 'integrationMethod', 'trapez'); 
    
end