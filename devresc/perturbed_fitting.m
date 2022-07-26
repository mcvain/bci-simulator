% Perturbed encoding calculation - single subject, single session 
% Shin 9/10/2021

clear; close all;
myDir = 'C:\Users\mcvai\OneDrive\Desktop\BCI\jake-dataset\S5_Session_1.mat';

data_table = nan(1,1,4,2); % Subject,Session,Direction,Hemisphere 
control_data_table = nan(1, 1, 450, 6); % Subject, Session, Trial, (alphaC3, alphaC4, x_velocity, y_velocity)
baseline_data_table = nan(1, 1, 450, 6); % Subject, Session, Trial, (alphaC3, alphaC4, x_velocity, y_velocity)

load(myDir)
C3_laplacians = []; C4_laplacians = []; 
for ch_idx = 1:length(BCI.chaninfo.label)
  if strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C3')
    C3_idx = ch_idx;
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C4')
    C4_idx = ch_idx;
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'FC3')
    C3_laplacians = [C3_laplacians, ch_idx]; 
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C5')
    C3_laplacians = [C3_laplacians, ch_idx]; 
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C1')
    C3_laplacians = [C3_laplacians, ch_idx];
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'CP3')
    C3_laplacians = [C3_laplacians, ch_idx];
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'FC4')
    C4_laplacians = [C4_laplacians, ch_idx]; 
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C2')
    C4_laplacians = [C4_laplacians, ch_idx]; 
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'C6')
    C4_laplacians = [C4_laplacians, ch_idx];
  elseif strcmp(cell2mat(BCI.chaninfo.label(ch_idx)), 'CP4')
    C4_laplacians = [C4_laplacians, ch_idx];
  end
end


trialdata = squeeze(struct2cell(BCI.TrialData));
tasknumbers = trialdata(1,:); tasknumbers = cell2mat(tasknumbers);
% classes = trialdata(8,:); 
% emptyIndex = cellfun(@isnan, classes); % Find indices of empty cells
% classes(emptyIndex) = {false};    % Fill empty cells with 0
% classes = logical(cell2mat(classes));  % Convert the cell array
targets = trialdata(4,:); targets = cell2mat(targets); 

targetnumbers = trialdata(4,:); targetnumbers = cell2mat(targetnumbers);
artifact = trialdata(10,:); artifact = cell2mat(artifact);
result = trialdata(8,:);
emptyIndex = cellfun(@isnan, result); % Find indices of empty cells
result(emptyIndex) = {false};    % Fill empty cells with 0
result = logical(cell2mat(result));  % Convert the cell array
% forcedresult = trialdata(9,:); forcedresult = logical(cell2mat(forcedresult));
    
resultinds = trialdata(7,:); resultinds = cell2mat(resultinds); 
    

rightward_trials = find(targetnumbers == 1 & artifact == false & result == true & tasknumbers ~= 3 & resultinds > 4700);  
leftward_trials = find(targetnumbers == 2 & artifact == false & result == true & tasknumbers ~= 3 & resultinds > 4700); 
upward_trials = find(targetnumbers == 3 & artifact == false & result == true & tasknumbers ~= 3 & resultinds > 4700);
downward_trials = find(targetnumbers == 4 & artifact == false & result == true & tasknumbers ~= 3 & resultinds > 4700);

clear targetnumbers artifact result  
trials_organized_by_directions = {rightward_trials; leftward_trials; upward_trials; downward_trials}; 
for dir = 1:4 
    trials = trials_organized_by_directions{dir};
    % C3_alpha_summation_over_trials = 0; C4_alpha_summation_over_trials = 0; 
    for tr = 1:length(trials)
        trial = trials(tr);
        eeg = BCI.data{trial};
        
        
        
        if (dir == 1) || (dir == 2)
            baseline_trials = [upward_trials, downward_trials]; 
        elseif (dir == 3) || (dir == 4)
            baseline_trials = [rightward_trials, leftward_trials];
        end
        
        C3_baseline_alpha = [];
        C4_baseline_alpha = [];
        for btr = [baseline_trials]
            eeg_baseline = BCI.data{btr}; 
            eeg_baseline = eeg_baseline(:, 4001:BCI.TrialData(btr).resultind);
            C3_data_baseline_tr = eeg_baseline(C3_idx,:) - 0.25 .* (eeg_baseline(C3_laplacians(1),:) + eeg_baseline(C3_laplacians(2),:) + eeg_baseline(C3_laplacians(3),:) + eeg_baseline(C3_laplacians(4),:));
            C4_data_baseline_tr = eeg_baseline(C4_idx,:) - 0.25 .* (eeg_baseline(C4_laplacians(1),:) + eeg_baseline(C4_laplacians(2),:) + eeg_baseline(C4_laplacians(3),:) + eeg_baseline(C4_laplacians(4),:));
            C3_data_baseline_tr = preproc_highpassfilter(double(C3_data_baseline_tr), 1000, 1); 
            C4_data_baseline_tr = preproc_highpassfilter(double(C4_data_baseline_tr), 1000, 1);
            [C3_baseline_averaged_PSD, F] = pwelch(C3_data_baseline_tr, 256, 256/2, 4096, 1000, 'onesided', 'power');
            [C4_baseline_averaged_PSD, F] = pwelch(C4_data_baseline_tr, 256, 256/2, 4096, 1000, 'onesided', 'power');
            [~, alpha_start_index] = (min(abs(F - 8))); 
            alpha_start_index = alpha_start_index - 1;
            [~, alpha_end_index] = (min(abs(F - 14))); 
            alpha_end_index = alpha_end_index + 1;
            C3_baseline_alpha = [C3_baseline_alpha, mean(C3_baseline_averaged_PSD(alpha_start_index:alpha_end_index))];
            C4_baseline_alpha = [C4_baseline_alpha, mean(C4_baseline_averaged_PSD(alpha_start_index:alpha_end_index))]; 
        end
        
        if (dir == 1) || (dir == 2)
            C3_UD_alpha = mean(C3_baseline_alpha); 
            C4_UD_alpha = mean(C4_baseline_alpha); 
        elseif (dir == 3) || (dir == 4)
            C3_LR_alpha = mean(C3_baseline_alpha); 
            C4_LR_alpha = mean(C4_baseline_alpha); 
        end

        eeg_control = eeg(:,4001:BCI.TrialData(trial).resultind);
        C3_data_control = eeg_control(C3_idx,:) - 0.25 .* (eeg_control(C3_laplacians(1),:) + eeg_control(C3_laplacians(2),:) + eeg_control(C3_laplacians(3),:) + eeg_control(C3_laplacians(4),:));
        C4_data_control = eeg_control(C4_idx,:) - 0.25 .* (eeg_control(C4_laplacians(1),:) + eeg_control(C4_laplacians(2),:) + eeg_control(C4_laplacians(3),:) + eeg_control(C4_laplacians(4),:));

        C3_data_control = preproc_highpassfilter(double(C3_data_control), 1000, 1); 
        C4_data_control = preproc_highpassfilter(double(C4_data_control), 1000, 1); 

        [C3_control_averaged_PSD, F] = pwelch(C3_data_control, 256, 256/2, 4096, 1000, 'onesided', 'power');
        [C4_control_averaged_PSD, F] = pwelch(C4_data_control, 256, 256/2, 4096, 1000, 'onesided', 'power'); 

        [~, alpha_start_index] = (min(abs(F - 8))); 
        alpha_start_index = alpha_start_index - 1;
        [~, alpha_end_index] = (min(abs(F - 14))); 
        alpha_end_index = alpha_end_index + 1; % verified to be working as intended as long as sampling freq (fs) is defined in the power esimation function e.g. pwelch, pburg. 

        if dir == 2
            C3_left_alpha = mean(C3_control_averaged_PSD(alpha_start_index:alpha_end_index));
            C4_left_alpha = mean(C4_control_averaged_PSD(alpha_start_index:alpha_end_index));
        elseif dir == 1
            C3_right_alpha = mean(C3_control_averaged_PSD(alpha_start_index:alpha_end_index));
            C4_right_alpha = mean(C4_control_averaged_PSD(alpha_start_index:alpha_end_index));
        elseif dir == 3
            C3_up_alpha = mean(C3_control_averaged_PSD(alpha_start_index:alpha_end_index));
            C4_up_alpha = mean(C4_control_averaged_PSD(alpha_start_index:alpha_end_index));
        elseif dir == 4
            C3_down_alpha = mean(C3_control_averaged_PSD(alpha_start_index:alpha_end_index));
            C4_down_alpha = mean(C4_control_averaged_PSD(alpha_start_index:alpha_end_index));
        end
    end
end


%% Fitting 
x = [-1 0 1]; 
% 1 
y1 = [C3_left_alpha, C3_UD_alpha, C3_right_alpha]; 
% 3 
y3 = [C4_left_alpha, C4_UD_alpha, C4_right_alpha]; 
LR_max = max([y1 y3]); 
y1 = y1 ./ LR_max;
y3 = y3 ./ LR_max;


% 2
y2 = [C3_up_alpha, C3_LR_alpha, C3_down_alpha]; 
% 4 
y4 = [C4_up_alpha, C4_LR_alpha, C4_down_alpha];
UD_max = max([y2 y4]); 
y2 = y2 ./ UD_max;
y4 = y4 ./ UD_max;

[param1,stat1] = sigm_fit(x,y1,[0, 1 , NaN , NaN],[],1);
[param2,stat2] = sigm_fit(x,y2,[0, 1 , NaN , NaN],[],1);
[param3,stat3] = sigm_fit(x,y3,[0, 1 , NaN , NaN],[],1);
[param4,stat4] = sigm_fit(x,y4,[0, 1 , NaN , NaN],[],1);
