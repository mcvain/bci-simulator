%% fft/power/etc

clear all
close all
clc

%%

timelen = 10; %seconds
fs = 1000;
t = 0:1/fs:timelen-1/fs;                      % 10 second span time vector

% make a signal that's
x = (1.3)*sin(2*pi*15*t) ...             % 15 Hz component
  + (1.7)*sin(2*pi*40*(t-2)) ...         % 40 Hz component
  + 2.5*randn(size(t));  

y = fft(x);
n = length(x);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
power = abs(y).^2/n;    % power of the DFT

figure;
plot(f,power)
xlabel('Frequency')
ylabel('Power')
xlim([0 60]);

%%

% change to load your local .mat file
load sample_EEG_data.mat

realx = EEG_timetrace(1:timelen*fs);
y = fft(realx);

n = length(realx);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
power = abs(y).^2/n;    % power of the DFT

plot(f,power)
xlabel('Frequency')
ylabel('Power')
xlim([0 10]);