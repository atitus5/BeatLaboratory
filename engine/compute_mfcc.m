function [features] = compute_mfcc(wav_path)
% Compute Aurora AFE features for all frames in the given WAV file.
% These are the log frame energy and first 13 MFCC coefficients.

% Read in the WAV file
[y, Fs] = audioread(wav_path);

frame_len = 400;        % 25 ms
frame_shift = 160;      % 10 ms
frame_count = floor(length(y) / frame_shift) - 1;

% Filter out the DC component
dc = mean(y);
dc_filtered = y - dc;

% Pad end with zeros
pad_count = frame_shift * ((frame_count - 1) - ((length(y) - frame_len)/frame_shift));
padding = zeros(int64(pad_count), 1);
dc_filtered_padded = cat(1, dc_filtered, padding);

features = zeros(frame_count, 14);

% Create pre-emphasis filter
s_pe = 1.0;       % Output; s_pe[n]
s_of = [1.0 -0.97];     % Input;  s_of[n] - 0.97s_of[n - 1] 
filtered = filter(s_of, s_pe, dc_filtered_padded);

% Load Mel frequency filter bank
%load('/mit/sgi-desktop1/6345_data/wavs/lab1/mel_filters.mat');
load('/Users/atitus/6.345/lab/mel_filters.mat', 'mel_filters');

for i=1:frame_count
    frame_start = 1 + (frame_shift * (i - 1));
    
    % Log frame energy of DC-filtered (NOT pre-emphasis) signal
    fe = 0.0;
    for j = 0:frame_len - 1
        fe = fe + (dc_filtered_padded(frame_start + j))^2;
    end
    lfe = max(-50.0, log(fe));
    
    % Add to feature vector
    features(i,1) = lfe;

    % Window the frame
    frame_end = frame_start + frame_len - 1;
    filtered_frame = filtered(frame_start:frame_end);
    windowed_frame = filtered_frame .* hamming(frame_len);
    fft_padding = zeros(512 - frame_len, 1);
    windowed_frame_padded = cat(1, windowed_frame, fft_padding);

    % Compute FFT and then MSFCs
    frame_spectrum = fft(windowed_frame_padded, 512);
    mfsc = zeros(23, 1);
    for j=1:23
        current_band = mel_filters(:,j);
        filtered_spectrum = frame_spectrum(1:257) .* current_band;
        
        energy = 0.0;
        for k=1:length(filtered_spectrum)
            energy = energy + abs(filtered_spectrum(k));
        end
        
        mfsc(j) = max(-50.0, log(energy));
    end
    
    % Compute MFCCs
    mfcc = zeros(13, 1);
    for j=0:12
        for k=1:23
            mfcc(j + 1) = mfcc(j + 1) + (mfsc(k) * cos(pi * j / 23.0 * (k - 0.5)));
        end
    end
    
    % Add to feature vector
    for j=2:14
        features(i,j) = mfcc(j - 1);
    end
end