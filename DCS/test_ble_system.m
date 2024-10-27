
% Example helper function to test the complete system
function test_ble_system(image_path)
    % Load image
    original_image = imread(image_path);
    if size(original_image, 3) == 3
        original_image = rgb2gray(original_image);
    end
    
    % Save original size
    original_size = size(original_image);
    
    % Set parameters
    params.samples_per_bit = 8;
    params.modulation_index = 0.5;
    params.packet_size = 256;
    params.SNR_dB = 15;
    
    % Transmit
    [I_signal, Q_signal] = transmit_ble_image(image_path, params.SNR_dB);
    
    % Add noise
    noise_power = 10^(-params.SNR_dB/10);
    I_noisy = I_signal + sqrt(noise_power/2) * randn(size(I_signal));
    Q_noisy = Q_signal + sqrt(noise_power/2) * randn(size(Q_signal));
    
    % Receive
    received_image = receive_ble_signal(I_noisy, Q_noisy, original_size, params);
    
    % Display results
    figure('Name', 'System Test Results');
    subplot(1,2,1); imshow(original_image); title('Original Image');
    subplot(1,2,2); imshow(received_image); title('Received Image');
end
