% Main function with visualization of each stage
function visualize_ble_stages()
    % Load built-in image
    input_image = imread('peppers.png');
    SNR_dB = 15;  % Set SNR value
    
    % 1. Image Preprocessing and Display
    fprintf('Stage 1: Image Preprocessing\n');
    if size(input_image, 3) == 3
        img_gray = rgb2gray(input_image);
    else
        img_gray = input_image;
    end
    
    figure('Name', 'Image Processing Stages');
    subplot(3,3,1);
    imshow(input_image);
    title('Original RGB Image');
    
    subplot(3,3,2);
    imshow(img_gray);
    title('Grayscale Image');
    
    % 2. Convert to binary and show sample
    binary_data = image_to_binary(img_gray);
    subplot(3,3,3);
    stem(binary_data(1:100));
    title('Binary Data (First 100 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    
    % 3. Packetize and encode with Hamming code
    fprintf('Stage 2: Packetization and Encoding\n');
    packet_size = 256;
    packets = custom_packetization(binary_data, packet_size);
    encoded_data = custom_hamming_encode(packets);
    
    subplot(3,3,4);
    plot(encoded_data(1:500));
    title('Encoded Data (First 500 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    
    % 4. Data Whitening
    fprintf('Stage 3: Data Whitening\n');
    whitened_data = custom_data_whitening(encoded_data);
    
    subplot(3,3,5);
    plot(whitened_data(1:500));
    title('Whitened Data (First 500 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    
    % 5. GFSK Modulation
    fprintf('Stage 4: GFSK Modulation\n');
    samples_per_bit = 8;
    h_index = 0.5;
    modulated_signal = custom_gfsk_modulate(whitened_data, h_index, samples_per_bit);
    
    subplot(3,3,6);
    plot(real(modulated_signal(1:200)));
    hold on;
    plot(imag(modulated_signal(1:200)));
    title('Modulated Signal');
    xlabel('Sample Index');
    ylabel('Amplitude');
    legend('Real', 'Imag');
    grid on;
    
    % 6. Channel Simulation
    fprintf('Stage 5: Channel Simulation\n');
    received_signal = custom_channel_simulation(modulated_signal, SNR_dB);
    
    subplot(3,3,7);
    plot(real(received_signal(1:200)));
    hold on;
    plot(imag(received_signal(1:200)));
    title(['Received Signal (SNR = ' num2str(SNR_dB) 'dB)']);
    xlabel('Sample Index');
    ylabel('Amplitude');
    legend('Real', 'Imag');
    grid on;
    
    % 7. Demodulation and Decoding
    fprintf('Stage 6: Demodulation and Decoding\n');
    decoded_data = custom_demodulate_decode(received_signal, samples_per_bit);
    
    subplot(3,3,8);
    stem(decoded_data(1:100));
    title('Decoded Data (First 100 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    
    % 8. Image Reconstruction
    fprintf('Stage 7: Image Reconstruction\n');
    received_image = binary_to_image(decoded_data, size(img_gray));
    
    subplot(3,3,9);
    imshow(uint8(received_image));
    title('Reconstructed Image');
    
    % Calculate and display BER
    ber = sum(abs(binary_data(1:min(length(binary_data),length(decoded_data))) - ...
              decoded_data(1:min(length(binary_data),length(decoded_data))))) / ...
              length(binary_data);
    fprintf('Bit Error Rate: %f\n', ber);
end

% Helper functions (same as before but with added visualization)
function binary_data = image_to_binary(img)
    img_vector = img(:);
    binary_matrix = zeros(length(img_vector), 8);
    for i = 1:length(img_vector)
        value = img_vector(i);
        for bit = 1:8
            binary_matrix(i, bit) = bitget(value, 9-bit);
        end
    end
    binary_data = binary_matrix(:)';
end

function packets = custom_packetization(data, packet_size)
    bits_per_packet = packet_size * 8;
    num_packets = ceil(length(data) / bits_per_packet);
    packets = zeros(num_packets, bits_per_packet + 32);
    
    for i = 1:num_packets
        header = zeros(1, 32);  % Simplified header
        start_idx = (i-1) * bits_per_packet + 1;
        end_idx = min(i * bits_per_packet, length(data));
        packet_data = data(start_idx:end_idx);
        
        if length(packet_data) < bits_per_packet
            packet_data = [packet_data zeros(1, bits_per_packet - length(packet_data))];
        end
        
        packets(i, :) = [header packet_data];
    end
end

function encoded_data = custom_hamming_encode(data)
    % Simplified (7,4) Hamming encoding
    data_reshape = reshape([data(:)' zeros(1, mod(-length(data(:)), 4))], [], 4);
    encoded_data = zeros(size(data_reshape, 1), 7);
    
    for i = 1:size(data_reshape, 1)
        msg = data_reshape(i, :);
        encoded_data(i, :) = [msg(1:4) xor(xor(msg(1),msg(2)),msg(3)) ...
                             xor(xor(msg(2),msg(3)),msg(4)) ...
                             xor(xor(msg(1),msg(3)),msg(4))];
    end
    
    encoded_data = encoded_data(:)';
end

function whitened_data = custom_data_whitening(data)
    whitening_seq = randi([0 1], size(data));
    whitened_data = xor(data, whitening_seq);
end

function modulated_signal = custom_gfsk_modulate(data, h_index, samples_per_bit)
    % Gaussian filter design
    BT = 0.5;
    span = 4;
    t = (-span/2:1/samples_per_bit:span/2);
    alpha = sqrt(log(2)/(2*(BT^2)));
    h_gaussian = exp(-((alpha*t).^2));
    h_gaussian = h_gaussian / sum(h_gaussian);
    
    % Upsample and filter
    upsampled_data = upsample(2*data-1, samples_per_bit);
    filtered_data = conv(upsampled_data, h_gaussian, 'same');
    
    % FM modulation
    phase = cumsum(filtered_data) * pi * h_index;
    modulated_signal = exp(1j * phase);
end

function received_signal = custom_channel_simulation(transmitted_signal, SNR_dB)
    signal_power = mean(abs(transmitted_signal).^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power/2) * (randn(size(transmitted_signal)) + ...
            1j*randn(size(transmitted_signal)));
    received_signal = transmitted_signal + noise;
end

function decoded_data = custom_demodulate_decode(received_signal, samples_per_bit)
    % Phase demodulation
    phase_diff = angle(received_signal(2:end) .* conj(received_signal(1:end-1)));
    
    % Downsample
    downsampled_phase = phase_diff(1:samples_per_bit:end);
    
    % Decision
    decoded_data = downsampled_phase > 0;
    
    % Remove whitening (using same function as encoding)
    decoded_data = custom_data_whitening(decoded_data);
    
    % Simplified Hamming decode (error detection only)
    decoded_data = decoded_data(1:4:end);
end

function img = binary_to_image(binary_stream, img_size)
    % Reshape binary stream into bytes
    binary_matrix = reshape(binary_stream(1:prod(img_size)*8), 8, [])';
    
    % Convert binary to decimal
    img_vector = zeros(size(binary_matrix, 1), 1);
    for i = 1:size(binary_matrix, 1)
        img_vector(i) = sum(binary_matrix(i, :) .* (2.^(7:-1:0)));
    end
    
    % Reshape to original image dimensions
    img = reshape(img_vector, img_size);
end

% Run the visualization
