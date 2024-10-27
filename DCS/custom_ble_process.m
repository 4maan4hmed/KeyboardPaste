% Main function for BLE image processing

function [received_image, ber] = custom_ble_process(input_image, SNR_dB)
    % System Parameters
    packet_size = 256;          % bytes per packet
    h_index = 0.5;             % modulation index for GFSK
    bit_rate = 1e6;            % 1 Mbps (BLE standard)
    samples_per_bit = 8;        % oversampling factor
    fs = bit_rate * samples_per_bit;
    preamble = [0 1 0 1 0 1 0 1]; % BLE preamble pattern
    % Image Preprocessing
    if ischar(input_image)
        img = imread(input_image);
    else
        img = input_image;
    end
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % Convert image to binary stream
    binary_data = custom_image_to_binary(img_gray);
    
    % 1. Custom Packetization with Header and CRC
    packets = custom_packetization(binary_data, packet_size);
    
    % 2. Custom Hamming Encoding
    encoded_data = custom_hamming_encode(packets);
    
    % 3. Custom Whitening (Pattern Mapping)
    whitened_data = custom_data_whitening(encoded_data);
    
    % 4. Custom GFSK Modulation
    modulated_signal = custom_gfsk_modulate(whitened_data, h_index, samples_per_bit);
    
    % 5. Channel Simulation
    received_signal = custom_channel_simulation(modulated_signal, SNR_dB);
    
    % 6. Demodulation and Decoding
    decoded_data = custom_demodulate_decode(received_signal, samples_per_bit);
    
    % 7. Reconstruct Image
    received_image = custom_binary_to_image(decoded_data, size(img_gray));
    
    % Calculate BER
    ber = sum(abs(binary_data(1:min(length(binary_data),length(decoded_data))) - ...
              decoded_data(1:min(length(binary_data),length(decoded_data))))) / ...
              length(binary_data);
end

% Custom function to convert image to binary stream
function binary_stream = custom_image_to_binary(img)
    img_vector = img(:);
    binary_matrix = zeros(length(img_vector), 8);
    for i = 1:length(img_vector)
        value = img_vector(i);
        for bit = 1:8
            binary_matrix(i, bit) = bitget(value, 9-bit);
        end
    end
    binary_stream = binary_matrix(:)';
end

% Custom packetization with header and CRC
function packets = custom_packetization(data, packet_size)
    % Add packet headers and CRC
    bits_per_packet = packet_size * 8;
    num_packets = ceil(length(data) / bits_per_packet);
    
    % Initialize packets array
    packets = zeros(num_packets, bits_per_packet + 32); % +32 for header
    
    for i = 1:num_packets
        % Create packet header (4 bytes = 32 bits)
        header = generate_packet_header(i, num_packets);
        
        % Get data for this packet
        start_idx = (i-1) * bits_per_packet + 1;
        end_idx = min(i * bits_per_packet, length(data));
        packet_data = data(start_idx:end_idx);
        
        % Pad if necessary
        if length(packet_data) < bits_per_packet
            packet_data = [packet_data zeros(1, bits_per_packet - length(packet_data))];
        end
        
        % Add CRC
        crc = custom_crc_calculate(packet_data);
        
        % Combine header, data, and CRC
        packets(i, :) = [header packet_data crc];
    end
end

% Generate packet header
function header = generate_packet_header(packet_num, total_packets)
    header = zeros(1, 32);
    % Convert packet number to binary (12 bits)
    packet_num_bin = de2bi(packet_num, 12, 'left-msb');
    % Convert total packets to binary (12 bits)
    total_packets_bin = de2bi(total_packets, 12, 'left-msb');
    % Flags (8 bits)
    flags = zeros(1, 8);
    % Combine all parts
    header = [packet_num_bin total_packets_bin flags];
end

% Custom CRC calculation
function crc = custom_crc_calculate(data)
    polynomial = [1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1]; % CRC-16-CCITT
    crc = ones(1, 16);
    
    for i = 1:length(data)
        feedback = xor(data(i), crc(1));
        crc = [crc(2:end) 0];
        if feedback
            crc = xor(crc, polynomial);
        end
    end
end

% Custom Hamming encoding
function encoded_data = custom_hamming_encode(data)
    % Implementation of (7,4) Hamming code
    G = [1 1 0 1 0 0 0;  % Generator matrix
         1 0 1 1 0 1 0;
         1 0 0 0 1 1 1;
         0 1 1 1 1 0 0];
    
    % Reshape data into 4-bit blocks
    data_reshape = reshape([data zeros(1, mod(-length(data), 4))], [], 4);
    encoded_data = zeros(size(data_reshape, 1), 7);
    
    % Encode each block
    for i = 1:size(data_reshape, 1)
        encoded_data(i, :) = mod(data_reshape(i, :) * G, 2);
    end
    
    encoded_data = encoded_data(:)';
end

% Custom data whitening
function whitened_data = custom_data_whitening(data)
    % Generate whitening sequence using LFSR
    polynomial = [1 0 0 0 0 1 1]; % x^6 + x + 1
    lfsr_state = [1 1 1 1 1 1];   % Initial state
    whitening_sequence = zeros(1, length(data));
    
    for i = 1:length(data)
        whitening_sequence(i) = lfsr_state(end);
        feedback = xor(lfsr_state(1), lfsr_state(5));
        lfsr_state = [lfsr_state(2:end) feedback];
    end
    
    % Apply whitening
    whitened_data = xor(data, whitening_sequence);
end

% Custom GFSK modulation
function modulated_signal = custom_gfsk_modulate(data, h_index, samples_per_bit)
    % Create Gaussian filter
    gaussian_filter = custom_gaussian_filter(samples_per_bit);
    
    % Upsample data
    upsampled_data = zeros(1, length(data) * samples_per_bit);
    upsampled_data(1:samples_per_bit:end) = 2 * data - 1; % NRZ coding
    
    % Apply Gaussian filter
    filtered_data = conv(upsampled_data, gaussian_filter, 'same');
    
    % Frequency modulation
    phase = cumsum(filtered_data) * pi * h_index;
    modulated_signal = exp(1j * phase);
end

% Custom Gaussian filter design
function h = custom_gaussian_filter(samples_per_bit)
    BT = 0.5;  % Bandwidth-time product
    span = 4;  % Filter span in symbol periods
    t = (-span/2:1/samples_per_bit:span/2);
    alpha = sqrt(log(2)/(2*(BT^2)));
    h = exp(-((alpha*t).^2));
    h = h / sum(h); % Normalize
end

% Custom channel simulation
function received_signal = custom_channel_simulation(transmitted_signal, SNR_dB)
    % Calculate noise power
    signal_power = mean(abs(transmitted_signal).^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    
    % Generate complex AWGN
    noise = sqrt(noise_power/2) * (randn(size(transmitted_signal)) + ...
            1j*randn(size(transmitted_signal)));
    
    % Add noise to signal
    received_signal = transmitted_signal + noise;
end

% Custom demodulation and decoding
function decoded_data = custom_demodulate_decode(received_signal, samples_per_bit)
    % Phase demodulation
    phase_diff = angle(received_signal(2:end) .* conj(received_signal(1:end-1)));
    
    % Downsample
    downsampled_phase = phase_diff(1:samples_per_bit:end);
    
    % Decision
    decoded_bits = downsampled_phase > 0;
    
    % Remove whitening
    dewhitened_data = custom_data_whitening(decoded_bits); % Same function as whitening
    
    % Hamming decode
    decoded_data = custom_hamming_decode(dewhitened_data);
end

% Custom Hamming decoding
function decoded_data = custom_hamming_decode(data)
    % Parity check matrix
    H = [1 0 1 0 1 0 1;
         0 1 1 0 0 1 1;
         0 0 0 1 1 1 1];
    
    % Reshape data into 7-bit blocks
    data_blocks = reshape([data zeros(1, mod(-length(data), 7))], [], 7);
    decoded_blocks = zeros(size(data_blocks, 1), 4);
    
    % Decode each block
    for i = 1:size(data_blocks, 1)
        % Calculate syndrome
        syndrome = mod(data_blocks(i, :) * H', 2);
        
        % Error correction
        if any(syndrome)
            error_pos = bi2de(syndrome', 'left-msb');
            if error_pos <= 7
                data_blocks(i, error_pos) = ~data_blocks(i, error_pos);
            end
        end
        
        % Extract data bits
        decoded_blocks(i, :) = data_blocks(i, [3 5 6 7]);
    end
    
    decoded_data = decoded_blocks(:)';
end

% Custom function to convert binary stream back to image
function img = custom_binary_to_image(binary_stream, img_size)
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

% Example usage:
% img = imread('test_image.jpg');
% SNR_dB = 20;
% [received_img, bit_error_rate] = custom_ble_process(img, SNR_dB);
% figure;
% subplot(1,2,1); imshow(img); title('Original Image');
% subplot(1,2,2); imshow(uint8(received_img)); title(['Received Image (BER: ' num2str(bit_error_rate) ')']);