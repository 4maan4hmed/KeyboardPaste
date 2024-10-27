function received_image = receive_ble_signal(I_signal, Q_signal, original_image_size, params)
    % Set default parameters if not provided
    if nargin < 4
        params.samples_per_bit = 8;
        params.modulation_index = 0.5;
        params.packet_size = 256;
    end 
    
    % Create figure for visualization
    figure('Name', 'BLE Reception Stages');
    
    % Stage 1: Signal Processing and Demodulation
    fprintf('Stage 1: GFSK Demodulation\n');
    received_bits = gfsk_demodulate(I_signal, Q_signal, params.samples_per_bit, params.modulation_index);
    
    % Plot received constellation
    subplot(4, 2, 1);
    plot_constellation(I_signal(1:1000), Q_signal(1:1000));
    
    % Plot demodulated bits
    subplot(4, 2, 2);
    plot_binary_sample(received_bits(1:100), 'Demodulated Bits');
    
    % Stage 2: De-whitening
    fprintf('Stage 2: Data De-whitening\n');
    dewhitened_data = apply_data_dewhitening(received_bits);
    
    subplot(4, 2, 3);
    plot_binary_sample(dewhitened_data(1:100), 'De-whitened Data');
    
    % Stage 3: Hamming Decoding
    fprintf('Stage 3: Hamming Decoding\n');
    decoded_data = apply_hamming_decoding(dewhitened_data);
    
    subplot(4, 2, 4);
    plot_binary_sample(decoded_data(1:100), 'Decoded Data');
    
    % Stage 4: Depacketization
    fprintf('Stage 4: Depacketization\n');
    [depacketized_data, packet_success_rate] = depacketize_data(decoded_data, params.packet_size);
    
    subplot(4, 2, 5);
    visualize_packet_success(packet_success_rate, 'Packet Reception Success');
    
    % Stage 5: Binary to Image Conversion
    fprintf('Stage 5: Image Reconstruction\n');
    received_image = binary_to_image(depacketized_data, original_image_size);
    
    % Display reconstructed image
    subplot(4, 2, 6);
    imshow(received_image);
    title('Reconstructed Image');
    
    % Display error metrics
    subplot(4, 2, [7,8]);
    display_error_metrics(received_image, original_image_size);
end

function demodulated_bits = gfsk_demodulate(I_signal, Q_signal, samples_per_bit, h_index)
    % Calculate phase from I/Q
    phase = unwrap(atan2(Q_signal, I_signal));
    
    % Differentiate phase to get frequency deviation
    freq_dev = diff(phase);
    freq_dev = [freq_dev(1) freq_dev];  % Pad to maintain length
    
    % Apply matched filter
    span = 4;
    beta = 0.5;  % Gaussian filter parameter
    t = (-span/2:1/samples_per_bit:span/2);
    matched_filter = exp(-(t.^2)/(2*beta^2));
    matched_filter = matched_filter/sum(matched_filter);
    
    filtered_signal = conv(freq_dev, matched_filter, 'same');
    
    % Sample at bit centers
    bit_centers = samples_per_bit/2:samples_per_bit:length(filtered_signal);
    sampled_signal = filtered_signal(round(bit_centers));
    
    % Decision threshold
    demodulated_bits = sampled_signal > 0;
end

function dewhitened_data = apply_data_dewhitening(data)
    % Initialize LFSR with same seed as transmitter
    lfsr = [1 0 1 0 1 0 1];
    whitening_sequence = zeros(size(data));
    
    for i = 1:length(data)
        whitening_sequence(i) = lfsr(end);
        feedback = xor(lfsr(7), lfsr(4));
        lfsr = [feedback lfsr(1:end-1)];
    end
    
    % Apply de-whitening (XOR operation is its own inverse)
    dewhitened_data = xor(data, whitening_sequence);
end

function decoded_data = apply_hamming_decoding(data)
    % Hamming(7,4) parity-check matrix
    H = [1 0 1 0 1 0 1;
         0 1 1 0 0 1 1;
         0 0 0 1 1 1 1];
    
    % Process data in 7-bit blocks
    num_blocks = floor(length(data)/7);
    decoded_data = zeros(1, num_blocks * 4);
    
    for i = 1:num_blocks
        % Extract block
        block = data((i-1)*7 + 1 : i*7);
        
        % Calculate syndrome
        syndrome = mod(H * block', 2);
        
        % Error correction
        if any(syndrome)
            error_pos = bi2de(syndrome');
            if error_pos <= 7
                block(error_pos) = ~block(error_pos);
            end
        end
        
        % Extract data bits (positions 3,5,6,7 in Hamming(7,4))
        decoded_data((i-1)*4 + 1 : i*4) = block([3 5 6 7]);
    end
end

function [depacketized_data, packet_success_rate] = depacketize_data(data, packet_size)
    % BLE packet parameters
    preamble = [1 0 1 0 1 0 1 0];
    preamble_length = length(preamble);
    access_address_length = 32;
    header_length = preamble_length + access_address_length;
    
    % Calculate total packet length including header
    total_packet_length = header_length + packet_size * 8;
    
    % Split data into packets
    num_packets = floor(length(data)/total_packet_length);
    valid_packets = zeros(1, num_packets);
    depacketized_data = [];
    
    for i = 1:num_packets
        % Extract packet
        packet_start = (i-1)*total_packet_length + 1;
        packet_end = i*total_packet_length;
        packet = data(packet_start:packet_end);
        
        % Check preamble
        received_preamble = packet(1:preamble_length);
        if isequal(received_preamble, preamble)
            valid_packets(i) = 1;
            % Extract payload (skip header)
            payload = packet(header_length+1:end);
            depacketized_data = [depacketized_data payload];
        end
    end
    
    packet_success_rate = mean(valid_packets);
end

function reconstructed_image = binary_to_image(binary_data, original_size)
    % Convert binary data to uint8 pixels
    num_pixels = prod(original_size);
    pixel_values = zeros(1, num_pixels);
    
    for i = 1:num_pixels
        bit_start = (i-1)*8 + 1;
        if bit_start+7 <= length(binary_data)
            pixel_values(i) = bi2de(binary_data(bit_start:bit_start+7));
        end
    end
    
    % Reshape into image
    reconstructed_image = reshape(pixel_values, original_size);
    reconstructed_image = uint8(reconstructed_image);
end

function plot_constellation(I, Q)
    scatter(I, Q, 10, 'filled');
    title('Received Signal Constellation');
    xlabel('I');
    ylabel('Q');
    grid on;
    axis equal;
end

function plot_binary_sample(data, title_str)
    stem(1:length(data), data, 'LineWidth', 1.5);
    title(title_str);
    xlabel('Bit Index');
    ylabel('Bit Value');
    ylim([-0.2 1.2]);
    grid on;
end

function visualize_packet_success(success_rate, title_str)
    bar(success_rate * 100);
    title(title_str);
    xlabel('Overall');
    ylabel('Success Rate (%)');
    ylim([0 100]);
    grid on;
end

function display_error_metrics(received_image, original_size)
    text_str = sprintf(['Reception Metrics:\n' ...
                       'Image Size: %dx%d\n' ...
                       'Total Pixels: %d\n' ...
                       'Packet Success Rate: %.1f%%\n'], ...
                       original_size(1), original_size(2), ...
                       prod(original_size), ...
                       mean(received_image(:) > 0) * 100);
    text(0.1, 0.5, text_str, 'FontSize', 12);
    axis off;
end
