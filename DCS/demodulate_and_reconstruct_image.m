function demodulate_and_reconstruct_image(modulated_signal, samples_per_bit, h_index, packet_size)
    % Demodulation Stage
    fprintf('Stage 5: GFSK Demodulation\n');
    demodulated_data = custom_gfsk_demodulate(modulated_signal, h_index, samples_per_bit);

    % Data De-whitening
    fprintf('Stage 6: Data De-whitening\n');
    dewhitened_data = custom_data_whitening(demodulated_data);

    % Hamming Decode
    fprintf('Stage 7: Hamming Decoding\n');
    decoded_data = custom_hamming_decode(dewhitened_data);

    % Packet Reconstruction
    fprintf('Stage 8: Packet Reconstruction\n');
    reconstructed_binary_data = custom_packet_reconstruction(decoded_data, packet_size);

    % Convert Binary to Image
    fprintf('Stage 9: Binary to Image Conversion\n');
    reconstructed_image = binary_to_image(reconstructed_binary_data, [512 512]); % Assuming a 512x512 grayscale image

    % Display Reconstructed Image
    figure;
    imshow(reconstructed_image);
    title('Reconstructed Image');
end

% Demodulation function
function demodulated_data = custom_gfsk_demodulate(modulated_signal, h_index, samples_per_bit)
    % Phase extraction and differentiation
    phase = angle(modulated_signal);
    phase_diff = diff([0 phase]); % Calculate the phase difference

    % Normalize the phase difference
    demodulated_signal = phase_diff / (pi * h_index);

    % Sample the signal to get the data bits
    num_bits = floor(length(demodulated_signal) / samples_per_bit);
    demodulated_data = zeros(1, num_bits);
    for i = 1:num_bits
        sample_idx = (i - 1) * samples_per_bit + round(samples_per_bit / 2);
        if demodulated_signal(sample_idx) > 0
            demodulated_data(i) = 1;
        else
            demodulated_data(i) = 0;
        end
    end
end

% Hamming decoding function (reverse of encoding)
function decoded_data = custom_hamming_decode(encoded_data)
    encoded_blocks = reshape(encoded_data, 7, [])';
    decoded_data = zeros(size(encoded_blocks, 1), 4);
    
    for i = 1:size(encoded_blocks, 1)
        block = encoded_blocks(i, :);
        syndrome = [mod(sum(block([1 3 5 7])), 2), ...
                    mod(sum(block([2 3 6 7])), 2), ...
                    mod(sum(block([4 5 6 7])), 2)];
        error_pos = bin2dec(num2str(syndrome));
        
        if error_pos > 0
            block(error_pos) = mod(block(error_pos) + 1, 2); % Correct the error
        end
        
        % Extract the original 4 data bits
        decoded_data(i, :) = block(1:4);
    end
    
    decoded_data = decoded_data(:)';
end

% Packet reconstruction
function reconstructed_binary_data = custom_packet_reconstruction(data, packet_size)
    bits_per_packet = packet_size * 8;
    packet_size_with_header = bits_per_packet + 32;  % 32-bit header

    num_packets = length(data) / packet_size_with_header;
    reconstructed_binary_data = zeros(1, num_packets * bits_per_packet);

    for i = 1:num_packets
        start_idx = (i - 1) * packet_size_with_header + 1 + 32;  % Skip header
        end_idx = start_idx + bits_per_packet - 1;
        packet_data = data(start_idx:end_idx);
        reconstructed_binary_data((i - 1) * bits_per_packet + 1 : i * bits_per_packet) = packet_data;
    end
end

% Convert binary data back to an image
function img = binary_to_image(binary_data, img_size)
    img_vector = zeros(1, length(binary_data) / 8);
    for i = 1:length(img_vector)
        start_idx = (i - 1) * 8 + 1;
        end_idx = i * 8;
        img_vector(i) = bin2dec(num2str(binary_data(start_idx:end_idx)));
    end
    
    img = reshape(uint8(img_vector), img_size(1), img_size(2));
end
