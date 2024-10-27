% Main function with visualization of each stage
function visualize_ble_stage()
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
    stem(1:10, binary_data(1:10), 'LineWidth', 2);
    title('Binary Data (First 10 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    ylim([-0.2 1.2]);
    
    % 3. Packetize and encode with Hamming code
    fprintf('Stage 2: Packetization and Encoding\n');
    packet_size = 256;
    packets = custom_packetization(binary_data, packet_size);
    encoded_data = custom_hamming_encode(packets);
    
    subplot(3,3,4);
    stem(1:10, encoded_data(1:10), 'LineWidth', 2);
    title('Encoded Data (First 10 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    ylim([-0.2 1.2]);
    
    % 4. Data Whitening
    fprintf('Stage 3: Data Whitening\n');
    whitened_data = custom_data_whitening(encoded_data);
    
    subplot(3,3,5);
    stem(1:10, whitened_data(1:10), 'LineWidth', 2);
    title('Whitened Data (First 10 bits)');
    xlabel('Bit Index');
    ylabel('Bit Value');
    grid on;
    ylim([-0.2 1.2]);
    
    % 5. GFSK Modulation
    fprintf('Stage 4: GFSK Modulation\n');
    samples_per_bit = 8;
    h_index = 0.5;
    modulated_signal = custom_gfsk_modulate(whitened_data, h_index, samples_per_bit);
    
    subplot(3,3,6);
    plot(1:80, real(modulated_signal(1:80)), 'LineWidth', 2);
    hold on;
    plot(1:80, imag(modulated_signal(1:80)), 'LineWidth', 2);
    title('Modulated Signal (First 10 bits)');
    xlabel('Sample Index');
    ylabel('Amplitude');
    legend('Real', 'Imag');
    grid on;
    
    % Display the first few bits of each stage
    fprintf('\nFirst 10 bits at each stage:\n');
    fprintf('Binary data: ');
    fprintf('%d ', binary_data(1:10));
    fprintf('\nEncoded data: ');
    fprintf('%d ', encoded_data(1:10));
    fprintf('\nWhitened data: ');
    fprintf('%d ', whitened_data(1:10));
    fprintf('\n');
end

function binary_data = image_to_binary(img)
    % Convert each pixel to 8-bit binary
    img_vector = double(img(:));
    binary_data = zeros(1, length(img_vector) * 8);
    
    for i = 1:length(img_vector)
        pixel_val = img_vector(i);
        for bit = 1:8
            bit_idx = (i-1)*8 + bit;
            binary_data(bit_idx) = bitget(uint8(pixel_val), 9-bit);
        end
    end
end

function packets = custom_packetization(data, packet_size)
    bits_per_packet = packet_size * 8;
    num_packets = ceil(length(data) / bits_per_packet);
    packets = zeros(num_packets, bits_per_packet + 32);
    
    for i = 1:num_packets
        header = ones(1, 32);  % Changed to ones for visibility
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
    data_flat = data(:)';
    remainder = mod(length(data_flat), 4);
    if remainder > 0
        data_flat = [data_flat zeros(1, 4-remainder)];
    end
    
    % Reshape data into 4-bit blocks
    data_blocks = reshape(data_flat, 4, [])';
    encoded_data = zeros(size(data_blocks, 1), 7);
    
    % Encode each 4-bit block into 7-bit Hamming code
    for i = 1:size(data_blocks, 1)
        block = data_blocks(i, :);
        % Compute parity bits
        p1 = mod(sum(block([1 2 4])), 2);
        p2 = mod(sum(block([1 3 4])), 2);
        p3 = mod(sum(block([2 3 4])), 2);
        encoded_data(i, :) = [block(1:4) p1 p2 p3];
    end
    
    encoded_data = encoded_data(:)';
end

function whitened_data = custom_data_whitening(data)
    % Generate whitening sequence using LFSR
    whitening_seq = zeros(size(data));
    lfsr = [1 1 1 1 1 1 1];  % 7-bit LFSR initial state
    
    for i = 1:length(data)
        whitening_seq(i) = lfsr(end);
        new_bit = mod(sum(lfsr .* [1 0 0 0 1 0 1]), 2);  % Polynomial: x^7 + x^4 + x^1
        lfsr = [new_bit lfsr(1:end-1)];
    end
    
    whitened_data = xor(data, whitening_seq);
end

function modulated_signal = custom_gfsk_modulate(data, h_index, samples_per_bit)
    % Create upsampled NRZ data manually
    upsampled_data = zeros(1, length(data) * samples_per_bit);
    for i = 1:length(data)
        start_idx = (i-1) * samples_per_bit + 1;
        end_idx = i * samples_per_bit;
        upsampled_data(start_idx:end_idx) = 2 * data(i) - 1;
    end
    
    % Gaussian filter design
    BT = 0.5;
    span = 4;
    t = (-span/2:1/samples_per_bit:span/2);
    alpha = sqrt(log(2)/(2*(BT^2)));
    h_gaussian = exp(-((alpha*t).^2));
    h_gaussian = h_gaussian / sum(h_gaussian);
    
    % Apply Gaussian filter
    filtered_data = conv(upsampled_data, h_gaussian, 'same');
    
    % FM modulation
    phase = cumsum(filtered_data) * pi * h_index;
    modulated_signal = exp(1j * phase);
end

% Run the visualization
