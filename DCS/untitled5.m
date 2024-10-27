% Parameters: Use the same settings as the modulation step
samples_per_bit = 8; % Number of samples per bit used in modulation
h_index = 0.5;       % Modulation index used in GFSK modulation
packet_size = 256;   % Packet size used during packetization

% Reconstruct the image from the modulated signal
demodulate_and_reconstruct_image(modulated_signal, samples_per_bit, h_index, packet_size);
