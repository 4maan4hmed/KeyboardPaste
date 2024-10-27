% --- IMAGE PROCESSING, MODULATION, AND TRANSMISSION SIMULATION ---

clc; clear; close all;

% STEP 1: READ AND COMPRESS THE IMAGE (JPEG-Like Compression)
image = imread('peppers.png'); % Use one of MATLAB's built-in images
grayImage = rgb2gray(image); % Convert to grayscale
imshow(grayImage);
title('Original Image');

% Image Compression using Discrete Cosine Transform (DCT)
dctImage = dct2(grayImage); % Perform 2D DCT
dctThreshold = 20; % Zero out small DCT coefficients for compression
compressedDCT = dctImage;
compressedDCT(abs(compressedDCT) < dctThreshold) = 0; % Compression by thresholding
compressedImage = uint8(idct2(compressedDCT)); % Reconstruct compressed image
figure;
imshow(compressedImage);
title('Compressed Image');

% STEP 2: CONVERT COMPRESSED IMAGE TO BINARY STREAM
binaryImage = dec2bin(compressedImage(:), 8); % 8-bit binary conversion of compressed image
binaryImage = binaryImage(:); % Flatten to 1D

% STEP 3: ADD ERROR CORRECTION (HAMMING CODE)
% Hamming(7,4) encoding to add redundancy for error correction
encodedData = [];
for i = 1:4:length(binaryImage) - 3
    nibble = binaryImage(i:i+3)'; % Take 4 bits
    encodedData = [encodedData; encodeHamming74(nibble')]; % Hamming encode
end

% STEP 4: MODULATION (16-QAM)
M = 16; % 16-QAM modulation
dataSymbols = binaryToDecimal(reshape(encodedData, log2(M), [])); % Custom binary-to-decimal conversion
modulatedSignal = qammod(dataSymbols, M); % QAM modulation

% STEP 5: TRANSMISSION THROUGH A SIMULATED NOISY CHANNEL
SNR = 15; % Signal-to-noise ratio in dB

% Use comm.RayleighChannel for Rayleigh fading simulation
rayleighChan = comm.RayleighChannel('SampleRate', 1e5, 'MaximumDopplerShift', 30);

% Apply fading
fadedSignal = rayleighChan(modulatedSignal);

% Add AWGN noise to the signal
noisySignal = awgn(fadedSignal, SNR, 'measured');

% STEP 6: CHANNEL EQUALIZATION (ZERO-FORCING)
eqSignal = noisySignal ./ rayleighChan.PathGains; % Zero-forcing equalization

% STEP 7: DEMODULATION (16-QAM)
receivedSymbols = qamdemod(eqSignal, M); % Demodulate QAM
receivedBinaryData = decimalToBinary(receivedSymbols, log2(M)); % Custom decimal-to-binary conversion
receivedBinaryData = receivedBinaryData(:); % Flatten the stream

% STEP 8: ERROR CORRECTION DECODING (HAMMING CODE)
decodedData = [];
for i = 1:7:length(receivedBinaryData)
    hammingCode = receivedBinaryData(i:i+6); % Take 7-bit Hamming codeword
    decodedData = [decodedData; decodeHamming74(hammingCode')]; % Decode using Hamming(7,4)
end

% STEP 9: CONVERT BINARY STREAM BACK TO IMAGE FORMAT
receivedBinaryImage = num2str(decodedData(:));
receivedImage = reshape(uint8(bin2dec(reshape(receivedBinaryImage, 8, [])')), size(compressedImage));

% STEP 10: IMAGE FILTERING (POST-PROCESSING)
h = fspecial('average', [3, 3]); % Low-pass filter to reduce noise
filteredImage = imfilter(receivedImage, h);

% STEP 11: DISPLAY FINAL RECONSTRUCTED IMAGE
figure;
imshow(filteredImage);
title('Reconstructed Image after Transmission with DSP');




function encodedBits = encodeHamming74(dataBits)
    dataBits = dataBits(:)'; % Ensure dataBits is a row vector
    G = [1 0 0 0 1 1 0; 
         0 1 0 0 1 0 1; 
         0 0 1 0 1 1 1; 
         0 0 0 1 0 1 1]; % Generator matrix
    encodedBits = mod(dataBits * G, 2); % Multiply data by generator matrix
end

% Hamming(7,4) Decoding
function decodedBits = decodeHamming74(codeword)
    H = [1 1 1 0 1 0 0; 
         1 0 1 1 0 1 0; 
         0 1 1 1 0 0 1]; % Parity-check matrix
    syndrome = mod(codeword * H', 2); % Compute syndrome
    errorPosition = bin2dec(num2str(syndrome)) + 1; % Find error position
    if errorPosition <= 7
        codeword(errorPosition) = ~codeword(errorPosition); % Correct error
    end
    decodedBits = codeword(1:4); % Extract original 4 bits
end

% Custom Binary to Decimal Conversion
function decimal = binaryToDecimal(binaryMatrix)
    [rows, cols] = size(binaryMatrix); % Get size of binary matrix
    decimal = zeros(1, cols); % Initialize output array
    for i = 1:cols
        decimal(i) = sum(binaryMatrix(:, i)' .* 2.^(rows-1:-1:0)); % Binary to decimal conversion
    end
end

% Custom Decimal to Binary Conversion
function binaryMatrix = decimalToBinary(decimalArray, bitLength)
    binaryMatrix = zeros(bitLength, length(decimalArray)); % Initialize output matrix
    for i = 1:length(decimalArray)
        binaryMatrix(:, i) = de2bi(decimalArray(i), bitLength, 'left-msb')'; % Convert decimal to binary
    end
end
