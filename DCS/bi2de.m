function decimal_value = bi2de(binary_vector, msb_position)


    if nargin < 2
        msb_position = 'right-msb'; % Default is right-msb
    end

    % Validate the MSB position
    if ~ismember(msb_position, {'left-msb', 'right-msb'})
        error('Invalid MSB position. Use "left-msb" or "right-msb".');
    end

    [rows, cols] = size(binary_vector);
    decimal_value = zeros(rows, 1); % Pre-allocate output

    for i = 1:rows
        switch msb_position
            case 'right-msb'
                % Convert binary to decimal with right-most bit as LSB
                decimal_value(i) = sum(binary_vector(i, :) .* (2.^(0:cols-1)));
            case 'left-msb'
                % Convert binary to decimal with left-most bit as MSB
                decimal_value(i) = sum(binary_vector(i, :) .* (2.^(cols-1:-1:0)));
        end
    end
end
