def lookup_table(angle, input):

    table = {-16: [[0.107, 100], [0.28, 150], [0.45, 200], [0.515, 250], [0.548, 300], [0.44, 350], [0.37, 400]],
             -15: [[0.115, 100], [0.28, 150], [0.48, 200], [0.542, 250], [0.57, 300], [0.46, 350], [0.38, 400]],
             -14: [[0.11, 100], [0.295, 150], [0.51, 200], [0.562, 250], [0.585, 300], [0.49, 350], [0.4, 400]],
             -13: [[0.115, 100], [0.3, 150], [0.58, 200], [0.595, 250], [0.62, 300], [0.49, 350], [0.44, 400]],
             -12: [[0.117, 100], [0.305, 150], [0.64, 200], [0.608, 250], [645, 300], [0.5, 350], [0.47, 400]],
             -11: [[0.116, 100], [0.325, 150], [0.665, 200], [0.62, 250], [0.67, 300], [0.51, 350], [0.49, 400]],
             -10: [[0.115, 100], [0.34, 150], [0.7, 200], [0.635, 250], [0.7, 300], [0.54, 350], [0.5, 400]],
             -9: [[0.115, 100], [0.375, 150], [0.72, 200], [0.66, 250], [0.72, 300], [0.59, 350], [0.51, 400]],
             -8: [[0.114, 100], [0.4, 150], [0.725, 200], [0.689, 250], [0.725, 300], [0.62, 350], [0.53, 400]],
             -7: [[0.112, 100], [0.45, 150], [0.74, 200], [0.71, 250], [0.725, 300], [0.63, 350], [0.54, 400]],
             -6: [[0.11, 100], [0.45, 150], [0.76, 200], [0.745, 250], [0.735, 300], [0.65, 350], [0.59, 400]],
             -5: [[0.11, 100], [0.47, 150], [0.79, 200], [0.765, 250], [0.74, 300], [0.68, 350], [0.63, 400]],
             -4: [[0.112, 100], [0.49, 150], [0.8, 200], [0.785, 250], [0.745, 300], [0.72, 350], [0.65, 400]],
             -3: [[0.115, 100], [0.51, 150], [0.805, 200], [0.79, 250], [0.75, 300], [0.76, 350], [0.65, 400]],
             -2: [[0.119, 100], [0.53, 150], [0.804, 200], [0.8, 250], [0.76, 300], [0.8, 350], [0.65, 400]],
             -1: [[0.12, 100], [0.55, 150], [0.82, 200], [0.82, 250], [0.78, 300], [0.84, 350], [0.67, 400]],
             0: [[0.125, 100], [0.58, 150], [0.83, 200], [0.89, 250], [0.79, 300], [0.86, 350], [0.7, 400]],
             1: [[0.14, 100], [0.6, 150], [0.8, 200], [0.97, 250], [0.81, 300], [0.88, 350], [0.76, 400]],
             2: [[0.147, 100], [0.65, 150], [0.77, 200], [0.98, 250], [0.83, 300], [0.89, 350], [0.76, 400]]}

    if input >= 1.1:   # Means we are trying to find the RMS value
        # use pulsewidth

        minimum_difference = 1000000
        found_rms = 0

        for pair in table[angle]:
            pw_val = pair[1]
            if abs(input-pw_val) < minimum_difference:
                minimum_difference = abs(input-pw_val)
                found_rms = pair[0]

        return found_rms

    else:   # Means we are trying to find the pulsewidth value
        # use rms
        pair_index = 0
        minimum_difference = 1000000
        found_pulsewidth = 0

        for pair in table[angle]:
            rms_val = pair[pair_index]
            if abs(input-rms_val) < minimum_difference:
                minimum_difference = abs(input-rms_val)
                found_pulsewidth = pair[1]

        return found_pulsewidth
