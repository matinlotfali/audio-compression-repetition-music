import numpy as np
import mdct
import io
from tqdm.auto import trange


def peripheral_adjustment(mdct_data, round_decimal_threshold=4, zero_threshold=0.001):
    # mdct_data = np.round(mdct_data, decimals=round_decimal_threshold)
    # ramp = np.arange(512)/512
    if zero_threshold > 0:
        mdct_data = np.where(abs(mdct_data) < zero_threshold, 0, mdct_data)
    return mdct_data


def encode_mdct(raw_wave, sample_rate, npz_file,
                cut_threshold=400,
                zero_threshold=0.001,
                lossy=True):

    def encode_frame(n):
        return peripheral_adjustment(data[n], zero_threshold=zero_threshold * max(data[n]))

    data = mdct.mdct(raw_wave)
    data = np.float16(data)
    data = np.transpose(data)
    if lossy:
        # data[:, cut_threshold:] = 0
        for i in trange(len(data), desc='MDCT', leave=False):
            data[i] = encode_frame(i)
    np.savez_compressed(npz_file, data=data, sample_rate=[sample_rate])
    return data


def calc_errors(data, lossy):
    if lossy:
        # return np.count_nonzero((data != 0) & (data != -15.945))
        return np.count_nonzero(data)
    else:
        compressed_array = io.BytesIO()
        np.savez_compressed(compressed_array, data)
        return len(compressed_array.getvalue())


def calc_diff(frame1, frame2):
    # eps = 1e-7
    # frame1 = np.log(frame1**2 + eps)
    # frame2 = np.log(frame2**2 + eps)
    diff = frame1 - frame2
    return diff


def encode_mdct_diff(raw_wave, sample_rate, npz_file,
                     zero_threshold_i_frame=0.000,
                     zero_threshold_p_frame=0.001,
                     cut_threshold=400,
                     i_rate=2,
                     trick=False,
                     lossy=True):

    def encode_frame_diff(n):
        best_p = 0
        base = peripheral_adjustment(ref[n], zero_threshold=zero_threshold_i_frame * max(ref[n])) if lossy else ref[n]
        best_diff = base
        least_err = calc_errors(best_diff, lossy)
        initial_err = least_err
        if least_err > 0:
            for p in range(1, n):

                if (n - p) % (i_rate * sample_rate // 512) == 0:
                    break

                diff = calc_diff(ref[n], ref[n - p])
                if lossy:
                    diff = peripheral_adjustment(diff, zero_threshold=zero_threshold_p_frame)

                # for i in range(512):
                #     if base[i] == 0 and diff[i] != 0:
                #         diff[i] = -10
                if trick:
                    diff = np.where((base == 0) & (diff != 0), -10, diff)

                err = calc_errors(diff, lossy)

                if err < least_err:
                    least_err = err
                    best_p = p
                    best_diff = diff

                if err == 0:
                    break

        return best_p, best_diff, initial_err - least_err

    data = mdct.mdct(raw_wave)
    data = np.float16(data)
    data = np.transpose(data)
    # if lossy:
        # data[:, cut_threshold:] = 0
    # eps = 1e-7
    # signs = np.int8(np.sign(data))
    # data = np.log(data ** 2 + eps)

    ref = data.copy()
    relative_indices = np.zeros(len(data), dtype=np.int16)
    improves = np.zeros(len(data), dtype=np.int16)

    for i in trange(len(data), desc='MDCT DIFF', leave=False):
        relative_indices[i], data[i], improves[i] = encode_frame_diff(i)

    np.savez_compressed(npz_file, data=data, relative=relative_indices, sample_rate=[sample_rate])#, signs=signs)
    return data, relative_indices, improves


def decode_mdct(encoded_data):
    encoded_data = np.transpose(encoded_data)
    return mdct.imdct(encoded_data)


def decode_mdct_diff(loaded, trick=False):
    encoded_data = loaded['data']
    relative_indices = loaded['relative']
    # signs = loaded['signs']

    for n in trange(1, len(encoded_data), desc='MDCT DIFF Decode', leave=False):
        p = relative_indices[n]
        if relative_indices[n] > 0:
            # encoded_data[n] = signs[n] * np.sqrt(np.exp(encoded_data[n] + prev_frame))
            encoded_data[n] += encoded_data[n-p]

            if trick:
                diff = encoded_data[n].copy()
                encoded_data[n] = np.where(diff <= -5, 0, encoded_data[n])
                # for i in range(512):
                #     if diff[i] <= -5:
                #         encoded_data[n][i] = 0

    # encoded_data = signs * np.sqrt(np.exp(encoded_data))
    return decode_mdct(encoded_data)
