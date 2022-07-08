import warnings
warnings.simplefilter('ignore')
import datetime

# import matplotlib.pyplot as plt
# import librosa
# import soundfile as sf
# from IPython.display import Audio
# from ipywidgets import widgets, Layout

# def draw_freq_spectogram(data, rate, duration, title, size=(4, 6)):
#     data = data[:, :int(rate * duration) // 512]
#     data = np.float64(data)
#     eps = 1e-10
#     data = np.log(eps + data ** 2)
#     data = np.flipud(data)
#
#     plt.figure(figsize=size)
#     plt.title(title)
#     plt.xlabel('Time')
#     plt.ylabel('Frequency Index')
#     plt.imshow(data, aspect='auto', extent=[0, duration, 0, 512])
#     plt.show()


file = "audio/repet/historyrepeating_7olLrex"
# audio_path = nussl.efz_utils.download_audio_file('historyrepeating_7olLrex.wav')

print('Importing modules...')

import librosa
import Encoder
import soundfile as sf
import ffmpeg
import os
import io
import numpy as np
from tqdm.auto import tqdm, trange
import matlab.engine
import pandas as pd
import nussl


def encode_ffmpeg(input_file, output_file):
    ffmpeg.input(input_file).output(output_file, audio_bitrate='128k').run(quiet=True, overwrite_output=True)


def process_lossless(wave_file, dir, t, f):
    wave_size = os.path.getsize(wave_file)

    encode_ffmpeg(wave_file, dir+'/raw.flac')
    size = os.path.getsize(dir+'/raw.flac')
    f.write(f'{size * 100 / wave_size},')
    t.update()

    data, sr = librosa.load(wave_file)
    Encoder.encode_mdct(data, sr, dir+'/raw_mdct_lossless.npz', lossy=False)
    size_mdct = os.path.getsize(dir+'/raw_mdct_lossless.npz')
    f.write(f'{size_mdct * 100 / wave_size},')
    t.update()

    data, sr = librosa.load(wave_file)
    Encoder.encode_mdct_diff(data, sr, dir+'/raw_mdct_diff_lossless.npz', lossy=False)
    size_mdct_diff = os.path.getsize(dir+'/raw_mdct_diff_lossless.npz')
    f.write(f'{size_mdct_diff * 100 / wave_size},')
    t.update()


def run_musdb18():
    def process_IV_Input(musdb_item, df: pd.DataFrame) -> pd.DataFrame:

        def process_DVs(decoded_data, sample_rate, input_type, frame_type, relative_indices: np.ndarray = None) -> pd.Series:
            sf.write('tmp/decoded.wav', decoded_data, sample_rate)
            ffmpeg.input('tmp/decoded.wav', ss=50, t=10).output('tmp/test.wav', ac=1, ar='48K').run(quiet=True,
                                                                                                    overwrite_output=True)
            result = pd.Series()
            result['input-type'] = input_type
            result['frame_type'] = frame_type
            if input_type == 'sum':
                result['compression_ratio'] = (os.path.getsize('tmp/encoded_fg.npz') + os.path.getsize('tmp/encoded_bg.npz')) / os.path.getsize('tmp/decoded.wav') / 2
            else:
                result['compression_ratio'] = os.path.getsize('tmp/encoded.npz') / os.path.getsize('tmp/decoded.wav')
            result['peaq'] = eng.PQevalAudio('tmp/ref.wav', 'tmp/test.wav', stdout=io.StringIO())
            result['desirability'] = result['peaq'] / 4 / result['compression_ratio']
            if relative_indices is not None:
                result['p-ratio'] = np.count_nonzero(relative_indices) / len(relative_indices)
                result['mean-ref'] = np.mean(relative_indices[relative_indices != 0])
                result['std-ref'] = np.std(relative_indices[relative_indices != 0])
                result['quantile-25'] = np.quantile(relative_indices[relative_indices != 0], 0.25)
                result['quantile-50'] = np.quantile(relative_indices[relative_indices != 0], 0.5)
                result['quantile-75'] = np.quantile(relative_indices[relative_indices != 0], 0.75)
            os.remove('tmp/test.wav')
            os.remove('tmp/decoded.wav')
            return result

        def process_sum_repet(fg_file, bg_file, ref_wave_file, df: pd.DataFrame) -> pd.DataFrame:

            data, sr = librosa.load(ref_wave_file)
            sf.write('tmp/raw.wav', data, sr)
            ffmpeg.input('tmp/raw.wav', ss=50, t=10).output('tmp/ref.wav', ac=1, ar='48K').run(quiet=True,
                                                                                               overwrite_output=True)

            data, sr = librosa.load(fg_file)
            Encoder.encode_mdct(data, sr, 'tmp/encoded_fg.npz', zero_threshold=0.0001)
            loaded = np.load('tmp/encoded_fg.npz', allow_pickle=True)
            decoded_data_fg = Encoder.decode_mdct(loaded['data'])

            for i in tqdm([0.1, 1, 2, 5], desc='I-Rates', leave=False):
                data, sr = librosa.load(bg_file)
                Encoder.encode_mdct_diff(data, sr, 'tmp/encoded_bg.npz', zero_threshold_i_frame=0,
                                         zero_threshold_p_frame=0.0001, i_rate=i)
                loaded = np.load('tmp/encoded_bg.npz', allow_pickle=True)
                decoded_data_bg = Encoder.decode_mdct_diff(loaded)

                result = process_DVs(decoded_data_fg + decoded_data_bg, sr, 'sum', 'with-pframe', loaded['relative'])
                result['i-rate'] = i
                df = df.append(result, ignore_index=True)
                df.to_csv('output.csv')

            os.remove('tmp/encoded_fg.npz')
            os.remove('tmp/encoded_bg.npz')
            os.remove('tmp/ref.wav')
            return df

        def process_IV_IP_frames(ref_wave_file, input_type: str, df: pd.DataFrame) -> pd.DataFrame:

            data, sr = librosa.load(ref_wave_file)
            sf.write('tmp/raw.wav', data, sr)
            ffmpeg.input('tmp/raw.wav', ss=50, t=10).output('tmp/ref.wav', ac=1, ar='48K').run(quiet=True,
                                                                                               overwrite_output=True)

            Encoder.encode_mdct(data, sr, 'tmp/encoded.npz', lossy=False)
            loaded = np.load('tmp/encoded.npz', allow_pickle=True)
            decoded_data = Encoder.decode_mdct(loaded['data'])
            result = process_DVs(decoded_data, sr, input_type, 'lossless')
            df = df.append(result, ignore_index=True)
            df.to_csv('output.csv')

            data, sr = librosa.load(ref_wave_file)
            Encoder.encode_mdct(data, sr, 'tmp/encoded.npz', zero_threshold=0.0001)
            loaded = np.load('tmp/encoded.npz', allow_pickle=True)
            decoded_data = Encoder.decode_mdct(loaded['data'])
            result = process_DVs(decoded_data, sr, input_type, 'all-iframe')
            df = df.append(result, ignore_index=True)
            df.to_csv('output.csv')

            for i in tqdm([0.1, 1, 2, 5], desc='I-Rates', leave=False):
                Encoder.encode_mdct_diff(data, sr, 'tmp/encoded.npz', zero_threshold_i_frame=0,
                                         zero_threshold_p_frame=0.0001, i_rate=i)
                loaded = np.load('tmp/encoded.npz', allow_pickle=True)
                decoded_data = Encoder.decode_mdct_diff(loaded)
                result = process_DVs(decoded_data, sr, input_type, 'with-pframe', loaded['relative'])
                result['i-rate'] = i
                df = df.append(result, ignore_index=True)
                df.to_csv('output.csv')

            os.remove('tmp/encoded.npz')
            os.remove('tmp/ref.wav')
            os.remove('tmp/raw.wav')
            return df

        with tqdm(total=7, desc='One MusDB Item', leave=False) as t:
            df = process_IV_IP_frames(f'tmp/test/{musdb_item}/mixture.wav', 'mixture', df)
            t.update()
            df = process_IV_IP_frames(f'tmp/test/{musdb_item}/vocals.wav', 'vocals', df)
            t.update()
            df = process_IV_IP_frames(f'tmp/test/{musdb_item}/drums.wav', 'drums', df)
            t.update()
            mix = nussl.AudioSignal(f'tmp/test/{musdb_item}/mixture.wav')
            repet = nussl.separation.primitive.Repet(mix)
            repet_bg, repet_fg = repet()
            t.update()
            repet_fg.to_mono().write_audio_to_file('tmp/raw_fg.wav')
            df = process_IV_IP_frames('tmp/raw_fg.wav', 'repet-fg', df)
            t.update()
            repet_bg.to_mono().write_audio_to_file('tmp/raw_bg.wav')
            df = process_IV_IP_frames('tmp/raw_bg.wav', 'repet-bg', df)
            t.update()
            df = process_sum_repet('tmp/raw_fg.wav', 'tmp/raw_bg.wav', f'tmp/test/{musdb_item}/mixture.wav', df)
            t.update()
            os.remove('tmp/raw_bg.wav')
            os.remove('tmp/raw_fg.wav')
            return df

    # db = nussl.datasets.MUSDB18('tmp', download=True)
    eng = matlab.engine.start_matlab()
    eng.addpath("PQevalAudioMATLAB/PQevalAudio", "PQevalAudioMATLAB/PQevalAudio/CB",
                "PQevalAudioMATLAB/PQevalAudio/Misc", "PQevalAudioMATLAB/PQevalAudio/MOV",
                "PQevalAudioMATLAB/PQevalAudio/Patt")
    df = pd.DataFrame()

    db = os.listdir('tmp/test')
    for music in tqdm(db, desc='MUSDB'):
    #for i in tqdm([27], desc='MusDB'):
        #music = db[i]
        df = process_IV_Input(music, df)


def run_musdb18_lossless():
    # db = nussl.datasets.MUSDB18('tmp', download=True)
    db = os.listdir('tmp/musdb18hq')


    with open('tmp/output_lossless.csv', 'w') as f:
        f.write('name,flac,mdct,mdct_diff,\n')

    for i in trange(50, desc='MUSDB'):
        item = db[i]
        with open('tmp/output_lossless.csv', 'a') as f:
            with tqdm(desc='One Music', total=5, leave=False) as t:

                y, sr = librosa.load(f'tmp/musdb18hq/{item}/mixture.wav')
                t.update()

                y = librosa.to_mono(y)
                sf.write('tmp/raw.wav', y, sr)
                t.update()

                f.write(item + ',')

                os.makedirs(f'tmp/output/{item}/lossless', exist_ok=True)
                process_lossless('tmp/raw.wav', f'tmp/output/{item}/lossless', t, f)
                os.remove('tmp/raw.wav')


def decode_musdb18():
    db = os.listdir('tmp/output')

    for i in trange(50, desc='Decoding'):
        item = db[i]
        with tqdm(desc='One Music', total=8, leave=False) as t:

            encode_ffmpeg(f'tmp/output/{item}/mix/raw.mp3', f'tmp/output/{item}/mix/decoded_mp3.wav')
            t.update()

            encode_ffmpeg(f'tmp/output/{item}/mix/raw.aac', f'tmp/output/{item}/mix/decoded_aac.wav')
            t.update()

            data, sr, _, _ = Encoder.read_compressed(f'tmp/output/{item}/mix/raw_mdct.npz')
            t.update()
            data = Encoder.decode_mdct(data)
            t.update()
            sf.write(f'tmp/output/{item}/mix/decoded_mdct.wav', data, sr)
            t.update()

            data, sr, r, signs = Encoder.read_compressed(f'tmp/output/{item}/mix/raw_mdct_diff.npz')
            t.update()
            data = Encoder.decode_mdct_diff(data, r, signs)
            t.update()
            sf.write(f'tmp/output/{item}/mix/decoded_mdct_diff.wav', data, sr)
            t.update()


def run_sample():
    with open('tmp/output.csv', 'a') as f:
        with tqdm(desc='One Music', total=15, leave=False) as t:
            process1('samples/raw.wav', t, f)

    # time = datetime.datetime.now()
    # data, sr = librosa.load('/home/matin/Git/UVic/DrumEncoder/samples/raw_fg_repet.wav')
    # data = Encoder.encode_mdct(data)
    # Encoder.write_compressed(data, sr, 'tmp/raw_mdct.npz')
    # print(datetime.datetime.now() - time)
    #
    # time = datetime.datetime.now()
    # data, sr = librosa.load('/home/matin/Git/UVic/DrumEncoder/samples/raw_fg_repet.wav')
    # Encoder.encode_mdct_diff(data, sr, 'tmp/raw_mdct_diff.npz')
    # print(datetime.datetime.now() - time)

    # time = datetime.datetime.now()
    # data, sr = Encoder.read_compressed('tmp/raw_mdct.npz')
    # data = Encoder.decode_mdct(data)
    # sf.write('tmp/raw_revert.wav', data, sr)
    # print(datetime.datetime.now() - time)

    # time = datetime.datetime.now()
    # data, sr, r = Encoder.read_compressed('tmp/raw_mdct_diff.npz')
    # data = Encoder.decode_mdct_diff(data, r)
    # sf.write('tmp/raw_revert_diff.wav', data, sr)
    # print(datetime.datetime.now() - time)

    # mix = nussl.AudioSignal(file + ".wav")
    # repet = nussl.separation.primitive.Repet(mix)
    # repet_bg, repet_fg = repet()

def run_sample2():

    file = 'tmp/test/The Easton Ellises - Falcon 69/mixture.wav'
    ffmpeg.input(file, ss=60, t=10).output('tmp/ref.wav', ac=1, ar='48K').run(quiet=True, overwrite_output=True)

    print('Open Matlab...')
    eng = matlab.engine.start_matlab()
    eng.addpath("PQevalAudioMATLAB/PQevalAudio", "PQevalAudioMATLAB/PQevalAudio/CB",
                "PQevalAudioMATLAB/PQevalAudio/Misc", "PQevalAudioMATLAB/PQevalAudio/MOV",
                "PQevalAudioMATLAB/PQevalAudio/Patt")

    for f in ['mixture']:#, 'vocals', 'drums']:

        # file = f'tmp/test/Signe Jakobsen - What Have You Done To Me/{f}.wav'
        file = 'samples/billie_jean.wav'
        result = pd.DataFrame()
        print('Open Wave file...')
        wave_data, sr = librosa.load(file)
        print('Convert Wave to mono...')
        wave_data = librosa.to_mono(wave_data)
        print('Create Ref Wave...')
        sf.write('tmp/ref.wav', wave_data, sr)
        ffmpeg.input('tmp/ref.wav', ss=60, t=10).output('tmp/ref_short.wav', ac=1, ar='48K').run(quiet=True, overwrite_output=True)

        for i in tqdm([0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03], desc='I column'):
        # for i in tqdm([0, 0.001, 0.1, 0.5, 1, 1.5, 2, 2.5, 3], desc='I column'):

            Encoder.encode_mdct(wave_data, sr, 'tmp/mdct.npz', zero_threshold=i)
            result.at['size', i] = os.path.getsize('tmp/mdct.npz') / os.path.getsize('tmp/ref.wav')/2

            loaded = np.load('tmp/mdct.npz', allow_pickle=True)
            decoded_data = Encoder.decode_mdct(loaded['data'])
            sf.write(f'tmp/mdct_{i}.wav', decoded_data, loaded['sample_rate'])
            ffmpeg.input(f'tmp/mdct_{i}.wav', ss=60, t=10).output('tmp/test.wav', ac=1, ar='48K').run(quiet=True, overwrite_output=True)
            result.at['peaq', i] = eng.PQevalAudio('tmp/ref_short.wav', 'tmp/test.wav', stdout=io.StringIO())
            result.sort_index(inplace=True)
            result.to_csv(f'tmp/output_{f}.csv')

            for p in tqdm([0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03], desc='P column', leave=False):
            # for p in tqdm([0, 0.001, 0.1, 0.5, 1, 1.5, 2, 2.5, 3], desc='P column', leave=False):

                encoded_mdct_diff_data, relative_indices, _ = Encoder.encode_mdct_diff(wave_data, sr,
                                                                                              'tmp/mdct_diff.npz',
                                                                                              zero_threshold_i_frame=i,
                                                                                              zero_threshold_p_frame=p,
                                                                                              i_rate=2)
                result.at[f'pi_{p}', i] = np.count_nonzero(relative_indices) / len(relative_indices)
                result.at[f'size_{p}', i] = os.path.getsize('tmp/mdct_diff.npz') / os.path.getsize('tmp/ref.wav')/2

                loaded = np.load('tmp/mdct_diff.npz', allow_pickle=True)
                decoded_data = Encoder.decode_mdct_diff(loaded)
                sf.write(f'tmp/mdct_{i}_diff_{p}.wav', decoded_data, loaded['sample_rate'])
                ffmpeg.input(f'tmp/mdct_{i}_diff_{p}.wav', ss=60, t=10).\
                    output('tmp/test.wav', ac=1, ar='48K').run(quiet=True, overwrite_output=True)

                result.at[f'peaq_{p}', i] = eng.PQevalAudio('tmp/ref_short.wav', 'tmp/test.wav', stdout=io.StringIO())
                result.sort_index(inplace=True)
                result.to_csv(f'tmp/output_{f}.csv')


if __name__ == '__main__':
    # decode_musdb18()
    run_musdb18()
    # run_sample2()
