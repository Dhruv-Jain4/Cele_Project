import random
import os
from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.io import wavfile
from typing import Union
import tensorflow as tf
import numpy as np
import shutil
import datetime
import pyaudio
import time
import json
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Activation
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from scipy.signal import spectrogram
import wave
  
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
def convert_mp3_to_wav(input_path:str, output_path:str) -> None:
  """
  Given path of a mp3 file converts it to wav file and saves it in ginve output path
  """
  sound = AudioSegment.from_mp3(input_path)
  sound.export(output_path, format="wav")

def bulk_convert_mp3_to_wav(input_path:str, output_path:str) -> None:
  """
  Given path of a folder containing mp3 files converts it to wav file and saves it in ginve output folder.
  Automatically creates output folder if not exists
  Skips file that are not mp3 format
  Skips file that cannot be read using pydub
  """
  assert os.path.exists(input_path), "given input path does not exist"
  if not os.path.exists(output_path):
    # print("Given output directory does not exist. Creating new ......")
    os.makedirs(output_path)
  skipped = 0
  for i in os.listdir(input_path):
    if os.path.isdir(os.path.join(input_path, i)):
      bulk_convert_mp3_to_wav(os.path.join(input_path, i), os.path.join(output_path, i))
      continue
    file_name = i[:i.rfind(".")]
    ext = i[i.rfind(".") :]
    if ext == ".wav":
      shutil.copy(os.path.join(input_path, i), os.path.join(output_path, i))
      continue
    if ext != ".mp3":
    #   print(f"File with name {i} is not an mp3 file. Skipping it.")
      skipped += 1
      continue
    
    try:
      convert_mp3_to_wav(os.path.join(input_path, i), os.path.join(output_path, file_name + ".wav"))
    except Exception as exc:
    #   print(f"Unable to convert file with name {i}. Skipping it {exc}")
      skipped += 1
      continue
#   print(f"total files skipped : {skipped} out of total {len(os.listdir(input_path))} files")
# output_path = "/content/MyDrive/MyDrive/activates_helloworld_wav"
# input_path = "/content/MyDrive/MyDrive/activates_helloworld_mp3"
# bulk_convert_mp3_to_wav(input_path, output_path)

def quieten(file : Union[str, np.ndarray], rate : int) -> np.ndarray:
    """
    Given a path or array of wave file quietens it by given rate. It just divides it with given number
    """
    sr = None
    if isinstance(file, str):
        file = read(file)
        sr = file[0]
        file = file[1]
    return np.int16(file // rate), sr

def louden(file : Union[str, np.ndarray], rate : int) -> np.ndarray:
    """
    Given a path or array of wav file loudens it by given rate. It just multiples it
    """
    sr = None
    if isinstance(file, str):
        file = read(file)
        sr = file[0]
        file = file[1]
    return np.int16(file * rate), sr


def get_seconds(file : Union[str, np.ndarray],sr = None) -> float:
    """
    Given a path or array of wav file return it seconds
    """
    if isinstance(file, str):
        file = read(file)
        sr = file[0]
        file = file[1]
    else:
        if sr is None:
            raise ValueError("please provide sr(sampling rate) if given is not the path of the file")
    return file.shape[0] / sr

def trim_audio(audio_data : np.ndarray, sr : int, seconds : int, save_path :str) -> None:
  """
  TODO add methods to trim
  1. start
  2. end
  3. mid
  Given nd array of audio data trims it to given seconds and saves it in given path.
  """

  num_samples_to_keep = int(seconds * sr)

  trimmed_audio = audio_data[:num_samples_to_keep]

  wavfile.write(save_path, sr, trimmed_audio)


def pad_audio(audio_data, sr, seconds, save_path, pad_method) -> None:
  """
  Given nd array of audio data pads it to given seconds and saves it in given path.
  has 2 pad methods
  1. repeat : repeat the audio required times to pad it ot given seconds.
  2. silence : add silent audio at end to make the audio of given length
  """
  if len(audio_data.shape) > 1:
    n_channels = 2
  else:
    n_channels = 1
  current_duration_seconds = len(audio_data) / sr
  required_samples = sr * seconds
  num_samples_to_pad = required_samples - len(audio_data)

  if pad_method == "silence":
    if n_channels == 1:
      padding = np.zeros(num_samples_to_pad, dtype=audio_data.dtype)
    else:
      padding = np.zeros((num_samples_to_pad, 2), dtype=audio_data.dtype)
    padded_audio = np.concatenate((audio_data, padding))

  else:

    # Calculate the number of times to repeat the audio to reach or slightly exceed the target duration
    num_repeats = required_samples // len(audio_data)
    audio_len = len(audio_data)
    # Create a NumPy array for repeating the audio
    if n_channels == 1:
      repeated_audio = np.zeros(len(audio_data) * num_repeats,dtype = audio_data.dtype)
    else:
      repeated_audio = np.zeros((len(audio_data) * num_repeats, 2),dtype =  audio_data.dtype)
    for i in range(num_repeats):
      repeated_audio[i*audio_len : (i+1)*audio_len] = audio_data
    # Calculate the number of samples to add for the remaining duration
    num_samples_to_add = required_samples - len(repeated_audio)
    remaining_audio = audio_data[:num_samples_to_add]

    # Concatenate the repeated and remaining audio
    padded_audio = np.concatenate((repeated_audio, remaining_audio))
  wavfile.write(save_path, sr, padded_audio)

def trim_or_pad_wav_file(audio_path : str, seconds : int, save_path :str, pad_method : str) -> None:
  """
  Given path to audio (.wav format) file trims or pads it to given seconds and save it in given path.
  """
  if pad_method not in ["repeat", "silence"]:
    raise ValueError(f"attribute pad_method can either be one of ['repeat', 'silence'] not {pad_method}")
  try:
    sr, audio_data = wavfile.read(audio_path)
  except:
    raise RuntimeError(f"Unable to read given file {audio_path}")
  file_seconds = get_seconds(audio_data, sr)
  if file_seconds < seconds:
    pad_audio(audio_data, sr, seconds, save_path, pad_method)
  else:
    trim_audio(audio_data, sr, seconds, save_path)

def bulk_trim_or_pad_wav_file(input_path:str, seconds:str, output_path :str,pad_method : str, suffix = None) -> None:
  """
  Given path of a folder containing mp3 files converts it to wav file and saves it in ginve output folder.
  Automatically creates output folder if not exists
  Skips file that are not mp3 format
  Skips file that cannot be read using pydub
  """
  assert os.path.exists(input_path), "given input path does not exist"
  if not os.path.exists(output_path):
    # print("Given output directory does not exist. Creating new ......")
    os.makedirs(output_path)
  skipped = 0
  if suffix == None:
    suffix = f"_{seconds}s_{pad_method}"
  for i in os.listdir(input_path):
    if os.path.isdir(os.path.join(input_path, i)):
      bulk_trim_or_pad_wav_file(os.path.join(input_path, i), seconds, os.path.join(output_path, i), pad_method, suffix)
      continue
    file_name = i[:i.rfind(".")]
    ext = i[i.rfind(".") :]
    if ext != ".wav":
      print(f"File with name {i} is not an mp3 file. Skipping it.")
      skipped += 1
      continue
    try:
      trim_or_pad_wav_file(os.path.join(input_path, i), seconds, os.path.join(output_path, file_name +suffix+ ".wav"),pad_method)
    except Exception as exc:
      print(f"Unable to convert file with name {i}. Skipping it")
      skipped += 1
      continue
  print(f"total files skipped : {skipped} out of total {len(os.listdir(input_path))} files")

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    ### END CODE HERE ###

    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
  
    ### START CODE HERE ###
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    count = 0
    while is_overlapping(segment_time, previous_segments):
        # print(count)
        if count > 20:
          # print("Unable to find non overlapping skipping this.")
          return background, None
        segment_time = get_random_time_segment(segment_ms)
        count += 1


    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    ### END CODE HERE ###

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])

    return new_background, segment_time

# GRADED FUNCTION: create_training_example
def graph_spectrogram(wav_file):
    flag = True
    retry = 0
    while flag:
        if retry > 5:
            break
        try:
            rate, data = get_wav_info(wav_file)
            flag = False
            break
        except:
            # print(wav_file)
            time.sleep(0.1)
            retry += 1
            print(f"Retrying ({retry})")

    # print(data.shape, wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        _,_,x = spectrogram(data, fs = fs,nfft = nfft, noverlap = noverlap, nperseg=nfft)
    elif nchannels == 2:
        _,_,x = spectrogram(data[:,0], fs = fs,nfft = nfft, noverlap = noverlap, nperseg=nfft)
    return x


def assert_existance(path):
    """
    Raises error if given path does not exists
    """
    assert os.path.exists(path), f"Path {path} does not exist "

def walk(dir, ext):
    """
    Return path of files with given ext from a directory and its subdirectories and so on.
    """
    all_files = [os.path.join(path, file)
                for path, subdir, files in os.walk(dir) 
                for file in files if file.endswith(".wav")]
    return all_files


def change_sampling_rate_wav_file(audio: str, target_sr : int,save_path):
    """
    Change sampling rate of given audio file.
    """
    audio = AudioSegment.from_wav(audio)
    audio
    # print(audio.frame_rate)

    audio = audio.set_frame_rate(target_sr)
    # print(audio.frame_rate)

    audio.export(save_path, format="wav")

def bulk_change_sampling_rate_wav_file(input_dir:str, target_sr:int, output_dir:str, suffix :str = None):
    assert os.path.exists(input_dir), "given input path does not exist"
    if not os.path.exists(output_dir):
        print("Given output directory does not exist. Creating new ......")
        os.makedirs(output_dir)
    skipped = 0
    if suffix is None:
        suffix = f"_{target_sr}sr"
    for i in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, i)):
            bulk_change_sampling_rate_wav_file(os.path.join(input_dir, i), target_sr, os.path.join(output_dir, i), suffix=suffix)
            continue
        file_name = i[:i.rfind(".")]
        ext = i[i.rfind(".") :]
        if ext != ".wav":
            print(f"File with name {i} is not an wav file. Skipping it.")
            skipped += 1
            continue
        try:
            change_sampling_rate_wav_file(os.path.join(input_dir, i), target_sr, os.path.join(output_dir, file_name +suffix+".wav"))
        except Exception as exc:
            print(f"Unable to convert file with name {i}. Skipping it")
            skipped += 1
            continue
    print(f"total files skipped : {skipped} out of total {len(os.listdir(input_dir))} files")

def preprocess_dir(dir : str, processed_dir_name : str, conversion : bool, sr_correct :bool, trim_pad : bool, seconds : int, pad_method : str,add_suffix : bool, sr :int = 44100):
    """
    Given a directory that contains audio files (mp3 and wav) does all required processing on them and files in its subfolders.
    Make sure that you have disk space in the drive as it might take more space the it already occupied.
    convertion : convert mp3 to wav else skip all formats but wav. if given conversion = True only converts mp3 files and save in processed_dir_name. Just copy pastes files that are already in wav format.
    sr_correct: set sr of each file to given sr. (Only to wav files)
    trim_pad : trim_or pad audio files in the given directory and subdirectories.
    processed_dir_name : the new directory will be the same as the previous one with required changes. File names may change if add_suffix was set to true

    add_suffix : If given true all wav files will have suffixes representing some changes done in them.
    """
    new_dir = dir
    suffix = None if add_suffix else ""
    assert os.path.exists(dir), f"givem input path {dir} does not exists."
    if not os.path.exists(processed_dir_name):
        print("Output directory does not exist creating new ...")
        os.makedirs(processed_dir_name, exist_ok=True)
    if conversion:
        bulk_convert_mp3_to_wav(dir,"temp1")
        new_dir = "temp1"
    if sr_correct:
        
        bulk_change_sampling_rate_wav_file(new_dir, sr, "temp2",suffix=suffix)
        if new_dir != dir:
            shutil.rmtree(new_dir)
        new_dir = "temp2"
    if trim_pad:

        bulk_trim_or_pad_wav_file(new_dir, seconds, processed_dir_name,pad_method, suffix=suffix)
        if new_dir!= dir:
            shutil.rmtree(new_dir)
        new_dir = processed_dir_name

    if new_dir != processed_dir_name:
        shutil.copytree(new_dir, processed_dir_name, dirs_exist_ok=True)
        shutil.rmtree(new_dir)
    print(f"Processing done. File saved in {processed_dir_name}")

def bulk_change_loudness_wav_file(dir_path :str, min_mean :int, max_mean : int, save_dir :str):
    """
    Given a folder changes mean of all its wav files and wav files in subfolers between min_mean and max_mean. It adjusts the loudness of the files. 
    Saves the results in save_dir (creates one if does not exists)
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # List all WAV files in the directory
    # wav_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
    skipped = 0
    # Process each WAV file
    for wav_file in os.listdir(dir_path):
        input_path = os.path.join(dir_path, wav_file)
        output_path = os.path.join(save_dir, wav_file)
        if os.path.isdir(input_path):
            bulk_change_loudness_wav_file(input_path, min_mean, max_mean, output_path)
            continue
        if wav_file.endswith(".wav"):
            # Modify loudness and save the new audio file
            modified_audio = change_loudness_wav_file(input_path, min_mean, max_mean)
            wavfile.write(output_path, modified_audio[0],modified_audio[1])
        else:
            skipped += 1
    print(f"Files skipped {skipped} out of {len(os.listdir(dir_path))}")
def change_loudness_wav_file(audio_file : str, min_mean :int, max_mean :int):
    """
    Given path of wav file makes its mean between min_mean and max_mean. changes is loudness. does not save the file but returns the scaled array.
    """
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_mean = np.mean(np.abs(audio_data))  # Calculate absolute mean

    min_scale = audio_mean/min_mean
    max_scale = audio_mean/max_mean
    random_scale = random.uniform(min_scale, max_scale)
    new_audio_array = np.int16(audio_data/random_scale)
    return  sample_rate, new_audio_array

def save_file(name, frames, channels, format, rate):
    # print("came here 1")
    p = pyaudio.PyAudio()
    wf = wave.open(name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
def preprocess_audio(filename, save_name):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    # print(segment.frame_count())
    def pad_or_trim(audio, threshold):
      if len(audio) < threshold:
          silence = AudioSegment.silent(threshold - len(audio))
          return audio + silence
      else:
          return audio[:threshold]
    segment = pad_or_trim(segment, 10001 ) 
    segment.export(save_name, format='wav')
  
# def detect_triggerword(filename, normalize, show, model):
#     # plt.subplot(2, 1, 1)

#     x = graph_spectrogram(filename, show = show)
#     print(x.shape)
#     if normalize:
#       print('Normalizing')
#       x = librosa.pcen(x)
#     # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
#     x  = x.swapaxes(0,1)
#     x = np.expand_dims(x, axis=0)
#     predictions = model.predict(x, verbose = 0)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(predictions[0,:,0])
#     plt.ylabel('probability')
#     if show:
#       plt.show()
#     return predictions
# def chime_on_activate(filename, predictions, threshold):
#     audio_clip = AudioSegment.from_wav(filename)
#     chime = AudioSegment.from_wav(chime_file)
#     Ty = predictions.shape[1]
#     # Step 1: Initialize the number of consecutive output steps to 0
#     consecutive_timesteps = 0
#     # Step 2: Loop over the output steps in the y
#     for i in range(Ty):
#         # Step 3: Increment consecutive output steps
#         consecutive_timesteps += 1
#         # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
#         if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
#             # Step 5: Superpose audio and background using pydub
#             audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
#             # Step 6: Reset consecutive output steps to 0
#             consecutive_timesteps = 0        
#     audio_clip.export("chime_output.wav", format='wav')

def manage_overflow(frames, threshold, fps):
    if len(frames) >= threshold:
        frames = frames[fps:]
    return frames

def preprocess_dir(dir : str, processed_dir_name : str, conversion : bool, sr_correct :bool, trim_pad : bool, seconds : int, pad_method : str,add_suffix : bool, change_loudness :bool, min_mean:int, max_mean:int, sr :int = 44100):
    """
    Given a directory that contains audio files (mp3 and wav) does all required processing on them and files in its subfolders.
    Make sure that you have disk space in the drive as it might take more space the it already occupied.
    convertion : convert mp3 to wav else skip all formats but wav. if given conversion = True only converts mp3 files and save in processed_dir_name. Just copy pastes files that are already in wav format.
    sr_correct: set sr of each file to given sr. (Only to wav files)
    trim_pad : trim_or pad audio files in the given directory and subdirectories.
    processed_dir_name : the new directory will be the same as the previous one with required changes. File names may change if add_suffix was set to true

    add_suffix : If given true all wav files will have suffixes representing some changes done in them.
    """
    new_dir = dir
    suffix = None if add_suffix else ""
    assert os.path.exists(dir), f"givem input path {dir} does not exists."
    if not os.path.exists(processed_dir_name):
        print("Output directory does not exist creating new ...")
        os.makedirs(processed_dir_name, exist_ok=True)
    if conversion:
        bulk_convert_mp3_to_wav(dir,"temp1")
        new_dir = "temp1"
    if sr_correct:
        
        bulk_change_sampling_rate_wav_file(new_dir, sr, "temp2",suffix=suffix)
        if new_dir != dir:
            shutil.rmtree(new_dir)
        new_dir = "temp2"
    if trim_pad:

        bulk_trim_or_pad_wav_file(new_dir, seconds, "temp1",pad_method, suffix=suffix)
        if new_dir!= dir:
            shutil.rmtree(new_dir)
        new_dir = "temp1"
    if change_loudness:

        bulk_change_loudness_wav_file(new_dir, min_mean, max_mean, processed_dir_name)
        if new_dir!= dir:
            shutil.rmtree(new_dir)
        new_dir = processed_dir_name

    if new_dir != processed_dir_name:
        shutil.copytree(new_dir, processed_dir_name, dirs_exist_ok=True)
        shutil.rmtree(new_dir)
    print(f"Processing done. File saved in {processed_dir_name}")


class DirectoryDataset:
    def __init__(self, base_dir : str, read_mode : str, sr : int) -> None:
        assert read_mode in ["pydub", "scipy"], "Read mode must be either pydub or scipy"
        self.read_mode = read_mode
        self.base_dir = base_dir
        self.audio_files = walk(base_dir, ".wav")
        self.sr = sr
        print(f"Total fies found in {base_dir} are  {len(self.audio_files)}")


    def __getitem__(self, key : int):
        # logger.info(f"Accessing file {self.audio_files[key]}")
        if self.read_mode == "pydub":
            audio = AudioSegment.from_wav(self.audio_files[key])
            assert audio.frame_rate == self.sr, f"File with name {self.audio_files[key]} does not have a sr of {self.sr}"
            return audio
        else:
            # print("hello2")
            file =  wavfile.read(self.audio_files[key])
            assert file[0] == self.sr,  f"File with name {self.audio_files[key]} does not have a sr of {self.sr}"
            return file

    def __len__(self):
        return  len(self.audio_files)

def chime_on_activate(filename, predictions, threshold, save_path = None, ts = None):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav("chime.wav")
    if ts == None:
        ts = ""
    else:
        ts = "_" + ts
    if save_path == None:
        save_path = ""
    Ty = predictions.shape[0]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # print(consecutive_timesteps)
        # Step 3: Increment consecutive output steps
        if predictions[i, 0] > threshold:
            consecutive_timesteps += 1
        else:
            consecutive_timesteps = 0
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if consecutive_timesteps >= 50:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0        
    audio_clip.export(os.path.join(save_path,f"chime_output{ts}.wav"), format='wav')

class TriggerWordDataset:
    def __init__(self, data_dir:str,background_length: int,sr : int, ty : int, output_shape : tuple, min_activates :int = 1, max_activates :int = 3,
                  min_negatives :int = 1, max_negatives :int= 4, min_mean :int= 100, max_mean :int = 400, demo_data_save_perc :float = 0.1, save_dir :str = "train_data_generated",
                  change_activate_loudeness :bool = True, change_negative_loudness : bool = True, change_background_loudness = True,
                  spec_params :dict = {"nfft" : 200,"fs" : 8000 ,"noverlap" : 120,"nperseg" : 200},backgrounds_dir : str = None, negatives_dir : str = None, activates_dir :str = None) -> None:
        self.base_path = data_dir
        self.sr =sr
        self.background_dir = backgrounds_dir or os.path.join(data_dir, "backgrounds")
        self.negatives_dir = negatives_dir or os.path.join(data_dir, "negatives")
        self.activates_dir = activates_dir or os.path.join(data_dir, "activates_helloworld_wav")
        self.backgrounds_data = DirectoryDataset(self.background_dir, "pydub", sr=self.sr)
        self.negatives_data = DirectoryDataset(self.negatives_dir, "pydub", sr = self.sr)
        self.activates_data = DirectoryDataset(self.activates_dir, "pydub",sr = self.sr)
        self.background_length = background_length
        self.min_activates = min_activates
        self.max_activates = max_activates
        self.min_negatives = min_negatives
        self.max_negatives = max_negatives
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.spec_params = spec_params
        self.demo_data_save_perc = demo_data_save_perc
        self.save_dir = save_dir
        self.change_activate_loudeness = change_activate_loudeness
        self.change_negative_loudness = change_negative_loudness
        self.ty = ty
        self.change_background_loudness = change_background_loudness
        self.tx = self.get_tx()
        self.output_shape = output_shape
        self.input_shape = self.get_input_shape()
        if not os.path.exists(save_dir):
            print("Save path does not exists creating one. ...")
            os.makedirs(save_dir)
        assert_existance(self.background_dir)
        assert_existance(self.activates_dir)
        assert_existance(self.negatives_dir)
    
    def get_tx(self) -> int:
        return spectrogram(np.random.random(self.sr * self.background_length), **self.spec_params)[2].T.shape[1]
    
    def get_input_shape(self) -> tuple:
        return spectrogram(np.random.random(self.sr * self.background_length), **self.spec_params)[2].T.shape

    def create_dataset(self, samples):
        runs = (samples // len(self.background_dir)) + 1
        x = np.zeros((samples, self.input_shape[0], self.input_shape[1]), dtype=np.int16)
        y = np.zeros((samples, self.output_shape[0], self.output_shape[1]), dtype=np.int16)

        cnt = 0
        stop = False
        for i in range(runs):
            indices = random.sample(range(0, len(self.background_dir)), len(self.background_dir))
            for j in indices:
                x_temp, y_temp = self.create_training_example(j)
                x[cnt] = x_temp
                y[cnt] = y_temp
                cnt += 1
                if cnt >= samples:
                    stop = True
                    break
            if stop:
                break
        return x, y
    
    def create_dataset_generator(self, samples):
        runs = (samples // len(self.background_dir)) + 1
        cnt = 0
        stop = False
        for i in range(runs):
            indices = random.sample(range(0, len(self.backgrounds_data)), len(self.backgrounds_data))
            for j in indices:
                # print("\r", end = "")
                # print(f"{cnt}/{samples}", end="")
                x_temp, y_temp = self.create_training_example(j)
                
                cnt += 1
                yield tf.convert_to_tensor(x_temp), tf.convert_to_tensor(y_temp)
                if cnt >= samples:
                    stop = True
                    break
            if stop:
                break

    def get_wav_info(self, wav_file :str) -> tuple:
        rate, data = wavfile.read(wav_file)
        return rate, data

    def get_random_time_segment(self, segment_ms : int) -> tuple:
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """

        segment_start = np.random.randint(low=0, high=(1000*self.background_length)-segment_ms)   # Make sure segment doesn't run past the 10sec background
        segment_end = segment_start + segment_ms - 1

        return (segment_start, segment_end)

    def is_overlapping(self, segment_time : tuple, previous_segments : list) -> bool:
        """
        Checks if the time of a segment overlaps with the times of existing segments.

        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
        
        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """

        segment_start, segment_end = segment_time

        ### START CODE HERE ### (≈ 4 line)
        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False

        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and segment_end >= previous_start:
                overlap = True
        ### END CODE HERE ###

        return overlap

    def insert_audio_clip(self, background : AudioSegment, audio_clip : AudioSegment, previous_segments : list) -> tuple:
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the
        audio segment does not overlap with existing segments.

        Arguments:
        background -- a 10 second background audio recording.
        audio_clip -- the audio clip to be inserted/overlaid.
        previous_segments -- times where audio segments have already been placed

        Returns:
        new_background -- the updated background audio
        """

        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)

        ### START CODE HERE ###
        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
        # the new audio clip. (≈ 1 line)
        segment_time = get_random_time_segment(segment_ms)

        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
        # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
        count = 0
        while is_overlapping(segment_time, previous_segments):
            if count > 20:
            # print("Unable to find non overlapping skipping this.")
                return background, None
            segment_time = get_random_time_segment(segment_ms)
            count += 1


        # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)
        ### END CODE HERE ###

        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])

        return new_background, segment_time

    # GRADED FUNCTION: create_training_example
    def graph_spectrogram(self, wav_file :str) -> np.ndarray:
        flag = True
        retry = 0
        while flag:
            if retry > 5:
                break
            try:
                rate, data = get_wav_info(wav_file)
                assert rate == self.sr, f"There is a wav file whose sr is not {self.sr}"
                flag = False
                break
            except:
                # print(wav_file)
                time.sleep(0.1)
                retry += 1
                print(f"Retrying ({retry})")

        # print(data.shape, wav_file)
        
        nchannels = data.ndim
        if nchannels == 1:
            _,_,x = spectrogram(data, **self.spec_params)
        elif nchannels == 2:
            _,_,x = spectrogram(data[:,0], **self.spec_params)
        return x

    def create_training_example(self, background_idx : int) -> tuple:
        """
        Creates a training example with a given background, activates, and negatives.

        Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"

        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """

        # Set the random seed
        # np.random.seed(18)

        # Make background quieter
        # background = background - 20
        background = self.backgrounds_data[background_idx]
        background.export("real_background.wav", format = "wav")
        if self.change_background_loudness:
            background = self.change_loudness_pydub(background)
        ### START CODE HERE ###
        # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
        y = np.zeros((1, self.ty))

        # Step 2: Initialize segment times as empty list (≈ 1 line)
        previous_segments = []
        ### END CODE HERE ###

        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(self.min_activates, self.max_activates + 1)
        random_indices = np.random.randint(len(self.activates_data), size=number_of_activates)
        random_activates = [self.activates_data[i] for i in random_indices]
        ### START CODE HERE ### (≈ 3 lines)
        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for idx, random_activate in enumerate(random_activates):
            # Insert the audio clip on the background
            if self.change_activate_loudeness:
                random_activate = self.change_loudness_pydub(random_activate)
            background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
            if segment_time == None:
                continue
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = self.insert_ones(y, segment_end_ms=segment_end)
        ### END CODE HERE ###
        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(self.min_negatives, self.max_negatives + 1)
        random_indices = np.random.randint(len(self.negatives_data), size=number_of_negatives)
        random_negatives = [self.negatives_data[i] for i in random_indices]

        ### START CODE HERE ### (≈ 2 lines)
        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background
            if self.change_negative_loudness:
                random_negative = self.change_loudness_pydub(random_negative)
            background, segment_time_neg = insert_audio_clip(background, random_negative, previous_segments)

            if segment_time_neg == None:
                continue



        ### END CODE HERE ###
        # Standardize the volume of the audio clip

        # Export new training example
        # print(len(background))
        background.export("train" + ".wav", format="wav",)
        val = np.random.uniform(0, 1)    # print("File (train.wav) was saved in your directory.")
        if val <= self.demo_data_save_perc:
            cur_time = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
            file_name = "train_{}.wav".format(cur_time)
            
            
            background.export(os.path.join(self.save_dir, file_name), format = "wav")
            # print(np.sum(y))
            chime_on_activate(filename=os.path.join(self.save_dir, file_name),predictions=y.T, threshold=0.5, save_path=self.save_dir, ts=cur_time)
        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        x = graph_spectrogram("train.wav")
        # print(number_of_negatives, number_of_activates)
        
        return x.T, y.T
    def change_loudness(self, audio : np.ndarray, min_mean :float, max_mean : float) ->  np.ndarray:
        """
        Change loudness of the audio.
        If min_mean and max_mean are gives uses those to make sure mean of audio is between them.
        If it is not given then it will randomly quieten upto 3 times and louden upto 3 times.
        """
        mean = np.abs(audio.mean())
        if min_mean:
            low = min_mean / mean
        else:
            low = 0.33
        if max_mean:
            high = max_mean / mean
        else:
            high = 3
            
        rate = np.random.uniform(low, high)
        audio = louden(audio, rate) # this single function can be used to both louden and quieten. It quietens if rate < 1. Therefore not quietining.
        return audio
    
    def change_loudness_pydub(self, audio : AudioSegment) -> AudioSegment:
        """
        Change loudness of the audio object of pydub
        """
        db = np.random.randint(-5, 10)
        return audio.apply_gain(db)

    def insert_ones(self, y : np.ndarray, segment_end_ms : int) -> np.ndarray:
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 followinf labels should be ones.


        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms -- the end time of the segment in ms

        Returns:
        y -- updated labels
        """

        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * self.ty / (self.background_length * 1000))
        # print(segment_end_y)
        # print(segment_end_ms)

        # Add 1 to the correct index in the background label (y)
        ### START CODE HERE ### (≈ 3 lines)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < self.ty:
                y[0, i] = 1
        ### END CODE HERE ###
        return y

def get_model():
    X_input = Input(shape = (5511, 101))
        
    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    # Add a Conv1D with 196 units, kernel size of 15 and stride of 4
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    # Batch normalization
    X = BatchNormalization()(X)
    # ReLu activation
    X = Activation('relu')(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.4)(X)                                  

    # Step 2: First GRU Layer (≈4 lines)
    # GRU (use 128 units and return the sequences)
    X = GRU(128, return_sequences=True)(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.4)(X)
    # Batch normalization.
    X = BatchNormalization()(X)                           

    # Step 3: Second GRU Layer (≈4 lines)
    # GRU (use 128 units and return the sequences)
    X = GRU(128, return_sequences=True)(X)
    # dropout (use 0.8)
    X = Dropout(rate=0.4)(X)       
    # Batch normalization
    X = BatchNormalization()(X) 
    # dropout (use 0.8)
    X = Dropout(rate=0.4)(X)                                 

    # Step 4: Time-distributed dense layer (≈1 line)
    # TimeDistributed  with sigmoid activation 
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    ### END CODE HERE ###

    model = Model(inputs = X_input, outputs = X)

    return model
def train_model(model, dataset_path, valid_dataset_path,  epochs, callbacks, batch_size, metrics, optimizer, loss, save_path, class_weight):
    def verify_input_output_shape(model, path):
        kwargs = json.load(open(os.path.join(path, "meta.json"), "r"))
        assert kwargs['INPUT_SHAPE'] == list(model.input_shape[1:]), f"Input shape of dataset and model is not same {kwargs['INPUT_SHAPE']} != {model.input_shape[1:]}"
        assert kwargs['OUTPUT_SHAPE'] == list(model.output_shape[1:]), f"Output shape of dataset and model is not same {kwargs['OUTPUT_SHAPE']} != {model.output_shape[1:]}"

    assert not os.path.exists(save_path), "Model save path already exists please make new"
    
    
    # verify_input_output_shape(model, dataset_path)
    # verify_input_output_shape(model, valid_dataset_path)

    train_dataset = tf.data.Dataset.load(os.path.join(dataset_path))
    # valid_dataset = tf.data.Dataset.load(os.path.join(valid_dataset_path))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)
    # valid_dataset = valid_dataset.shuffle(buffer_size=batch_size).batch(batch_size)
    os.makedirs(save_path)
    model.compile(optimizer = optimizer, metrics = metrics, loss= loss)
    model.fit(train_dataset, batch_size=batch_size, epochs=epochs,callbacks=callbacks, class_weight = class_weight)
