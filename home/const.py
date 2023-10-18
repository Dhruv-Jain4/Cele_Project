from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
SR = 44100
N_SAMPLES = 1000
BACKGROUND_LENGTH = 10
CONTROL_BACKGROUND_LOUNDESS =True # Make sure it is not very loud or quiet when preprocessing
BACKGROUND_LOUDNESS_RANGE = (100, 500) # No use if CONTROL_BACKGROUND_LOUDNESS is False
CONTROL_ACTIVATES_LOUDNESS = True # Make sure it is not very loud or quiet when preprocessing
ACTIVATE_LOUDNESS_RANGE = (300, 1000) # No use if CONTROL_ACTIVATES_LOUDNESS is False
CONTROL_NEGATIVES_LOUDNESS =True # Make sure it is not very loud or quiet when preprocessing
NEGATIVE_LOUDNESS_RANGE = (300, 1000)# No use if CONTROL_NEGATIVES_LOUDNESS is False
ADJUST_BACKGROUND_LOUNDESS =True # Change a little when training
ADJUST_ACTIVATES_LOUDNESS =False # Change a little when training
ADJUST_NEGATIVES_LOUDNESS =False # Change a little when training
SPEC_PARAMS = {"nfft" : 200,"fs" : 8000 ,"noverlap" : 120,"nperseg" : 200}
SAVE_DATA_PERCENTAGE = 0

CLASS_WEIGHT = {0 : 0.2, 1:1.8}
