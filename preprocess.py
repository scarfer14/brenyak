import librosa
import numpy as np
import os

SAMPLE_RATE = 16000
N_MFCC = 13
MAX_LEN = 32  # frames

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=400, hop_length=160
    )
    mfcc = mfcc.T

    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - len(mfcc)), (0, 0)))

    return mfcc

X, y = [], []
label_map = {"help": 0, "yes": 1, "no": 2, "noise": 3}

for label in label_map:
    folder = f"data/{label}"
    for file in os.listdir(folder):
        mfcc = extract_mfcc(os.path.join(folder, file))
        X.append(mfcc)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved MFCC features")
