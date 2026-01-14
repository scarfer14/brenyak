import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X = np.load("X.npy")
y = np.load("y.npy")

X = X[..., np.newaxis]  # add channel

model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2)

model.save("kws_model")
