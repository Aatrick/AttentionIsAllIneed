# source : https://victorzhou.com/blog/keras-cnn-tutorial/

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 16
filter_size = 6
pool_size = 4

# Build the model.
model = Sequential(
    [
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        Conv2D(
            num_filters,
            filter_size,
            strides=2,
            padding="same",
            activation="gelu",
        ),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation="gelu"),
        Dense(10, activation="softmax"),
    ]
)

# Compile the model.
model.compile(
    "adamW",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model.
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=10,
    validation_data=(test_images, to_categorical(test_labels)),
)
model.save_weights("cnn.weights.h5")

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5])  # [7, 2, 1, 0, 4]
