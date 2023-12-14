from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Input shape
input_shape = (448, 832, 3)

# Define autoencoder model
encoder_input = keras.Input(shape=input_shape)

x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
encoder_output = layers.MaxPooling2D(2, padding='same')(x)

x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(encoder_output)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
decoder_output = layers.UpSampling2D(2)(x)

# Output layer with a depth of 3
final_output = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(decoder_output)

autoencoder = keras.Model(encoder_input, final_output)

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')

# Load images
images_folder = './frames'
image_files = os.listdir(images_folder)
image_files.sort()

images_data = []
for filename in image_files:
    img_path = os.path.join(images_folder, filename)
    img = load_img(img_path, target_size=input_shape[:2])
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    images_data.append(img_array)

images_data = np.array(images_data)

# Target data is the same as the input data for an autoencoder
target_data = images_data

# Train model on images
autoencoder.fit(images_data, target_data, epochs=100)

# Get the weights of the last layer
last_layer_index = -1
weights = autoencoder.layers[last_layer_index].get_weights()[0]

# Reshape the weights to the desired shape
measurement_matrix = weights.reshape((-1, np.prod(weights.shape)))

# Save the trained autoencoder model
autoencoder.save('model.h5')
