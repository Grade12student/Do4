import numpy as np
import pickle
import os
import cv2
import tensorflow as tf
from scipy.fftpack import idct
from argparse import ArgumentParser
import glob

# Reconstruction using the iterative hard thresholding algorithm
def iterative_reconstruction(encoded_measurements, measurement_matrix, initial_guess, num_iterations):
    current_guess = initial_guess.copy().reshape(-1, 1)
    print("Shapes:", measurement_matrix.shape, current_guess.shape)

    for _ in range(num_iterations):
        decoded_measurements = np.dot(measurement_matrix, current_guess.flatten())
        error = encoded_measurements - decoded_measurements.reshape(-1, 1)
        current_guess += np.dot(measurement_matrix.T, error).flatten()

    return current_guess.reshape(initial_guess.shape)

# Inverse of the DCT transform
def apply_inverse_transform(data, transform_type='dct'):
    if transform_type == 'dct':
        return idct(idct(data, axis=0, norm='ortho'), axis=1, norm='ortho')
    else:
        raise ValueError("Invalid transform type")

def load_pickled_data(file_pattern):
    data = []
    for file_path in sorted(glob.glob(file_pattern)):
        print(f"Loading file: {file_path}")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        with open(file_path, 'rb') as file:
            file_data = pickle.load(file)
            data.append(file_data)
    if not data:
        print("No data loaded.")
    return np.concatenate(data)



# Decoder function
def decoder(encoded_measurements_pattern, measurement_matrix_pattern, output_folder, transform_type='dct', num_iterations=100):
    # Load encoded measurements, measurement matrix, and other necessary data
    encoded_measurements = load_pickled_data(encoded_measurements_pattern)
    measurement_matrix = load_pickled_data(measurement_matrix_pattern)

    # Initial guess for the reconstruction (you might want to improve this based on your knowledge)
    initial_guess = np.zeros(measurement_matrix.shape[1])

    # Reconstruct the image using iterative hard thresholding
    reconstructed_image = iterative_reconstruction(encoded_measurements, measurement_matrix, initial_guess, num_iterations)

    # Apply the inverse transform to obtain the final image
    reconstructed_image = apply_inverse_transform(reconstructed_image, transform_type)

    # Save the reconstructed image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'reconstructed_image.png')
    cv2.imwrite(output_path, (reconstructed_image * 255).astype(np.uint8))

# Process multiple frames
def process_frames(frame_folder, output_base_folder):
    encoded_measurements_pattern = os.path.join(frame_folder, 'frame*quantized_measurements.pkl')
    measurement_matrix_pattern = os.path.join(frame_folder, 'frame*measurement_matrix.pkl')

    # Call the decoder function for all frames
    decoder(encoded_measurements_pattern, measurement_matrix_pattern, output_base_folder)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--frame_folder', type=str, dest="frame_folder", default='./testpkl/F/', help="folder containing frame measurements and matrices")
    parser.add_argument('--output_base_folder', type=str, dest="output_base_folder", default='./decoded_output/', help="output base folder for the reconstructed images")
    parser.add_argument('--transform_type', type=str, dest="transform_type", default='dct', help="sparsity-inducing transform type")
    parser.add_argument('--num_iterations', type=int, dest="num_iterations", default=100, help="number of iterations for reconstruction")

    args = parser.parse_args()

    # Call the function to process multiple frames
    process_frames(args.frame_folder, args.output_base_folder)
