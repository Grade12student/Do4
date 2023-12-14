import numpy as np
from sklearn.cluster import KMeans
import imageio
import os

# Specify the folder containing all frames
frames_folder = './frames/'

# Get all file paths in the folder
frame_paths = [os.path.join(frames_folder, filename) for filename in os.listdir(frames_folder)]

# Read frames and flatten them
frames = [imageio.imread(path).flatten() for path in frame_paths]
frames = np.array(frames) / 255.0  # Normalize pixel values to [0, 1]

#  Option 1: Increase the number of clusters in K-Means
num_clusters = 3

# Use K-Means clustering to generate the dictionary
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(frames)

# Get the dictionary
dictionary_kmeans = kmeans.cluster_centers_

# Save the dictionary
np.save('dictionary_kmeans.npy', dictionary_kmeans)

# Print the dimensions of the dictionary
print(f"Dictionary dimensions: {dictionary_kmeans.shape}")

# Option 2: Use a random dictionary
num_atoms = 128  # Adjust as needed
dictionary_random = np.random.rand(frames.shape[1], num_atoms)

# Save the dictionary
np.save('dictionary_random.npy', dictionary_random)

# Print the dimensions of the dictionary
print(f"Dictionary dimensions: {dictionary_random.shape}")

'''import numpy as np
from sklearn.cluster import KMeans

# Load your frames (you might need to adjust the paths)
frame_paths = ['./path/to/frame1.png', './path/to/frame2.png', './path/to/frame3.png']

# Read frames and flatten them
frames = [imageio.imread(path).flatten() for path in frame_paths]
frames = np.array(frames) / 255.0  # Normalize pixel values to [0, 1]

# Set the number of clusters (adjust as needed)
num_clusters = 3

# Use K-Means clustering to generate the dictionary
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(frames)

# Get the dictionary
dictionary = kmeans.cluster_centers_

# Save the dictionary
np.save('dictionary.npy', dictionary)
'''