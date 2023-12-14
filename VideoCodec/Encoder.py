import pickle
import os
import tensorflow as tf
import imageio
import numpy as np
from argparse import ArgumentParser

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def iterative_greedy_pixel_selection(original_image, num_samples):
    selected_pixels = []
    residual_image = original_image.copy()

    num_pixels = np.prod(original_image.shape)
    num_samples = min(num_samples, num_pixels)  # Ensure num_samples is within a reasonable range

    for i in range(num_samples):
        if np.sum(np.abs(residual_image)) == 0:
            break  # If all pixels have been set to 0, exit the loop

        # Use np.argmax directly on the 2D array
        idx = np.argmax(np.abs(residual_image - np.mean(residual_image)))

        # Wrap the integer idx in a tuple for unravel_index
        unravel_index = np.unravel_index(idx, residual_image.shape)
        row_idx, col_idx = unravel_index[0], unravel_index[1]

        selected_pixels.append(((row_idx, col_idx), residual_image[row_idx, col_idx]))

        residual_image[row_idx, col_idx] = 0  # Set sampled pixel to 0

    return selected_pixels

def generate_dictionary():
    num_atoms = 1024
    atom_size = 32*32*3
    dictionary = np.random.randn(num_atoms, atom_size).astype(np.float32)
    np.save('dictionary.npy', dictionary)


def encoder(loadmodel, input_path, refer_path, outputfolder, num_samples=0.1):
    graph = load_graph(loadmodel)
    prefix = 'import/build_towers/tower_0/train_net_inference_one_pass/train_net/'

    Res = graph.get_tensor_by_name(prefix + 'Residual_Feature:0')
    inputImage = graph.get_tensor_by_name('import/input_image:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')
    Res_prior = graph.get_tensor_by_name(prefix + 'Residual_Prior_Feature:0')
    motion = graph.get_tensor_by_name(prefix + 'Motion_Feature:0')
    bpp = graph.get_tensor_by_name(prefix + 'rate/Estimated_Bpp:0')
    psnr = graph.get_tensor_by_name(prefix + 'distortion/PSNR:0')
    reconframe = graph.get_tensor_by_name(prefix + 'ReconFrame:0')

    with tf.compat.v1.Session(graph=graph) as sess:
        im1 = imageio.imread(input_path)
        im2 = imageio.imread(refer_path)
        im1 = im1 / 255.0
        im2 = im2 / 255.0
        im1 = np.expand_dims(im1, axis=0)
        im2 = np.expand_dims(im2, axis=0)

        selected_pixels = iterative_greedy_pixel_selection(im1.squeeze(), int(num_samples * np.prod(im1.shape[1:])))

        bpp_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = sess.run(
            [bpp, Res, Res_prior, motion, psnr, reconframe], feed_dict={
                inputImage: im1,
                previousImage: im2
            })

    print(bpp_est)
    print(psnr_val)
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    output_res = open(outputfolder + 'quantized_res_feature.pkl', 'wb')
    pickle.dump((selected_pixels, Res_q), output_res)
    output_res.close()

    output_res_prior = open(outputfolder + 'quantized_res_prior_feature.pkl', 'wb')
    pickle.dump(Res_prior_q, output_res_prior)
    output_res_prior.close()

    output_motion = open(outputfolder + 'quantized_motion_feature.pkl', 'wb')
    pickle.dump(motion_q, output_motion)
    output_motion.close()

if __name__ == "__main__":

    # Generate dictionary 
    generate_dictionary()

    parser = ArgumentParser()   
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", 
                        default='./model/L2048/frozen_model_E.pb')
                        
    parser.add_argument('--input_frame', type=str, dest="input_path",
                        default='./image/172.png')
                        
    parser.add_argument('--refer_frame', type=str, dest="refer_path", 
                        default='./image/170.png')

    parser.add_argument('--outputpath', type=str, dest="outputfolder",
                        default='./testpkl/')  

    parser.add_argument('--num_samples', type=float, dest="num_samples", 
                        default=0.1)

    args = parser.parse_args()
    
    encoder(**vars(args))