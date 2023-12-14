import pickle
import tensorflow as tf
import imageio
import numpy as np
from argparse import ArgumentParser
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_array


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def save_image(decoded_image, output_path):
    decoded_image = np.clip(decoded_image.squeeze(), 0, 1) * 255
    decoded_image = decoded_image.astype(np.uint8)
    imageio.imwrite(output_path, decoded_image)

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph



def sparse_coding(selected_pixels, dictionary):
    
    # Extract selected pixels 
    pixel_values = np.array([p[1] for p in selected_pixels])
    
    # Initialize reconstruction as zero array
    recon = np.zeros_like(pixel_values)
    
    # Dictionary matrix
    D = dictionary
    
    # Use L1 regularized least squares (Lasso) 
    # to find sparse code that reconstructs pixel values
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(D, pixel_values)
    code = lasso.coef_
    
    # Reconstruct image using dictionary & sparse code 
    recon = D.dot(code)
    
    return recon

def decoder(loadmodel, refer_path, outputfolder):
    graph = load_graph(loadmodel)

    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/train_net_inference_one_pass/train_net/ReconFrame:0')
    res_input = graph.get_tensor_by_name('import/quant_feature:0')
    res_prior_input = graph.get_tensor_by_name('import/quant_z:0')
    motion_input = graph.get_tensor_by_name('import/quant_mv:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')

    with tf.compat.v1.Session(graph=graph) as sess:
        with open(outputfolder + 'quantized_res_feature.pkl', 'rb') as f:
            selected_pixels, residual_feature = pickle.load(f)

        with open(outputfolder + 'quantized_res_prior_feature.pkl', 'rb') as f:
            residual_prior_feature = pickle.load(f)

        with open(outputfolder + 'quantized_motion_feature.pkl', 'rb') as f:
            motion_feature = pickle.load(f)

        # Load dictionary 
        dictionary = np.load('./dictionary.npy')
        
        # Sparse coding
        decoded_image = sparse_coding(selected_pixels, dictionary)
    
        im1 = imageio.imread(refer_path)
        im1 = im1 / 255.0
        im1 = np.expand_dims(im1, axis=0)

        # Reconstructed image
        recon_d = sess.run(
            [reconframe],
            feed_dict={
                res_input: residual_feature,
                res_prior_input: residual_prior_feature,
                motion_input: motion_feature,
                previousImage: im1
            })

        # Save the reconstructed image
        save_image(recon_d[0], outputfolder + 'reconstructed_image.png')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_D.pb',
                        help="decoder model")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./image/170.png',
                        help="refer image path")
    parser.add_argument('--loadpath', type=str, dest="outputfolder", default='./testpkl/',
                        help="saved pkl file")

    args = parser.parse_args()
    decoder(**vars(args))
