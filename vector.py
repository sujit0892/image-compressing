import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy import misc

def build_arg_parser():
    parser =argparse.ArgumentParser(description='compress input image')
    parser.add_argument("--input-file", dest="input_file", required=True, help="Input image")
    parser.add_argument("--num-bits", dest="num_bits", required=False, type=int, help="no. of bits")
    return parser

def compress_image(img, num_cluster):
    X = img.reshape((-1,1))
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)
    return input_image_compressed

def plot_image(img,title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)


if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError('Number of bits should be between 1 and 8')

    num_clusters = np.power(2, num_bits)

    # Print compression rate
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, 2)
    print("\nThe size of the image will be reduced by a factor of", 8.0/args.num_bits)
    print("\nCompression rate = " + str(compression_rate) + "%")

    # Load input image
    input_image = misc.imread(input_file, True).astype(np.uint8)

    # original image 
    plot_image(input_image, 'Original image')

    # compressed image 
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, 'Compressed image; compression rate = ' 
            + str(compression_rate) + '%')

    plt.show()

