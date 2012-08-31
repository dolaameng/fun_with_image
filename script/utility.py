## A set of useful tools for basic image processing
## Conventions: 
## Raw-image: image with PIL.
## Image - image in np.array
## gImage -- gray level image

from PIL import Image
from pylab import *
import os, re
from scipy.ndimage import filters

######################## Common Tools
def load_raw_image_group(folder_path, pattern = r'.+\.jpg'):
    """Load all images (right underneath one level) of the folder_path"""
    raw_images = []
    for fname in os.listdir(folder_path):
        if re.match(pattern, fname):
            raw_images.append(Image.open(os.path.join(folder_path, fname)))
    return raw_images
    
    

####################### Image Transformation    

## normalize image intensity using histogram
def histeq(gim, n_bins = 256):
    """Histogram equalization of a grayscale image."""
    ## get image histogram
    imhist, bins = histogram(gim.flatten(), n_bins, normed = True)
    cdf = imhist.cumsum()
    #print cdf[-1]
    cdf = 255 * cdf / cdf[-1]
    ## use linear interpolation of cdf to find new pixel values
    im2 = interp(gim.flatten(), bins[:-1], cdf)
    return im2.reshape(gim.shape), cdf
    
def pca(X):
    """Principal component analysis
    Input: X, matrix with training data stored as flattened arrays in rows
    Output: projection matrix (with important dimensions first), variance and mean"""
    ## get dimensions
    num_data, dim = X.shape
    ## center data
    mean_X = X.mean(axis = 0)
    X = X - mean_X
    # calcualte pca by svd
    if dim > num_data:
        M = dot(X, X.T)
        e, EV = linalg.eigh(M)
        tmp = dot(X.T, EV).T
        V = tmp[::-1]
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        U, S, V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    return V, S, mean_X

## Is an image gray-scale or color image    
def image_type(im):
    """Input: im - np.array representation of an image"""
    imshape = im.shape
    if len(imshape) == 3:
        return 'color'
    elif len(imshape) == 2:
        return 'gray'
    else:
        raise RuntimeException('unknown')

def image_derivatives(im, sigma = 5):
    """Input: np.array represenation of an image"""
    imx = zeros(im.shape)
    imy = zeros(im.shape)
    if image_type(im) == 'gray':
        filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
        filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    elif image_type(im) == 'color':
        for c in range(3):
            filters.gaussian_filter(im[:,:,c], (sigma, sigma), (0, 1), imx[:,:,c] )
            filters.gaussian_filter(im[:,:,c], (sigma, sigma), (1, 0), imy[:,:,c])
            imx, imy = uint8(imx), uint8(imy)
    else:
        raise RuntimeException('unkown type')
    return imx, imy



def test():
    ## test histogram equalization 
    """
    lena = array(Image.open('../data/lena.jpg').convert('L'))
    hlena, _ = histeq(lena)
    figure()
    gray()
    subplot(2, 1, 1)
    imshow(lena)
    subplot(2, 1, 2)
    imshow(hlena)
    """
    ## test pca
    """
    images = load_raw_image_group('../data/a_thumbs')
    images = [array(image.convert('L')) for image in images]
    m, n = images[0].shape[:2]
    X = array([image.flatten() for image in images])
    V, S, immean = pca(X)
    figure()
    gray()
    subplot(2, 4, 1)
    imshow(immean.reshape(m, n))
    for i in range(7):
        subplot(2, 4, i+2)
        imshow(V[i].reshape(m, n))
    """
    ## test image_type
    """
    raw_lena = array(Image.open('../data/lena.jpg'))
    print image_type(raw_lena)
    gray_lena = array(Image.fromarray(raw_lena).convert('L'))
    print image_type(gray_lena)
    """
    """
    ## test image_derivate
    color_lena = array(Image.open('../data/lena.jpg'))
    gray_lena = array(Image.open('../data/lena.jpg').convert('L')) 
    imx, imy = image_derivatives(color_lena, 10)
    figure()
    subplot(1, 3, 1)
    imshow(gray_lena)
    subplot(1, 3, 2)
    imshow(imx)
    subplot(1, 3, 3)
    imshow(imy)
    imx, imy = image_derivatives(gray_lena)
    figure()
    gray()
    subplot(1, 3, 1)
    imshow(gray_lena)
    subplot(1, 3, 2)
    imshow(imx)
    subplot(1, 3, 3)
    imshow(imy)
    """
    
    show()
        


if __name__ == '__main__':
    test()