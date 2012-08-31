## Extract features of dense SIFT from image. Another common
## name is Histogram of Oriented Gradients (HOG)

import os
from PIL import Image
from pylab import *

def dsift_image(raw_im, size = 20, 
                steps = 10, 
                force_orientation = False, resize = None):
    """Proces an image with densely sampled SIFT descriptors and get
    features of dense sift. 
    Optional Input: size of features, steps between locations,
    forcing computation of descriptor orientation 
    (False means all are oriented upward), tuple for resizing the image"""
    gim = raw_im.convert('L')
    if resize:
        gim = gim.resize(resize)
    m, n = gim.size
    
    imname = '/tmp/tmp.pgm'
    siftname = '/tmp/tmp.sift'
    framename = '/tmp/tmp.frame'
    gim.save(imname)
    
    ## create frames and save to temporary file
    scale = size / 3.0
    x, y = meshgrid(range(steps, m, steps), range(steps, n, steps))
    xx,yy = x.flatten(), y.flatten()
    frame = array([xx, yy, scale * ones(xx.shape[0]), zeros(xx.shape[0])])
    savetxt(framename, frame.T, fmt = '%03.3f')
    
    if force_orientation:
        cmd = str('sift' + ' ' + imname + ' '
                    + '--output=' + siftname + ' '
                    + '--read-frames=' + framename
                    + ' ' + '--orientations')
    else:
        cmd = str('sift' + ' ' + imname + ' '
                    + '--output=' + siftname + ' '
                    '--read-frames=' + framename)
    os.system(cmd)
    feat = loadtxt(siftname)
    ## x, y, scale, rotation angle
    ## rest features
    return feat[:,:4], feat[:,4:]
    
def test():
    ## test dsift_image
    lena = Image.open('../data/lena.jpg')
    pts, descs = dsift_image(lena)
    figure()
    imshow(array(lena))
    plot(pts[:, 0], pts[:, 1], 'r*')
    
    show()
    
if __name__ == '__main__':
    test()