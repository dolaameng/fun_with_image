import os
from PIL import Image
from pylab import *
import pydot

from utility import load_raw_image_group

def sift_image(raw_im, params = '--edge-thresh 10 --peak-thresh 5'):
    """Get the sift of an image"""
    ## output image to a temp file
    imname = '/tmp/tmp.pgm'
    siftname = '/tmp/tmp.sift'
    raw_im.convert('L').save(imname)
    ## call sift bin
    cmd = str("sift" + " " + imname + " "
                + "--output=" + siftname
                + " " + params)
    os.system(cmd)
    ## read features
    feat = loadtxt(siftname)
    ## x, y, scale, rotation angle
    ## rest features
    return feat[:,:4], feat[:,4:]
    
def match_descriptors(desc1, desc2, dist_ratio = 0.6):
    """For each descriptor in the first image,
    select its match in the second image.
    Input: desc1 (descriptor for the first image)
    desc2 (descriptor for the second image)
    They are usually extracted from the sift_image methods (2nd params)
    dist_ratio: distance ratio threshold"""
    
    ## normalize descriptors
    desc1 = array([d / linalg.norm(d) for d in desc1])
    desc2 = array([d / linalg.norm(d) for d in desc2])
    desc1_size = desc1.shape #ndesc, nfeat
    
    matchscores = zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T # precomput the matrix transpose
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:], desc2t) * 0.99999
        ## inverse cosine and sort, return index for features in second image
        indx = argsort(arccos(dotprods))
        ## check if nearast neighbor has angle less than dist_ratio 2nd
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
            
    return matchscores
    
def bimatch_descriptors(desc1, desc2, dist_ratio = 0.6):
    matches12 = match_descriptors(desc1, desc2)
    matches21 = match_descriptors(desc2, desc1)
    ndx12 = matches12.nonzero()[0]
    ## remove matches that are not symmetric
    for n in ndx12:
        if matches21[int(matches12[n])] != n:
            matches12[n] = 0
    return matches12
    
def descriptor_match_scores(raw_images, dist_ratio = 0.6):
    """Match each pair of images in the images (PIL images)"""
    nbr_images = len(raw_images)
    matchscores = zeros((nbr_images, nbr_images))
    for i in range(nbr_images):
        for j in range(i, nbr_images):
            pi, di = sift_image(raw_images[i])
            pj, dj = sift_image(raw_images[j])
            matches = bimatch_descriptors(di, dj, dist_ratio = 0.6)
            nbr_matches = sum(matches > 0)
            matchscores[i, j] = nbr_matches
            matchscores[j, i] = nbr_matches
            
    return matchscores

def test_geo_matching():
    raw_images = load_raw_image_group('../data/whitehouses/')
    nbr_images = len(raw_images)
    #matchscores = descriptor_match_scores(raw_images)
    #savetxt('whitehouse_match.txt', matchscores, '%i')
    matchscores = loadtxt('whitehouse_match.txt')
    ## construct the dot graph based on matchscores
    g = pydot.Dot(graph_type = 'graph') # not default directed graph
    threshold = 2 # similiarity threshold
    for i in range(nbr_images):
        for j in range(i+1, nbr_images):
            if matchscores[i, j] > threshold:
                ## first image in pair, create thumbnail
                thumbimg = '/tmp/' + str(i) + '.png'
                raw_images[i].thumbnail((100, 100))
                raw_images[i].save(thumbimg)
                g.add_node(pydot.Node(str(i), fontcolor = 'transparent', 
                                shape = 'rectangle', image = thumbimg)) 
                ## second image in pair
                thumbimg = '/tmp/' + str(j) + '.png'
                raw_images[j].thumbnail((100, 100))
                raw_images[j].save(thumbimg)
                g.add_node(pydot.Node(str(j), fontcolor = 'transparent',
                                shape = 'rectangle', image = thumbimg))
                g.add_edge(pydot.Edge(str(i), str(j)))
    g.write_png('whitehouses.png')
    print 'similiarity diagram generated at', 'whitehouses.png'
    
    
def test():
    ## test sift_image
    
    im = Image.open('../data/lena.jpg').convert('L')
    def draw_circle(c, r):
        t = arange(0, 1.01, .01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x, y, 'b', linewidth = 2)
    figure()
    gray()
    imshow(array(im))
    locs, _ = sift_image(im)
    plot(locs[:,0], locs[:,1], 'ob')
    
    ## test match_descriptor
    """
    im = Image.open('../data/lena.jpg').convert('L')
    pts1, feats1 = sift_image(im)
    pts2, feats2 = sift_image(im)
    matchscores = match_descriptors(feats1, feats2)
    for i, m in enumerate(matchscores):
        assert matchscores[i, 0] == m
    """
    ## test bimatch_descriptor
    """
    im = Image.open('../data/a_thumbs/22_t.jpg').convert('L')
    pts1, feats1 = sift_image(im)
    pts2, feats2 = sift_image(im)
    matchscores = bimatch_descriptors(feats1, feats2)
    assert all([matchscores[i, 0] == m for (i, m) in enumerate(matchscores)])
    """
    ## test descriptor_match_scores
    """
    images = load_raw_image_group('../data/whitehouses/')
    matchscores = descriptor_match_scores(images)
    print matchscores
    """
    ## test matching_geotagged images
    """
    test_geo_matching()
    """
    
           
    show()
    
if __name__ == '__main__':
    test()