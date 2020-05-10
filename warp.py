#!/usr/bin/env python

# use as follows:
# python warp.py --src=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/ --ref=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/IV20180727-tagRFP-brush-hairpull-pinch-airpuff-holding-airpuffwet-hot.tif --out=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/h1 --vec=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/h1-vectors.tsv
# python warp.py --src=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3-align1.tif --ref=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h1-outstack.tif --out=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3 --vec=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3-vectors.tsv

import os
import numpy as np
import cv2
import pandas as pd
import skimage.io
from matplotlib.patches import Ellipse, Rectangle, Polygon

import matplotlib.pyplot as plt

def readVectors(path) :
    # Create an array of points.
    df = pd.read_csv(path,header=0,sep='\t')
    
    points1 = tuple(map(tuple,df.iloc[:,-4:-2].values.astype(np.int32)))
    points2 = tuple(map(tuple,df.iloc[:,-2:].values.astype(np.int32)))

    
    return points1, points2

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps triangular regions from src_img1 to warped_img
def warpTriangle(src_img, warped_img, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    #r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    #tRect = []


    for i in range(0, 3):
        #tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2]), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r2[2], r2[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    #warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    #imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image

    warped_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = warped_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( 1 - mask ) + warpImage1 * mask
    

def warp_image(src_img,trgt_img,points_src,points_trgt):
    # Rectangle to be used with Subdiv2D
    size = trgt_img.shape
    rect = (0, 0, size[1], size[0])
    
    warped_img = np.zeros(trgt_img.shape, dtype = trgt_img.dtype)

    triangles = calculateDelaunayTriangles(rect, points_trgt)
    
    for triangle in triangles:
        x,y,z = triangle
        x = int(x)
        y = int(y)
        z = int(z)

        t_src = [points_src[x], points_src[y], points_src[z]]
        t_trgt = [points_trgt[x], points_trgt[y], points_trgt[z]]
        

        # Morph one triangle at a time.
        warpTriangle(src_img, warped_img, t_src, t_trgt)

    return warped_img
    
def draw_line(pt1,pt2,img_size):
    plt.plot([pt1[0], pt2[0]], [img_size[0] - pt1[1], img_size[0] - pt2[1]],color='black',linewidth=1)



def draw_vectors(img_size ,points_src,points_trgt,out_name):
    img_ratio = img_size[1] / img_size[0]
    plt.figure(figsize=[img_ratio * 10,10])
    ax = plt.gca()

    ax.add_artist(Rectangle((0,0),height=img_size[0],width=img_size[1],fill=False))

    for i,pt1 in enumerate(points_src):
        pt2 = points_trgt[i]
        ax.arrow(pt1[0], img_size[0] - pt1[1], pt2[0] - pt1[0],
                  pt1[1] - pt2[1], color='black', linewidth=1)

    plt.ylim([0,img_size[0]])
    plt.xlim([0,img_size[1]])

    ax.axis('off')
    plt.savefig(out_name, transparent=True)


def warp_stack(src_stack,ref_stack,points_src,points_ref):
    
    
    # Rectangle to be used with Subdiv2D
    size = ref_stack.shape[-2:]

    rect = (0, 0, size[1], size[0])
    
    # make new image
    n_channels = src_stack.shape[0]
    
    warped_stack = np.zeros((n_channels,) + ref_stack.shape[1:], dtype = ref_stack.dtype)

    triangles = calculateDelaunayTriangles(rect, points_ref)
    
    for channel in range(n_channels):
        
        warped_img = np.zeros(ref_stack.shape[1:], dtype = ref_stack.dtype)

        for triangle in triangles:

            x,y,z = triangle
            x = int(x)
            y = int(y)
            z = int(z)

            t_src = [points_src[x], points_src[y], points_src[z]]
            t_trgt = [points_ref[x], points_ref[y], points_ref[z]]


            # Morph one triangle at a time.
            warpTriangle(np.expand_dims(src_stack[channel,:,:],axis=-1), warped_img, t_src, t_trgt)

        warped_stack[channel,:,:] = warped_img
        
    return warped_stack

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_delaunay(img_size, points,outname):
    img_ratio = img_size[1] / img_size[0]
    plt.figure(figsize=[img_ratio * 10,10])
    ax = plt.gca()
    ax.add_artist(Rectangle((0, 0), height=img_size[0], width=img_size[1],
                            fill=False))


    # Rectangle to be used with Subdiv2D
    r = (0, 0, img_size[1], img_size[0])
    
    indices = calculateDelaunayTriangles(r, points)
    
    for t in indices :
        
        # draw triangle in ref image
        pt1 = points[t[0]]
        pt2 = points[t[1]]
        pt3 = points[t[2]]
        
        draw_line(pt1,pt2,img_size)
        draw_line(pt2,pt3,img_size)
        draw_line(pt3,pt1,img_size)

    ax.axis('off')
    plt.savefig(outname, transparent= True)

def draw_arrows(img,src_points,ref_points,arrow_color):
    
    for i,pt1 in enumerate(src_points):
        pt2 = ref_points[i]
        cv2.arrowedLine(img, pt1, pt2, arrow_color, 2)
        
    return img

def sort_stack(stack):
    if stack.ndim == 2:
        return stack.reshape((1,) + stack.shape)



if __name__ == '__main__' :    
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Morphing 2 Stacks with a Vector file')
    
    parser.add_argument('--src', required=True,
                        metavar="/path/to/source_stack.tif",
                        help='Source stack name')
    parser.add_argument('--ref', required=True,
                        metavar="/path/to/to/reference_stack.tif",
                        help="Reference stack name")
    parser.add_argument('--out', required=True,
                        metavar="/path/to/output",
                        help='Output file base name')
   
    parser.add_argument('--vec', required=True,
                        metavar="path/to/vectors.csv",
                        help="Vector file name (.csv)")
    parser.add_argument('--dir', required=False,
                        metavar="/base/directory/",
                        help='directory name')
    args = parser.parse_args()


    ref_path = os.path.join(args.dir,args.ref) if hasattr(args,'dir') else args.ref
    src_path = os.path.join(args.dir,args.src) if hasattr(args,'dir') else args.src
    vec_path = os.path.join(args.dir,args.vec) if hasattr(args,'dir') else args.vec
    out_path = os.path.join(args.dir,args.out) if hasattr(args,'dir') else args.out


    ref_stack = skimage.io.imread(ref_path)
    src_stack = skimage.io.imread(src_path)
    
    src_points,ref_points = readVectors(vec_path)

    # if one of the stacks is a single image turn it into a 1 x array
    if len(ref_stack.shape) == 2:
        ref_stack = ref_stack.reshape((1,) + ref_stack.shape)
    if len(src_stack.shape) == 2:
        src_stack = src_stack.reshape((1,) + src_stack.shape)

    # imread returns different dimension orders for different number of channels/slices
    # so we put the smallest dimension first
    src_stack = np.moveaxis(src_stack,np.argmin(src_stack.shape),0)
    ref_stack = np.moveaxis(ref_stack,np.argmin(ref_stack.shape),0)
    
    # create main output: the warped src stack
    warped_stack = warp_stack(src_stack,ref_stack,src_points,ref_points)
    
    # create input overlay rgb image
    
    arr_img = np.zeros(ref_stack.shape[1:] + (3,), dtype = ref_stack.dtype)
    arr_img[:,:,1] = ref_stack[0,:,:]
    arr_img[:,:,0] = src_stack[0,:,:]

    skimage.io.imsave(out_path + '-input.tif', arr_img)

    # create and vector arrow svg

    draw_vectors(arr_img.shape,src_points,ref_points,out_path + '-arrows.svg')

    # create control overlay image
    
    over_img = np.zeros(ref_stack.shape[1:] + (3,), dtype = ref_stack.dtype)
    over_img[:,:,1] = ref_stack[0,:,:]
    over_img[:,:,0] = warped_stack[0,:,:]
    skimage.io.imsave(out_path + '-overlay.tif', over_img)

    
    #create and save ref and src triangle svg's
    draw_delaunay(src_stack.shape[1:], src_points,out_path + '-srctriangles.svg')
    draw_delaunay(ref_stack.shape[1:], ref_points,out_path + '-reftriangles.svg')
    


    #skimage.io.imsave(out_path + '-outstack.tif', warped_stack,plugin='tifffile',metadata={'axes': 'ZYX'})
    skimage.external.tifffile.imsave(out_path + '-outstack.tif', warped_stack,imagej=True,metadata={'axes': 'YXZ'})
