#!/usr/bin/env python

#---------------------------------------------------------------
# warping an image stack with coordinate points
#
# written by Lars v. Buchholtz (2020)
#---------------------------------------------------------------


# use as follows:
# python warp.py --src=[SOURCE_FILENAME] --ref=[REFERENCE_FILENAME] --out=[BASENAME_FOR_OUTPUT] --vec=[VECTOR_FILENAME] --dir=[DIRECTORY]
#


# INPUT:
#
# SOURCE_FILENAME = image stack to be warped (alignment channel in first slice)
#
# REFERENCE_FILENAME = reference image stack that the source is being warped to (alignment channel in first slice)
#
# VECTOR_FILENAME = csv file with source X, source Y, target X, target Y coordintes as columns
#
# DIRECTORY = the directory that these 3 files are in and where the output is generated
#
# BASENAME_FOR_OUTPUT = arbritrary name that is appended by the specific output files

# OUTPUT:
#
# BASENAME_FOR_OUTPUT-outstack.tif = warped stack from SOURCE_FILENAME
# BASENAME_FOR_OUTPUT-input.tif = alignment channels overlaid before warping
# BASENAME_FOR_OUTPUT-overlay.tif = alignment channels overlaid after warping
# BASENAME_FOR_OUTPUT-arrows.svg = vector drawing of arrows connecting source and target coordinates (open in Illustrator or Inkscape)
# BASENAME_FOR_OUTPUT-srctriangles.svg = vector drawing of Delaunay triangles of source coordinates
# BASENAME_FOR_OUTPUT-reftriangles.svg = vector drawing of corresponding triangles of target coordinates


import os
import numpy as np
import cv2
import pandas as pd
import skimage.io
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse

def readVectors(path) :
    # read vector .tsv file and return source and target points as coordinate tuples

    df = pd.read_csv(path,header=0,sep='\t')
    
    points1 = tuple(map(tuple,df.iloc[:,-4:-2].values.astype(np.int32)))
    points2 = tuple(map(tuple,df.iloc[:,-2:].values.astype(np.int32)))

    return points1, points2


def rectContains(rect, point) :
	# Check if a point is inside a rectangle

    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


def calculateDelaunayTriangles(rect, points):
	#calculate delanauy triangles from coordinates and containing rectangle
	# this function was taken from FaceMorph on LearnOpenCV

    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    # get triangles
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
    
    # filter triangles
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points that are within the containing rectangle form a triangle
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri


def applyAffineTransform(src, srcTri, dstTri, size) :
	# Apply affine transform calculated using a single source triangle srcTri 
	# and a single destination triangle dstTri to src and
# output an image of size.
    
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


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []



    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2]), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    # Copy triangular region of the rectangular patch to the output image
    warped_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = warped_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( 1 - mask ) + warpImage1 * mask
    

def warp_image(src_img,trgt_img,points_src,points_trgt):

    # get Rectangle to be used with Subdiv2D
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
	# function to generate a vector graphic svg file that shows the
	# warping vectors as arrows

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
    
    warped_stack = np.zeros((n_channels,) + ref_stack.shape[1:], dtype = src_stack.dtype)

    triangles = calculateDelaunayTriangles(rect, points_ref)
    
    for channel in range(n_channels):
        
        warped_img = np.zeros(ref_stack.shape[1:], dtype = src_stack.dtype)

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


def draw_delaunay(img_size_src,img_size_ref, points_src, points_ref, outname1, outname2):
    img_ratio = img_size_src[1] / img_size_src[0]
    plt.figure(figsize=[img_ratio * 10, 10])
    ax = plt.gca()
    ax.add_artist(Rectangle((0, 0), height=img_size_src[0], width=img_size_src[1],
                            fill=False))

    # Rectangle to be used with Subdiv2D
    r = (0, 0, img_size_src[1], img_size_src[0])

    indices = calculateDelaunayTriangles(r, points_src)

    for t in indices:
        # draw triangle in src image
        pt1 = points_src[t[0]]
        pt2 = points_src[t[1]]
        pt3 = points_src[t[2]]

        draw_line(pt1, pt2, img_size_src)
        draw_line(pt2, pt3, img_size_src)
        draw_line(pt3, pt1, img_size_src)

    ax.axis('off')
    plt.savefig(outname1, transparent=True)

    img_ratio = img_size_ref[1] / img_size_ref[0]
    plt.figure(figsize=[img_ratio * 10, 10])
    ax = plt.gca()
    ax.add_artist(Rectangle((0, 0), height=img_size_ref[0], width=img_size_ref[1],
                            fill=False))
    for t in indices:
        # draw triangle in src image
        pt1 = points_ref[t[0]]
        pt2 = points_ref[t[1]]
        pt3 = points_ref[t[2]]

        draw_line(pt1, pt2, img_size_ref)
        draw_line(pt2, pt3, img_size_ref)
        draw_line(pt3, pt1, img_size_ref)

    ax.axis('off')
    plt.savefig(outname2, transparent=True)


def sort_stack(stack):
    if stack.ndim == 2:
        return stack.reshape((1,) + stack.shape)



if __name__ == '__main__' :    
    # main function
    

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


    ref_path = os.path.join(args.dir,args.ref) if args.dir is not None else args.ref
    src_path = os.path.join(args.dir,args.src) if args.dir is not None else args.src
    vec_path = os.path.join(args.dir,args.vec) if args.dir is not None else args.vec
    out_path = os.path.join(args.dir,args.out) if args.dir is not None else args.out

    ref_stack = skimage.io.imread(ref_path)
    src_stack = skimage.io.imread(src_path)

    src_points, ref_points = readVectors(vec_path)

    # if one of the stacks is a single image turn it into a 1 x array
    if len(ref_stack.shape) == 2:
        ref_stack = ref_stack.reshape((1,) + ref_stack.shape)
    if len(src_stack.shape) == 2:
        src_stack = src_stack.reshape((1,) + src_stack.shape)

    # imread returns different dimension orders for different number of channels/slices
    # so we put the smallest dimension first
    src_stack = np.moveaxis(src_stack, np.argmin(src_stack.shape), 0)
    ref_stack = np.moveaxis(ref_stack, np.argmin(ref_stack.shape), 0)

    # create main output: the warped src stack
    warped_stack = warp_stack(src_stack, ref_stack, src_points, ref_points)

    # create input overlay rgb image

    arr_img = np.zeros(ref_stack.shape[1:] + (3,), dtype=np.uint8)

    arr_img[:, :, 1] = ref_stack[0, :, :]/256 if ref_stack[0, :, :].max().max() > 255 else ref_stack[0, :, :]
    arr_img[:, :, 0] = src_stack[0, :, :]/256 if src_stack[0, :, :].max().max() > 255 else src_stack[0, :, :]

    skimage.io.imsave(out_path + '-input.tif', arr_img)

    # create and vector arrow svg

    draw_vectors(arr_img.shape, src_points, ref_points,
                 out_path + '-arrows.svg')

    # create control overlay image

    over_img = np.zeros(ref_stack.shape[1:] + (3,), dtype=np.uint8)
    over_img[:, :, 1] = ref_stack[0, :, :]/256 if ref_stack[0, :, :].max().max() > 255 else ref_stack[0, :, :]
    over_img[:, :, 0] = warped_stack[0, :, :]/256 if warped_stack[0, :, :].max().max() > 255 else warped_stack[0, :, :]

    skimage.io.imsave(out_path + '-overlay.tif', over_img)


    # create and save ref and src triangle svg's
    draw_delaunay(src_stack.shape[1:], ref_stack.shape[1:], src_points, \
                  ref_points, out_path + '-srctriangles.svg', \
                  out_path + '-reftriangles.svg')

    skimage.external.tifffile.imsave(out_path + '-outstack.tif', warped_stack,
                                     imagej=True, metadata={'axes': 'YXZ'})
