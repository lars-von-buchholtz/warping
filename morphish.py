#!/usr/bin/env python

# use as follows:
# python warp.py --src=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/ --ref=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/IV20180727-tagRFP-brush-hairpull-pinch-airpuff-holding-airpuffwet-hot.tif --out=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/h1 --vec=/Volumes/PASSLARS/analysis_ganglia/IV-2018-07-27-Tac1tagRFP/h1-vectors.tsv
# python warp.py --src=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3-align1.tif --ref=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h1-outstack.tif --out=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3 --vec=/Volumes/BLUEDISK/warping/IV-2018-08-24-Tac1tagRFP/h3-vectors.tsv

import os
import numpy as np
import cv2
import pandas as pd
import skimage.io


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

def readVectors(path):
    # Create an array of points.
    df = pd.read_csv(path, header=0)

    points1 = tuple(map(tuple, df.iloc[:, -4:-2].to_numpy().astype(np.int32)))
    points2 = tuple(map(tuple, df.iloc[:, -2:].to_numpy().astype(np.int32)))

    return points1, points2


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
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

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(
                rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(
                            pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps triangular regions from src_img1 to warped_img
def warpTriangle(src_img, warped_img, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    # r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    # tRect = []

    for i in range(0, 3):
        # tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r2[2], r2[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    # warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    # imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image

    warped_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = warped_img[
                                                           r2[1]:r2[1] + r2[3],
                                                           r2[0]:r2[0] + r2[
                                                               2]] * (
                                                                       1 - mask) + warpImage1 * mask


def warp_image(src_img, trgt_img, points_src, points_trgt):
    # Rectangle to be used with Subdiv2D
    size = trgt_img.shape
    rect = (0, 0, size[1], size[0])

    warped_img = np.zeros(trgt_img.shape, dtype=trgt_img.dtype)

    triangles = calculateDelaunayTriangles(rect, points_trgt)

    for triangle in triangles:
        x, y, z = triangle
        x = int(x)
        y = int(y)
        z = int(z)

        t_src = [points_src[x], points_src[y], points_src[z]]
        t_trgt = [points_trgt[x], points_trgt[y], points_trgt[z]]

        # Morph one triangle at a time.
        warpTriangle(src_img, warped_img, t_src, t_trgt)

    return warped_img


def draw_vectors(rgb_img, points_src, points_trgt):
    draw_color = (255, 255, 255)
    cv2.namedWindow('Vector image', cv2.WINDOW_NORMAL)

    # cv2.imshow("Vector image", rgb_img)

    for i, pt1 in enumerate(points_src):
        pt2 = points_trgt[i]
        cv2.line(rgb_img, tuple(pt1), tuple(pt2), draw_color, 1, cv2.LINE_AA, 0)

    cv2.imshow("Vector image", rgb_img)
    cv2.resizeWindow('Vector image', 1200, 800)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(ref_img, ref_points, delaunay_color = (255, 255, 255)):
    img = ref_img.copy()
    # Rectangle to be used with Subdiv2D
    r = (0, 0, ref_img.shape[1], ref_img.shape[0])

    mask = np.zeros(ref_img.shape[:2], dtype=ref_img.dtype)
    indices = calculateDelaunayTriangles(r, ref_points)

    for t in indices:
        # draw triangle in ref image
        pt1 = ref_points[t[0]]
        pt2 = ref_points[t[1]]
        pt3 = ref_points[t[2]]

        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    return img




def check_write_video(func):
  def inner(self, *args, **kwargs):
    if self.video:
      return func(self, *args, **kwargs)
    else:
      pass
  return inner


class Video(object):
  def __init__(self, filename, fps, w, h):
    self.filename = filename

    if filename is None:
      self.video = None
    else:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      self.video = cv2.VideoWriter(filename, fourcc, fps, (w, h), True)

  @check_write_video
  def write(self, img, num_times=1):
    for i in range(num_times):
      self.video.write(img[..., :3])

  @check_write_video
  def end(self):
    print(self.filename + ' saved')
    self.video.release()

def weighted_average_points(start_points, end_points, percent=0.5):
    """ Weighted average of two sets of supplied points
    :param start_points: *m* x 2 array of start face points.
    :param end_points: *m* x 2 array of end face points.
    :param percent: [0, 1] percentage weight on start_points
    :returns: *m* x 2 array of weighted average points
    """
    if percent <= 0:
        return end_points
    elif percent >= 1:
        return start_points
    else:
        return tuple(map(tuple,(np.asarray(start_points,np.int32)*percent + np.asarray(end_points,np.int32)*(1-percent)).astype(np.int32)))

def morph(src_img1, src_img2, src_points, dest_img, dest_points,
          num_frames=20, fps=10, out_video=None):
    """
    Create a morph sequence from source to destination image
    :param src_img: ndarray source image
    :param src_points: source image array of x,y face points
    :param dest_img: ndarray destination image
    :param dest_points: destination image array of x,y face points
    :param video: facemorpher.videoer.Video object
    """

    stall_frames = np.clip(int(fps*0.15), 1, fps)  # Show first & last longer

    num_frames -= (stall_frames * 2)  # No need to process src and dest image

    video = Video(out_video, fps, dest_img.shape[1], dest_img.shape[0])
    start_img = np.stack([src_img2,src_img1,dest_img],axis=-1).astype(np.uint8)



    video.write(start_img, stall_frames)

    # Produce morph frames!
    for percent in np.linspace(1, 0, num=num_frames):

        points = weighted_average_points(src_points, dest_points, percent)
        morphed1 = warp_image(src_img1, dest_img, src_points, points)
        morphed2 = warp_image(src_img2, dest_img, dest_points, points)
        curr_img = np.stack([morphed2,morphed1,dest_img],axis=-1).astype(np.uint8)
        #if triangles:
       #     average_face = draw_delaunay(average_face, points, delaunay_color=(255, 0, 0))
        # if background in ('transparent', 'average'):
        #   mask = blender.mask_from_points(average_face.shape[:2], points)
        #   average_face = np.dstack((average_face, mask))
        #
        #   if background == 'average':
        #     average_background = blender.weighted_average(src_img, dest_img, percent)
        #     average_face = blender.overlay_image(average_face, mask, average_background)

        video.write(curr_img)


    video.write(curr_img, stall_frames)

    video.end()

if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Morphing 2 Stacks with a Vector file')

    parser.add_argument('--src1', required=True,
                        metavar="/path/to/source_stack.tif",
                        help='Source image green')
    parser.add_argument('--src2', required=True,
                        metavar="/path/to/source_stack.tif",
                        help='Source image blue')
    parser.add_argument('--ref', required=True,
                        metavar="/path/to/to/reference_stack.tif",
                        help="Reference image name (red)")
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

    ref_path = os.path.join(args.dir, args.ref) if hasattr(args,
                                                           'dir') else args.ref
    src1_path = os.path.join(args.dir, args.src1) if hasattr(args,
                                                           'dir') else args.src1
    src2_path = os.path.join(args.dir, args.src2) if hasattr(args,
                                                             'dir') else args.src2
    vec_path = os.path.join(args.dir, args.vec) if hasattr(args,
                                                           'dir') else args.vec
    out_path = os.path.join(args.dir, args.out) if hasattr(args,
                                                           'dir') else args.out

    ref_img = skimage.io.imread(ref_path)
    src_img1 = skimage.io.imread(src1_path)
    src_img2 = skimage.io.imread(src2_path)


    src_points, ref_points = readVectors(vec_path)


    # create morphing movie
    morph(src_img1, src_img2, src_points, ref_img, ref_points, num_frames=40,
          out_video=out_path + '-outvideo.avi')


