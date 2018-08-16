
from config_manager import Config

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

import cv2
import numpy as np
import scipy.spatial
import json

from functools import wraps
from time import clock

import argparse
import os
import sys

sys.setrecursionlimit(10000)

ERR_INPUT_SHAPE = "input must be a bidimensional matrix."
ERR_INPUT_TYPE = "input elements must have type uint8."
ERR_INPUT_NOT_EXISTS = "input image doesn't exist."

MSG_TIME = '{} elapsed time: {} seconds'



def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = clock()
        result = f(*args, **kwargs)
        end = clock()
        print(MSG_TIME.format(f.__name__, end-start))
        return result
    return wrapper

# class Stroke:
#     def __init__(self, label, points):
#         self.label = label
#         self.points = points
#         self.max_x = max(self.points, key=1)
#         self.max_y = max(self.points, key=0)

@timing
def skew_correction(img, edges, threshold=None, _save=False, _save_path="./"):
    # get HoughLinesMap
    h, theta, d = hough_line(edges)
    h_p, theta_p, d_p = None, None, None
    # get peaks of the HoughLinesMap
    if threshold is None:
        h_p, theta_p, d_p = hough_line_peaks(h, theta, d)
    else:
        h_p, theta_p, d_p = hough_line_peaks(h, theta, d, threshold=threshold)
    # get the histogram of the values ​​of the angles
    hist, bins = np.histogram(theta_p, 720)
    # the most probable angle is complementary to the skew
    c_skew = np.rad2deg((bins[np.argmax(hist)]+bins[np.argmax(hist)+1])/2)
    skew = 90-c_skew if c_skew > 0 else -c_skew-90

    # create a rotation matrix and rotate the image around its center
    matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -skew, 1)
    dst = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    if _save:
        _path_name = os.path.join(_save_path, "_skew_hough")
        if not os.path.isdir(_path_name):
            os.mkdir(_path_name)

        hough_lines_all = img.copy()
        hough_lines_true = img.copy()

        diag = np.sqrt(np.power(img.shape[0],2)+np.power(img.shape[1],2))

        for _, angle, dist in zip(*(h_p, theta_p, d_p)):
            a, b = np.cos(angle), np.sin(angle)
            x0, y0 = a*dist, b*dist
            x1 = int(x0 + diag*(-b))
            y1 = int(y0 + diag*(a))
            x2 = int(x0 - diag*(-b))
            y2 = int(y0 - diag*(a))
            cv2.line(hough_lines_all,(x1,y1),(x2,y2),(0,255,0),4)
            if bins[np.argmax(hist)] <= angle <= bins[np.argmax(hist)+1]:
                cv2.line(hough_lines_true,(x1,y1),(x2,y2),(0,255,0),4)

        cv2.imwrite(os.path.join(_path_name, "pre_skew.jpg"), img)
        cv2.imwrite(os.path.join(_path_name, "post_skew.jpg"), dst)
        cv2.imwrite(os.path.join(_path_name, "hough_lines_all.jpg"), hough_lines_all)
        cv2.imwrite(os.path.join(_path_name, "hough_lines_true.jpg"), hough_lines_true)

    return dst

@timing
def swt_transform(
        img,
        edges,
        psi=np.pi/2,
        dark_on_light=True,
        _save=False,
        _save_path="./"):

    assert len(img.shape) is 2, ERR_INPUT_SHAPE
    assert img.dtype == np.uint8, ERR_INPUT_TYPE

    cp_img = 255-img.copy() if dark_on_light else img.copy()
    nr, nc = img.shape

    # Canny Edge detections
    edges = edges.astype(np.bool)

    # x gradient, y-gradient are computed using Sobel operator
    gx = cv2.Sobel(cp_img, cv2.CV_64F, 1, 0, ksize=-1)
    gy = cv2.Sobel(cp_img, cv2.CV_64F, 0, 1, ksize=-1)

    # the gradient magnitude (abs_g) and orientation (ang_g) can be estimated as
    # - sqrt(gx^2+gy^2)
    # - atan(gy/gx)
    # abs_g = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    ang_g = np.arctan2(gy, gx)

    # swt map has the same dimension of the image
    swt = np.empty(cp_img.shape)
    # the initial stroke width of each pixel is infinity
    swt[:] = np.Infinity
    # list of the rays found
    rays = []

    # the cosine and the sine of the gradient angle represent the basic increment for passing
    # from one point of the radius to the next
    cos_g, sin_g = np.cos(ang_g), np.sin(ang_g)

    for i in range(nr):
        for j in range(nc):

            # if the current point is and edge there is a candidate ray
            if edges[i, j]:

                # candidate ray for current stroke width
                ray = [(i, j)]

                # prev_i, prev_j : coordinates of the previous pixel
                # cnt : iteration counter
                prev_i, prev_j, cnt = i, j, 0

                while True:

                    cnt += 1

                    # coordinates of the new current pixel
                    cur_i = int(np.floor(i + sin_g[i, j] * cnt))
                    cur_j = int(np.floor(j + cos_g[i, j] * cnt))

                    # if the new coordinates are within the limits of the image
                    if cur_i < 0 or cur_i >= nr or cur_j < 0 or cur_j >= nc:
                        break

                    #  if the new point is inside the image
                    if cur_i != prev_i or cur_j != prev_j:

                        # if the new point is an edge then the candidate ray ends
                        if edges[cur_i, cur_j]:

                            # append the last point to the candidate ray
                            ray.append((cur_i, cur_j))

                            # a radius is valid if the angle of the gradient at the starting point is approximately
                            # opposite the angle of the gradient at the end point
                            v = np.abs(np.abs(ang_g[i, j] - ang_g[cur_i, cur_j]) - np.pi)
                            if v < psi:

                                # the width of the current stoke is the distance between the start and end points
                                width = np.sqrt(np.power((cur_i - i), 2) + np.power((cur_j - j), 2))

                                for (_i, _j) in ray:
                                    # assign the value to each pixel of the ray if it did not have a smaller value
                                    swt[_i, _j] = min(swt[_i, _j], width)

                                # add the rays to the list
                                rays.append(ray)
                            break
                        # if the new point is not an edge then add the point to the ray
                        ray.append((cur_i, cur_j))

                    # update previous coordinates
                    prev_i, prev_j = cur_i, cur_j

    # np.median():
    # Given a vector V of length N,
    # the median of V is the middle value of a sorted copy of V, V_sorted
    # V_sorted[(N-1)/2], when N is odd, and the average of the two middle values of V_sorted when N is even

    for ray in rays:
        # assign to each pixel in a ray the median swt value of pixels in that ray
        # if less than the previous value
        median = np.median([swt[i, j] for (i, j) in ray])
        for (i, j) in ray:
            swt[i, j] = min(median, swt[i, j])

    if _save:
        _path_name = os.path.join(_save_path, "_save")
        if not os.path.isdir(_path_name):
            os.mkdir(_path_name)

        cv2.imwrite(os.path.join(_path_name, "edges.jpg"), edges.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(_path_name, "gx.jpg"), gx)
        cv2.imwrite(os.path.join(_path_name, "gy.jpg"), gy)
        cv2.imwrite(os.path.join(_path_name, "ang_g.jpg"), (ang_g+np.pi)*255/(2*np.pi))
        cv2.imwrite(os.path.join(_path_name, "cos_g.jpg"), (cos_g+1)*255/2)
        cv2.imwrite(os.path.join(_path_name, "sin_x.jpg"), (cos_g+1)*255/2)

    return swt


@timing
def swt_segmentation(swt):

    # labels map initialized to 0
    labels = np.zeros(swt.shape, dtype=np.uint32)
    # layers list
    strokes = []

    # number of rows and columns of swt map
    nr, nc = swt.shape

    # first valid label and region
    label = 1

    for i in range(nr):
        for j in range(nc):

            # if the current pixel is in a stroke
            # assign it to a region with the current label
            # search ... for similar swt value
            if np.Infinity > swt[i, j] > 0 and labels[i, j] == 0:

                # list of the point in the current region
                stroke = []
                point_list = [(i, j)]
                labels[i, j] = label

                stroke.append((i, j))

                _min_i, _min_j = i, j
                _max_i, _max_j = i, j

                while len(point_list) > 0:
                    pi, pj = point_list.pop(0)
                    for ni in range(max(pi-1, 0), min(pi+2, nr-1)):
                        for nj in range(max(pj-1, 0), min(pj+2, nc-1)):
                            if labels[ni, nj] == 0 and np.Infinity > swt[ni, nj] > 0:

                                if 0.3 < swt[ni, nj]/swt[pi, pj] < 3.0 and 0.3 < swt[pi, pj]/swt[ni, nj] < 3.0:
                                    labels[ni, nj] = label
                                    point_list.append((ni, nj))

                                    stroke.append((ni, nj))
                                    # max and min x and y in current stroke
                                    _min_i, _min_j = min(_min_i, ni), min(_min_j, nj)
                                    _max_i, _max_j = max(_max_i, ni), max(_max_j, nj)

                width = _max_j-_min_j
                height = _max_i-_min_i
                diagonal = np.sqrt(np.power(width, 2) + np.power(height, 2))
                median = np.median([swt[e] for e in stroke])

                # add new layer
                strokes.append(
                    (
                        label,
                        stroke,
                        diagonal,
                        median,
                        (_min_i, _min_j),
                        (_max_i, _max_j),
                        width,
                        height
                    )
                )
                # pass to the next label and region
                label += 1

    return labels, strokes


@timing
def swt_extract_letters(
        swt,
        strokes,
        min_width=1,
        min_height=1,
        max_width=1.,
        max_height=1.,
        width_height_ratio=9.,
        height_width_ratio=9.,
        min_diag_mswt_ratio=3.,
        max_diag_mswt_ratio=21.,
    ):

    letters = []
    for stroke in strokes:

        _, _, diagonal, median, _, _, width, height = stroke

        # skip the stroke if too small
        if width < min_width or height < min_height:
            continue

        wh_ratio = width/height
        hw_ratio = height/width

        # skip the stroke if width/height is too big
        if wh_ratio > width_height_ratio:
            continue

        # skip the stroke if height/width is too big
        if hw_ratio > height_width_ratio:
            continue
        
        dm_ratio = diagonal/median
        # skip the stroke if .....
        if dm_ratio > max_diag_mswt_ratio or dm_ratio < min_diag_mswt_ratio:
            continue

        # skip the stroke if .....
        if width/swt.shape[1] > max_width or height/swt.shape[0] > max_height:
            continue

        letters.append(stroke)
    return letters

@timing
def swt_extract_words(
        letters,
        thresh_pairs_y=2,
        thresh_mswt=8,
        thresh_height=8,
        width_scale=1.5,
        height_scale=1.0,
        _save=False,
        _save_path="./"):

    stop_y_1d_list = np.asarray([[e[5][0]] for e in letters])
    tree_stop_y = scipy.spatial.KDTree(stop_y_1d_list)
    pairs_stop_y = set(tree_stop_y.query_pairs(thresh_pairs_y))

    start_y_1d_list = np.asarray([[e[4][0]] for e in letters])
    tree_start_y = scipy.spatial.KDTree(start_y_1d_list)
    pairs_start_y = set(tree_start_y.query_pairs(thresh_pairs_y))

    _elements = pairs_start_y.union(pairs_stop_y)

    chains = []
    for pair in _elements:

        id_a, id_b = pair[0], pair[1]
        letter_a, letter_b = letters[id_a], letters[id_b]

        widest = max(letter_a[6], letter_b[6])
        highest = max(letter_a[7], letter_b[7])

        distance_x = abs(letter_a[4][1]-letter_b[4][1])
        distance_y = abs(letter_a[4][0]-letter_b[4][0])

        if abs(letter_a[3]-letter_b[3]) < thresh_mswt:
            if abs(letter_a[7]-letter_b[7]) < thresh_height:
                if distance_x < widest * width_scale and distance_y < height_scale*highest:
                    added = False
                    for chain in chains:
                        if id_a in chain:
                            chain.add(id_b)
                            added = True
                        elif id_b in chain:
                            chain.add(id_a)
                            added = True
                    if not added:
                        chains.append({id_a, id_b})

    words = []
    _u = True
    while _u:
        _u = False
        i = 0
        while i < len(chains)-1:
            j = i+1
            while j < len(chains):
                if len(chains[i].intersection(chains[j])):
                    _u = True
                    chains[j] = chains[j].union(chains[i])
                    del chains[i]
                if _u: break
                j += 1
            if _u: break
            i += 1

    for chain in chains:
        words.append([])
        for idx in chain:
            words[len(words)-1].extend(letters[idx][1])
    return words


def extract_contours(element_list):
    return [(e[4][::-1], e[5][::-1]) for e in element_list]


def extract_contours_from_words(words):
    _l = []
    for word in words:
        x = np.asarray([point[1] for point in word])
        y = np.asarray([point[0] for point in word])
        _l.append(((np.min(x), np.min(y)), (np.max(x), np.max(y))))
    return _l


def draw_words(img, words):
    cp_img = img.copy()
    if len(img.shape) is 2:
        cp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for word in words:
        color = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        for point in word:
            cp_img[point[0], point[1]] = color

    return cp_img


def draw_contours(_img, _l):
    cp_img = _img.copy()

    if len(_img.shape) is 2:
        cp_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)

    for (p1, p2) in _l:
        cv2.rectangle(cp_img, p1, p2, (255, 0, 0), 2)
    return cp_img


def swt2png(swt):
    png_swt = np.zeros((swt.shape[0],swt.shape[1],4))
    for i in range(swt.shape[0]):
        for j in range(swt.shape[1]):
            if swt[i,j] == np.Infinity:
                png_swt[i,j] = [0, 0, 0, 0]
            else:
                png_swt[i,j] = [swt[i,j], swt[i,j], swt[i,j], 255]
    return png_swt

@timing
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("conf", help="JSON configuration file", type=str)
    args = parser.parse_args()

    # Load configuration file
    try:
        with open(args.conf, "r") as f:
            config = Config(**json.load(f))
    except Exception as e:
        print(e)
        return

    # if save is enabled crete the output folder if not exists
    if config.save:
        if not os.path.isdir(config.output):
            os.mkdir(config.output)

    # read the input image
    img = cv2.imread(config.input)
    # check if is a valid image
    if img is None:
        print(ERR_INPUT_NOT_EXISTS)
        return

    # convert the image to gray-scale
    gray_img = img.copy()
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = gray_img

    # get edges map
    thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(cv2.GaussianBlur(gray_img, (5, 5), 0), thresh/2, thresh, apertureSize=3)

    # if deskew is enabled 
    if config.deskew.enable:
        # deskew with hough lines based method
        img = skew_correction(
            img, 
            edges, 
            threshold=config.deskew.threshold,
            _save=config.save,
            _save_path=config.output)

        gray_img = img.copy()
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = gray_img
        thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(cv2.GaussianBlur(gray_img, (5, 5), 0), thresh/2, thresh, apertureSize=3)

    # apply Stroke Width Transform
    swt = swt_transform(
        gray_img,
        edges,
        psi=np.pi/2,
        dark_on_light=config.dark_on_light,
        _save=config.save,
        _save_path=config.output
    )
    
    # apply segmentation on swt ang get layers
    _, layers = swt_segmentation(swt)

    # filters layers in letter
    letters = swt_extract_letters(
        swt,
        layers,
        min_width=config.letters.min_width,
        min_height=config.letters.min_height,
        max_width=config.letters.max_width,
        max_height=config.letters.max_height,
        width_height_ratio=config.letters.width_height_ratio,
        height_width_ratio=config.letters.height_width_ratio,
        min_diag_mswt_ratio=config.letters.min_diag_mswt_ratio,
        max_diag_mswt_ratio=config.letters.max_diag_mswt_ratio,
    )
    
    # union letters in words
    words = swt_extract_words(
        letters,
        thresh_pairs_y=config.words.thresh_pairs_y,
        thresh_mswt=config.words.thresh_mswt,
        thresh_height=config.words.thresh_height,
        width_scale=config.words.width_scale,
        height_scale=config.words.height_scale,
        _save=config.save,
        _save_path=config.output
    )


    if config.save:

        cv2.imwrite("{}/swt.jpg".format(config.output), swt)
        cv2.imwrite("{}/swt.png".format(config.output), swt2png(swt))
        cv2.imwrite("{}/layers.jpg".format(config.output), draw_contours(img, extract_contours(layers)))
        cv2.imwrite("{}/letters.jpg".format(config.output), draw_contours(img, extract_contours(letters)))
        cv2.imwrite("{}/words_connection.jpg".format(config.output), draw_words(img, words))
        cv2.imwrite("{}/words_box.jpg".format(config.output), draw_contours(img, extract_contours_from_words(words)))


if __name__ == "__main__":
    main()
