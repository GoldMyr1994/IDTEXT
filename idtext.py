import argparse
import json
import os
import sys

from functools import wraps
from operator import itemgetter
from time import clock

import numpy as np
import scipy.spatial

from matplotlib import pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

import cv2
from config_manager import Config

sys.setrecursionlimit(10000)

ERR_INPUT_SHAPE = "input must be a bidimensional matrix."
ERR_INPUT_TYPE = "input elements must have type uint8."
ERR_INPUT_NOT_EXISTS = "input image doesn't exist."

MSG_TIME = '{} elapsed time: {} seconds'


class Stroke:
    def __init__(self, label, points, img, swt, skip_edges=False):

        self.label = label

        self.points = points

        self.strokes_values = [swt[i,j] for i,j in points]

        self.colors_values = [img[i,j] for i,j in points]

        self.max_x = max(self.points, key=itemgetter(1))[1]
        self.max_y = max(self.points, key=itemgetter(0))[0]
        self.min_x = min(self.points, key=itemgetter(1))[1]
        self.min_y = min(self.points, key=itemgetter(0))[0]

        if skip_edges: 
            self.max_x += 1 
            self.max_y -= 1
            self.min_x += 1 
            self.min_y -= 1

        self.start_point = self.min_y, self.min_x
        self.stop_point = self.max_y, self.max_x
        
        self.width = float(self.max_x-self.min_x)
        self.height = float(self.max_y-self.min_y)

        self.img_w = float(img.shape[1])
        self.img_h = float(img.shape[0])

        self.diag = None
        
        self.mswt = None

        self.dmswt_ratio = None

        self.mean_swt = None
        self.variance_swt = None

        self.mean_color = None
        self.variance_color = None

        self.hw_ratio = None
        self.wh_ratio = None

        self.wW_ratio = None
        self.hH_ratio = None
    
    def calc(self):

        self.center = int((self.max_y+self.min_y)/2), int((self.max_x+self.min_x)/2)

        self.diag = float(np.sqrt(self.width*self.width+self.height*self.height))
        
        self.mswt = float(np.median(self.strokes_values))

        self.dmswt_ratio = float(self.diag/self.mswt)

        self.mean_swt = float(np.mean(self.strokes_values))
        self.variance_swt = float(np.var(self.strokes_values))

        self.mean_color = float(np.mean(self.colors_values))
        self.variance_color = float(np.var(self.colors_values))

        self.hw_ratio = float(self.height/self.width)
        self.wh_ratio = float(self.width/self.height)

        self.wW_ratio = float(self.width/self.img_w)
        self.hH_ratio = float(self.height/self.img_h)


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = clock()
        result = f(*args, **kwargs)
        end = clock()
        print(MSG_TIME.format(f.__name__, end-start))
        return result
    return wrapper



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
    hist, bins = np.histogram(theta_p, 360)
    # the most probable angle is complementary to the skew
    c_skew = np.rad2deg((bins[np.argmax(hist)]+bins[np.argmax(hist)+1])/2)
    skew = 90-c_skew if c_skew > 0 else -c_skew-90

    # create a rotation matrix and rotate the image around its center
    matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -skew, 1)
    dst = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)

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
    return dst, -np.deg2rad(skew)


@timing
def swt_transform(
        img,
        edges,
        psi=np.pi/2,
        dark_on_light=True,
        skip_edges=False,
        _save=False,
        _save_path="./",
        ):

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
                ray = [(i, j)] if not skip_edges else []

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
                            if not skip_edges: 
                                ray.append((cur_i, cur_j))

                            # a radius is valid if the angle of the gradient at the starting point is approximately
                            # opposite the angle of the gradient at the end point
                            v = np.abs(np.abs(ang_g[i, j] - ang_g[cur_i, cur_j]) - np.pi)
                            if v < psi:

                                if len(ray)==0:# and skip_edges
                                    break

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
        _path_name = os.path.join(_save_path, "_save{}".format(dark_on_light))
        if not os.path.isdir(_path_name):
            os.mkdir(_path_name)

        cv2.imwrite(os.path.join(_path_name, "edges.jpg"), edges.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(_path_name, "gx.jpg"), gx)
        cv2.imwrite(os.path.join(_path_name, "gy.jpg"), gy)
        cv2.imwrite(os.path.join(_path_name, "ang_g.jpg"), (ang_g+np.pi)*255/(2*np.pi))
        cv2.imwrite(os.path.join(_path_name, "cos_g.jpg"), (cos_g+1)*255/2)
        cv2.imwrite(os.path.join(_path_name, "sin_x.jpg"), (cos_g+1)*255/2)


        # lista = []
        # for i in range(swt.shape[0]):
        #     for j in range(swt.shape[1]):
        #         if swt[i,j]<np.Infinity:
        #             lista.append(img)
        # plt.figure()

    return np.asarray(swt)


@timing
def swt_segmentation(swt, gray_img, color_thresh=25, skip_edges=False):

    assert len(gray_img.shape) is 2, ERR_INPUT_SHAPE

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

                #swt_mean_values = [swt[i,j]]
                #swt_mean = swt[i,j]
                #color_mean = gray_img[i,j]

                # list of the point in the current region
                stroke = []
                point_list = [(i, j)]
                labels[i, j] = label

                stroke.append((i, j))

                while len(point_list) > 0:
                    pi, pj = point_list.pop(0)
                    for ni in range(max(pi-1, 0), min(pi+2, nr-1)):
                        for nj in range(max(pj-1, 0), min(pj+2, nc-1)):
                            
                            if labels[ni, nj] == 0 and np.Infinity > swt[ni, nj] > 0:
                                
                                if 0.333 < swt[ni, nj]/swt[i,j] < 3.0:
                                    # labels[ni, nj] = label
                                    # point_list.append((ni, nj))
                                    # stroke.append((ni, nj))
                                    #if abs(color_mean-gray_img[i,j])<color_thresh:
                                        
                                    labels[ni, nj] = label
                                    point_list.append((ni, nj))
                                    stroke.append((ni, nj))
                                    #swt_mean_values.append(swt[ni,nj])
                                    #color_mean = np.mean([gray_img[e] for e in stroke])
                                    #swt_mean = np.mean(swt_mean_values)

                # add new layer
                strokes.append(Stroke(label,stroke,gray_img,swt))

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

        if stroke.width < min_width: 
            continue
        if stroke.height < min_height: 
            continue

        stroke.calc()

        #if stroke.variance_swt>stroke.mean_swt/2:
        #    continue

        if stroke.wh_ratio > width_height_ratio: 
            continue
        if stroke.hw_ratio > height_width_ratio: 
            continue
        if stroke.dmswt_ratio < min_diag_mswt_ratio:
            continue
        if stroke.dmswt_ratio > max_diag_mswt_ratio: 
            continue
        if stroke.wW_ratio > max_width: 
            continue
        if stroke.hH_ratio > max_height: 
            continue

        letters.append(stroke)

    return letters


@timing
def swt_extract_words(
        letters,
        thresh_pairs_y=2,
        thresh_mswt=8,
        thresh_height=8,
        width_scale=3.0,
        _save=False,
        _save_path="./"):

    if len(letters)==0:
        return [],[]
    cp_letters = letters[:]

    stop_y_1d_list = np.asarray([[e.stop_point[0]] for e in letters])
    tree_stop_y = scipy.spatial.KDTree(stop_y_1d_list)
    pairs_stop_y = set(tree_stop_y.query_pairs(thresh_pairs_y))

    start_y_1d_list = np.asarray([[e.start_point[0]] for e in letters])
    tree_start_y = scipy.spatial.KDTree(start_y_1d_list)
    pairs_start_y = set(tree_start_y.query_pairs(thresh_pairs_y))

    _elements = pairs_start_y.union(pairs_stop_y)

    chains = []
    for pair in _elements:

        id_a, id_b = pair[0], pair[1]
        letter_a, letter_b = letters[id_a], letters[id_b]

        widest = max(letter_a.width, letter_b.width)

        d_x = abs(letter_a.start_point[1]-letter_b.start_point[1])

        if abs(letter_a.mswt-letter_b.mswt) < thresh_mswt:
            if abs(letter_a.height-letter_b.height) < thresh_height:
                if d_x < width_scale*widest:
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

                    if letters[id_a] in cp_letters:
                        cp_letters.remove(letters[id_a])
                    if letters[id_b] in cp_letters:
                        cp_letters.remove(letters[id_b])

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
            words[len(words)-1].append(letters[idx])
    
    return words, cp_letters


def draw_strokes_contours(img, strokes, _color=(0,0,255), _size=4):
    cp_img = img.copy()

    if len(img.shape) is 2:
        cp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for (p1, p2) in [(stroke.start_point[::-1], stroke.stop_point[::-1]) for stroke in strokes]:
        cv2.rectangle(cp_img, p1, p2, _color, _size)
    return cp_img

def draw_strokes_centers(img, strokes, _radius=4, _color=(0,0,255), _size=4):
    cp_img = img.copy()

    if len(img.shape) is 2:
        cp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for stroke in strokes:
        cv2.circle(cp_img, stroke.center[::-1], _radius, _color, _size)
    return cp_img

def draw_strokes_connections(img, strokes):
    cp_img = img.copy()
    if len(img.shape) is 2:
        cp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for stroke in strokes:
        color = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        for point in stroke.points:
            cp_img[point] = color

    return cp_img

def get_stroke_from_word(word, swt, gray_img):
    labels = []
    point_list = []
    for stroke in word:
        labels.append(stroke.label)
        point_list.extend(stroke.points)
    s = Stroke(labels, point_list, gray_img, swt)
    s.calc()
    return s

def get_strokes_from_words(words, swt, gray_img):
    strokes = []
    for word in words:
        strokes.append(get_stroke_from_word(word, swt, gray_img))
    return strokes
    
def create_letters_edge_image(gray_img, edges, letters):
    cpimg = np.zeros(gray_img.shape,dtype=np.uint8)
    for l in letters:
        for i,j in l.points:
            if edges[i,j]:
                cpimg[i,j] = 255 
    return cpimg


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
    gray_img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    #thresh, _ = cv2.threshold(gray_img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(gray_img_blur, thresh/2, thresh, apertureSize=3)

    skew = 0.
    if config.deskew:
        # apply Stroke Width Transform
        swt = swt_transform(
            gray_img,
            edges,
            psi=np.pi/2,
            dark_on_light=config.dark_on_light,
            _save=config.save,
            _save_path=config.output,
        )
        
        # apply segmentation on swt ang get layers
        _, layers = swt_segmentation(swt, gray_img )

        # filters layers in letter
        letters = swt_extract_letters(
            gray_img,
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
        
        letters_image = create_letters_edge_image(gray_img, edges, letters)

        if config.save:
            cv2.imwrite("{}/pre_swt.jpg".format(config.output), swt)
            cv2.imwrite("{}/pre_layers_conn.jpg".format(config.output), draw_strokes_connections(img, layers))
            cv2.imwrite("{}/pre_layers.jpg".format(config.output), draw_strokes_contours(img, layers))
            cv2.imwrite("{}/pre_letters.jpg".format(config.output), draw_strokes_contours(img, letters))
            cv2.imwrite("{}/pre_letters_conmn.jpg".format(config.output), draw_strokes_connections(img, letters))
            cv2.imwrite("{}/deskew_points.jpg".format(config.output), letters_image)
        


        img, skew = skew_correction(
            img, 
            letters_image, 
            threshold=None,
            _save=config.save,
        _save_path=config.output)

        gray_img = img.copy()
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = gray_img

        gray_img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        #thresh, _ = cv2.threshold(gray_img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(gray_img_blur, thresh/2, thresh, apertureSize=3)

    # apply Stroke Width Transform
    swt = swt_transform(
        gray_img,
        edges,
        psi=np.pi/2,
        dark_on_light=config.dark_on_light,
        skip_edges=config.swt_skip_edges,
        _save=config.save,
        _save_path=config.output, 
    )
    #swt[ gray_img>np.mean(gray_img) ] = np.Infinity
    
    # apply segmentation on swt ang get layers
    _, layers= swt_segmentation(swt, gray_img, skip_edges=config.swt_skip_edges,)

    # filters layers in letter
    letters = swt_extract_letters(
        gray_img,
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
    words_h, _ = swt_extract_words(
        letters,
        thresh_pairs_y=config.words.thresh_pairs_y,
        thresh_mswt=config.words.thresh_mswt,
        thresh_height=config.words.thresh_height,
        width_scale=config.words.width_scale,
        _save=config.save,
        _save_path=config.output
    )

    words_h_strokes = get_strokes_from_words(words_h, swt, gray_img)


    if config.save and config.gt:

        aaa = draw_strokes_centers(draw_strokes_contours(draw_strokes_connections(img, words_h_strokes), words_h_strokes), words_h_strokes)
        matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -np.rad2deg(skew), 1)
        dst = cv2.warpAffine(aaa, matrix, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
        cv2.imwrite("{}/words_all_original.jpg".format(config.output), dst)
       
        _txt = "" 
        _words = words_h_strokes
        _label = 0
        for _w in _words:
            

            cx = ((float)(img.shape[1]))/2
            cy = ((float)(img.shape[0]))/2

            wcx = (float)(_w.center[1])
            wcy = (float)(_w.center[0])

            _corr_x = cx + (wcx-cx)*np.cos(skew)-(wcy-cy)*np.sin(skew) - wcx
            _corr_y = cy + (wcx-cx)*np.sin(skew)+(wcy-cy)*np.cos(skew) - wcy

            # print("word center x",wcx)
            # print("word center y",wcy)

            # print("displacement x",_corr_x)
            # print("displacement y",_corr_y)

            # print("minx:{} miny:{} maxx:{} maxy:{}".format(_w.min_x,_w.min_y,_w.max_x,_w.max_y))

            # add o.nico to regards!!!!!!!!

            if skew < -np.pi/4:
                
                _txt += "{} {} {} {} {} {} {} \n".format(
                    _label,
                    0,
                    int((wcx+_corr_x) - abs(((_w.min_y - wcy)))),
                    int((wcy+_corr_y) - abs(((_w.min_x - wcx)))),
                    int(_w.height),
                    int(_w.width),
                    np.pi/2+skew
                )
            else:
                _txt += "{} {} {} {} {} {} {} \n".format(
                    _label,
                    0,
                    int(_w.min_x+_corr_x),
                    int(_w.min_y+_corr_y),
                    int(_w.width),
                    int(_w.height),
                    skew
                    )
            _label += 1

        with open('{}/gt.gt'.format(config.output), 'w') as file:
            file.write(_txt)            

    #####

    if config.save:

        # save swt data
        cv2.imwrite("{}/swt.jpg".format(config.output), swt)

        # save layers data: box, connection
        cv2.imwrite("{}/layers.jpg".format(config.output), draw_strokes_contours(img, layers))
        cv2.imwrite("{}/layers_connection.jpg".format(config.output), draw_strokes_connections(img, layers))
        cv2.imwrite("{}/layers_all.jpg".format(config.output), draw_strokes_contours(draw_strokes_connections(img, layers), layers))
       
        # save letters data: box, connection, center
        cv2.imwrite("{}/letters.jpg".format(config.output), draw_strokes_contours(img, letters))
        cv2.imwrite("{}/letters_connection.jpg".format(config.output), draw_strokes_connections(img, letters))
        cv2.imwrite("{}/letters_center.jpg".format(config.output), draw_strokes_centers(img, letters))
        cv2.imwrite("{}/letters_all.jpg".format(config.output), draw_strokes_centers(draw_strokes_contours(draw_strokes_connections(img, letters), letters), letters))
       
        # save letters data: box, connection, center
        cv2.imwrite("{}/words_box.jpg".format(config.output), draw_strokes_contours(img, words_h_strokes))
        cv2.imwrite("{}/words_connection.jpg".format(config.output), draw_strokes_connections(img, words_h_strokes))
        cv2.imwrite("{}/words_center.jpg".format(config.output), draw_strokes_centers(img, words_h_strokes))
        cv2.imwrite("{}/words_all.jpg".format(config.output), draw_strokes_centers(draw_strokes_contours(draw_strokes_connections(img, words_h_strokes), words_h_strokes), words_h_strokes))
       

if __name__ == "__main__":
    main()
