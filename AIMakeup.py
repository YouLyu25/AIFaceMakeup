# -*- coding: utf-8 -*-
"""
Created on Tue July 24 2018
@author: You Lyu

Remarks:

landmark represents the X, Y coordinates of the face in the source image

feature_params contains box boundaries (corresponds to the coordinates) of a feature in the image
    Note that the boundary values are rectangle coordinates in the source image coordinate system

for BGR data like image_bgr and feature_bgr (a M*N*3 matrix), M and N are row and column numbers
    while the last 3 indicates that each point (e.g. point(0, 1) or feature_bgr[0, 1, :]) has three
    color channels and the order is Y (G), R and B

for HSV data like image_hsv and feature_hsv (a M*N*3 matrix), M and N are row and column numbers
    while the last 3 indicates that each point (e.g. point(0, 1) or feature_hsv[0, 1, :]) has three
    color channels and the order is Hue, Saturation and Value
    feature_hsv[:, :, 1] represents all the Saturation channel value of the feature

shape like image.shape returns a tuple like (M, N, C) where C is the number of channels and M and N
    represents number of rows and columns respectively

TODO: to replace the sliced part of image, features' landmark (boundary coordinates) is required
"""

import cv2
import dlib
import numpy as np


IMAGE_PATH = "src/image.jpg"
PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
MAX_FACES = 1
IMSHOW_DELAY = 10
RATE = 15
BRIGHTENING_RATE = 1.0
ADJUSTMENT = 8

JAW_RANGE = range(0, 17)
MOUTH_RANGE = range(48, 61)
NOSE_RANGE = range(27, 35)
LEFT_EYE_RANGE = range(42, 48)
RIGHT_EYE_RANGE = range(36, 42)
LEFT_BROW_RANGE = range(22, 27)
RIGHT_BROW_RANGE = range(17, 22)

DEBUG = False


class Feature:
    def __init__(self, landmark, image_bgr, image_hsv):
        """
        :param landmark: landmark (a set of coordinate pairs) points of a specific feature
        :param image_bgr: source face image in BGR
        :param image_hsv: source face image in HSV
        """
        self.feature_landmark = landmark
        self.image_bgr = image_bgr
        self.image_hsv = image_hsv
        # feature_params are parameters of a certain feature which will be used to obtain feature slice
        self.feature_params = self.get_feature_params(image_bgr)

        """
        feature_bgr and feature_hsv are sliced images of the feature, e.g. a mouth, a nose, etc.
        the slicing is done with corresponding feature landmark hence feature_params which contains
            the box boundaries of the face feature
        """
        self.feature_bgr = self.get_feature(self.image_bgr)
        self.feature_hsv = self.get_feature(self.image_hsv)
        self.feature_relative_mask = self.get_feature_relative_mask()
        cv2.waitKey(0)

    def get_gaussian_kernel_size(self, area, rate):
        size = max([int(np.sqrt(area / 3) / rate), 1])
        if size % 2 != 1:
            size = size + 1
        kernel_size = (size, size)
        return kernel_size

    def get_feature_params(self, image):
        """
        get the boundaries of face box

        :return: boundary and shape parameters the feature
        landmark[:, 0] is the array of x coordinates
        landmark[:, 1] is the array of y coordinates
        top/bottom and left/right represent the boundaries of feature box obtained from feature landmark
        """
        x_coordinates = self.feature_landmark[:, 0]
        y_coordinates = self.feature_landmark[:, 1]
        params = dict()
        params["top"] = np.min(y_coordinates)
        params["bottom"] = np.max(y_coordinates)
        params["left"] = np.min(x_coordinates)-ADJUSTMENT
        params["right"] = np.max(x_coordinates)+ADJUSTMENT

        params["shape"] = ((params["bottom"]-params["top"]), params["right"]-params["left"])
        params["area"] = params["shape"][0] * params["shape"][1] * 3
        params["adjustment"] = int(np.sqrt(params["area"]/3)/20)
        # Gaussian convolution kernel (matrix) size which will be used as parameters for Gaussian blur
        params["Gaussian_kernel_size"] = self.get_gaussian_kernel_size(params["area"], RATE)
        params["feature_boundary_y_lower"] = np.max([params["top"]-params["adjustment"], 0])
        params["feature_boundary_y_upper"] = np.min([params["bottom"]+params["adjustment"], image.shape[0]])
        params["feature_boundary_x_lower"] = np.max([params["left"]-params["adjustment"], 0])
        params["feature_boundary_x_upper"] = np.min([params["right"]+params["adjustment"], image.shape[1]])
        if DEBUG:
            print("top", params["top"])
            print("bottom", params["bottom"])
            print("left", params["left"])
            print("right", params["right"])
            print("area", params["area"])
            print("adjustment", params["adjustment"])
            print("feature_boundary_y_lower", params["feature_boundary_y_lower"])
            print("feature_boundary_y_upper", params["feature_boundary_y_upper"])
            print("feature_boundary_x_lower", params["feature_boundary_x_lower"])
            print("feature_boundary_x_upper", params["feature_boundary_x_upper"])
            print("shape", params["shape"])
        return params

    def get_feature(self, image):
        """
        get slicing part of face features

        :param image: source image containing target face
        :param params: parameters related to face shape
        :return: face feature image
        """
        return image[self.feature_params["feature_boundary_y_lower"]:self.feature_params["feature_boundary_y_upper"],
                     self.feature_params["feature_boundary_x_lower"]:self.feature_params["feature_boundary_x_upper"]]

    def get_feature_relative_mask(self):
        landmark = self.feature_landmark.copy()
        if DEBUG:
            print("feature_landmark:\n", landmark)
        # adjust feature landmark points
        landmark[:, 0] -= np.max([self.feature_params["left"]-self.feature_params["adjustment"], 0])
        landmark[:, 1] -= np.max([self.feature_params["top"]-self.feature_params["adjustment"], 0])
        # shape[:2] is (row_number, col_number)
        relative_mask = np.zeros(self.feature_bgr.shape[:2], dtype=np.float64)
        # fill feature shape with ones (white in color) and the rests are zeros (black in color)
        feature_points = cv2.convexHull(landmark)
        cv2.fillConvexPoly(relative_mask, feature_points, color=1)
        relative_mask = np.array([relative_mask, relative_mask, relative_mask]).transpose((1, 2, 0))
        # compare each element in returned matrix with 0, if > 0, change it to 1, else change it to 0
        relative_mask = (cv2.GaussianBlur(relative_mask, self.feature_params["Gaussian_kernel_size"], 0) > 0) * 1.0
        for item in relative_mask:
            print(item)
        return cv2.GaussianBlur(relative_mask, self.feature_params["Gaussian_kernel_size"], 0)

    def brightening(self, rate):
        """
        :param rate: the rate for brightening, higher rate -> brighter
        :return: nothing, the change will be applied directly on the image for testing purpose

        feature_hsv/bgr os a box which encompasses the feature
        feature_relative_mask is a black/white (zeros/white) box which encompasses the feature
            while feature shape is filled with white (ones)
        apply matrix dot multiplication will obtain the HSV/BGR shape (as matrix) of the feature
        any further operation could be applied to the obtained matrix
        """
        feature_hsv = self.feature_hsv[:, :, 1] * self.feature_relative_mask[:, :, 1] * rate
        print(self.feature_hsv.shape)
        print(feature_hsv.shape)
        # Gaussian convolution kernel size is (3, 3)
        feature_hsv = cv2.GaussianBlur(feature_hsv, (3, 3), 0)
        # feature_hsv[:, :, 1] is the red channel of feature_hsv
        self.feature_hsv[:, :, 1] = np.minimum(self.feature_hsv[:, :, 1]+feature_hsv, 255).astype('uint8')
        self.image_bgr = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2BGR)    
        cv2.imshow("changed", self.image_bgr)
        cv2.waitKey(0)

    def transform(self, zoom_to_pixel_x, zoom_to_pixel_y):
        feature_bgr = self.feature_bgr
        zoom_to_pixel = (zoom_to_pixel_x, zoom_to_pixel_y)
        feature_bgr = cv2.resize(feature_bgr, zoom_to_pixel, interpolation=cv2.INTER_AREA)

        width_adjustment = int((zoom_to_pixel_x - self.feature_bgr.shape[1]) / 2)
        height_adjustment = int((zoom_to_pixel_y - self.feature_bgr.shape[0]) / 2)

        left = self.feature_params["feature_boundary_x_lower"]
        right = self.feature_params["feature_boundary_x_upper"]
        top = self.feature_params["feature_boundary_y_lower"]
        bottom = self.feature_params["feature_boundary_y_upper"]
        x_range = zoom_to_pixel_x
        y_range = zoom_to_pixel_y
        self.image_bgr[top:bottom, left:right, 0] = 0
        self.image_bgr[top:bottom, left:right, 1] = 0
        self.image_bgr[top:bottom, left:right, 2] = 0

        if DEBUG:
            print("image.shape[0]", self.image_bgr.shape[0])
            print("image.shape[1]", self.image_bgr.shape[1])
            print("top", top)
            print("bottom", bottom)
            print("left", left)
            print("right", right)
            print("width_adj", width_adjustment)
            print("height_adj", height_adjustment)
            print("x_range", x_range)
            print("y_range", y_range)
            print("shape[0]1", self.feature_bgr.shape[0])
            print("shape[1]2", self.feature_bgr.shape[1])
            print("shape[0]", feature_bgr.shape[0])
            print("shape[1]", feature_bgr.shape[1])

        top_boundary = top - height_adjustment
        bottom_boundary = bottom + height_adjustment
        left_boundary = left - width_adjustment
        right_boundary = right + width_adjustment

        self.image_bgr[top_boundary:bottom_boundary, left_boundary:right_boundary, 0] =\
            feature_bgr[0:y_range, 0:x_range, 0]
        self.image_bgr[top_boundary:bottom_boundary, left_boundary:right_boundary, 1] =\
            feature_bgr[0:y_range, 0:x_range, 1]
        self.image_bgr[top_boundary:bottom_boundary, left_boundary:right_boundary, 2] =\
            feature_bgr[0:y_range, 0:x_range, 2]

        average_color_g = int(np.sum(self.image_bgr[top:bottom, left - 1:left, 0]) / (bottom - top))
        average_color_r = int(np.sum(self.image_bgr[top:bottom, left - 1:left, 1]) / (bottom - top))
        average_color_b = int(np.sum(self.image_bgr[top:bottom, left - 1:left, 2]) / (bottom - top))

        """
        try to get color interval and form a gradual change
        """
        """
        color_step_g = int((self.image_bgr[top:bottom, left_boundary + 1, 0] - self.image_bgr[top:bottom, left - 1, 0]) / (
                    left_boundary - left))
        color_step_r = int((self.image_bgr[top:bottom, left_boundary + 1, 1] - self.image_bgr[top:bottom, left - 1, 1]) / (
                    left_boundary - left))
        color_step_b = int((self.image_bgr[top:bottom, left_boundary + 1, 2] - self.image_bgr[top:bottom, left - 1, 2]) / (
                    left_boundary - left))
        for i in range(left, left_boundary+1):
            self.image_bgr[top:bottom, i, 0] += color_step_g
            self.image_bgr[top:bottom, i, 1] += color_step_r
            self.image_bgr[top:bottom, i, 2] += color_step_b

        color_step_g = int((self.image_bgr[top:bottom, right + 1, 0] - self.image_bgr[top:bottom, right_boundary - 1, 0]) /\
                           (right - right_boundary))
        color_step_r = int((self.image_bgr[top:bottom, right + 1, 1] - self.image_bgr[top:bottom, right_boundary - 1, 1]) /\
                           (right - right_boundary))
        color_step_b = int((self.image_bgr[top:bottom, right + 1, 2] - self.image_bgr[top:bottom, right_boundary - 1, 2]) /\
                           (right - right_boundary))
        for i in range(right_boundary, right + 1):
            self.image_bgr[top:bottom, i, 0] += color_step_g
            self.image_bgr[top:bottom, i, 1] += color_step_r
            self.image_bgr[top:bottom, i, 2] += color_step_b

        color_step_g = int((self.image_bgr[top:top_boundary, right_boundary + 1, 0] -
                            self.image_bgr[top:top_boundary, left_boundary - 1, 0]) / (right_boundary - left_boundary))
        color_step_r = int((self.image_bgr[top:top_boundary, right_boundary + 1, 1] -
                            self.image_bgr[top:top_boundary, left_boundary - 1, 1]) / (right_boundary - left_boundary))
        color_step_b = int((self.image_bgr[top:top_boundary, right_boundary + 1, 2] -
                            self.image_bgr[top:top_boundary, left_boundary - 1, 2]) / (right_boundary - left_boundary))

        for i in range(left_boundary, right_boundary + 1):
            self.image_bgr[top:top_boundary, i, 0] += color_step_g
            self.image_bgr[top:top_boundary, i, 1] += color_step_r
            self.image_bgr[top:top_boundary, i, 2] += color_step_b

        color_step_g = int((self.image_bgr[bottom_boundary:bottom, right_boundary + 1, 0] -
                            self.image_bgr[bottom_boundary:bottom, left_boundary - 1, 0]) / (right_boundary - left_boundary))
        color_step_r = int((self.image_bgr[bottom_boundary:bottom, right_boundary + 1, 0] -
                            self.image_bgr[bottom_boundary:bottom, left_boundary - 1, 0]) / (right_boundary - left_boundary))
        color_step_b = int((self.image_bgr[bottom_boundary:bottom, right_boundary + 1, 0] -
                            self.image_bgr[bottom_boundary:bottom, left_boundary - 1, 0]) / (right_boundary - left_boundary))

        for i in range(left_boundary, right_boundary + 1):
            self.image_bgr[bottom_boundary:bottom, i, 0] += color_step_g
            self.image_bgr[bottom_boundary:bottom, i, 1] += color_step_r
            self.image_bgr[bottom_boundary:bottom, i, 2] += color_step_b
        """
        cv2.imshow("feature", feature_bgr)
        cv2.imshow("check", self.image_bgr)
        cv2.waitKey(0)
        return


# Face class which stores face landmarks and features
class Face:
    def __init__(self, image_bgr, image_hsv, landmark):
        self.feature_range = {"jaw": list(JAW_RANGE),
                              "mouth": MOUTH_RANGE,
                              "nose": NOSE_RANGE,
                              "left eye": LEFT_EYE_RANGE,
                              "right eye": RIGHT_EYE_RANGE,
                              "left brow": LEFT_BROW_RANGE,
                              "right brow": RIGHT_BROW_RANGE}
        # landmark marks all feature points of a face
        self.landmark = landmark
        # instantiate each feature with Feature class
        self.jaw = Feature(landmark[self.feature_range["jaw"]], image_bgr, image_hsv)
        self.mouth = Feature(landmark[self.feature_range["mouth"]], image_bgr, image_hsv)
        self.nose = Feature(landmark[self.feature_range["nose"]], image_bgr, image_hsv)
        self.left_eye = Feature(landmark[self.feature_range["left eye"]], image_bgr, image_hsv)
        self.right_eye = Feature(landmark[self.feature_range["right eye"]], image_bgr, image_hsv)
        self.left_brow = Feature(landmark[self.feature_range["left brow"]], image_bgr, image_hsv)
        self.right_brow = Feature(landmark[self.feature_range["right brow"]], image_bgr, image_hsv)


def load_image(image_path):
    try:
        image_arr = np.fromfile(file=image_path, dtype=np.uint8)
        return cv2.imdecode(buf=image_arr, flags=-1)
    except BaseException as e:
        print(e)
        return None


def get_face_landmarks(detector, predictor, image):
    face_boxes = detector(image, MAX_FACES)
    landmarks = [MAX_FACES]
    for index, face_box in enumerate(face_boxes):
        landmarks[index] = np.zeros(dtype=int, shape=(0, 2))
        full_object_detection = predictor(image, face_box)
        for element in full_object_detection.parts():
            landmarks[index] = np.row_stack((landmarks[index], np.array([element.x, element.y])))

    if DEBUG:
        print("face_boxes:\n", face_boxes)

    # landmark is formatted to a 68x2 matrix
    return landmarks


if __name__ == '__main__':
    '''
    1.  load source picture (in both BGR and HSV format)
    '''
    image_bgr = load_image(IMAGE_PATH)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if DEBUG:
        cv2.imshow("image_bgr", image_bgr)
        cv2.imshow("image_hsv", image_hsv)
        cv2.waitKey(IMSHOW_DELAY)

    '''
    2.  init face detector and predictor (may use cnn model)
    '''
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    '''
    3.  get face landmarks hence feature points
        landmarks is a set of landmark (dlib.full_object_detection object)
        a landmark is a set of x, y coordinates each of which represents
        a point in the feature shape
        using the trained model, there are 68 points in a landmark object
    '''
    face_landmarks = get_face_landmarks(face_detector, face_predictor, image_bgr)
    if DEBUG:
        print("face_landmark:\n", face_landmarks[0])

    '''
    4.  adjust facial features and show results
    '''
    face = Face(image_bgr, image_hsv, face_landmarks[0])
    face.mouth.brightening(1.8)
    # face.mouth.transform(70, 30)


