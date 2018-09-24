/*
 * detector.h using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once

#include <jni_common/jni_fileutils.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
#include <glog/logging.h>
#include <jni.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <time.h>


using namespace dlib;
using namespace std;


class OpencvHOGDetctor {
 public:
  OpencvHOGDetctor() {}

  inline int det(const cv::Mat& src_img) {
    if (src_img.empty())
      return 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::vector<cv::Rect> found, found_filtered;
    hog.detectMultiScale(src_img, found, 0, cv::Size(8, 8), cv::Size(32, 32),
                         1.05, 2);
    size_t i, j;
    for (i = 0; i < found.size(); i++) {
      cv::Rect r = found[i];
      for (j = 0; j < found.size(); j++)
        if (j != i && (r & found[j]) == r)
          break;
      if (j == found.size())
        found_filtered.push_back(r);
    }

    for (i = 0; i < found_filtered.size(); i++) {
      cv::Rect r = found_filtered[i];
      r.x += cvRound(r.width * 0.1);
      r.width = cvRound(r.width * 0.8);
      r.y += cvRound(r.height * 0.06);
      r.height = cvRound(r.height * 0.9);
      cv::rectangle(src_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
    }
    mResultMat = src_img;
    // cv::imwrite(path, mResultMat);
    LOG(INFO) << "det ends";
    mRets = found_filtered;
    return found_filtered.size();
  }

  inline cv::Mat& getResultMat() { return mResultMat; }

  inline std::vector<cv::Rect>& getResult() { return mRets; }

 private:
  cv::Mat mResultMat;
  std::vector<cv::Rect> mRets;
};



class DLibHOGDetector {
 private:
  typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6>> image_scanner_type;
  dlib::object_detector<image_scanner_type> mObjectDetector;

  inline void init() {
    LOG(INFO) << "Model Path: " << mModelPath;
    if (jniutils::fileExists(mModelPath)) {
      dlib::deserialize(mModelPath) >> mObjectDetector;
    } else {
      LOG(INFO) << "Not exist " << mModelPath;
    }
  }

 public:
  DLibHOGDetector(const std::string& modelPath = "/sdcard/person.svm")
      : mModelPath(modelPath) {
    init();
  }

  virtual inline int det(const std::string& path) {
    using namespace jniutils;
    if (!fileExists(mModelPath) || !fileExists(path)) {
      LOG(WARNING) << "No modle path or input file path";
      return 0;
    }
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if (src_img.empty())
      return 0;
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    int im_size_min = MIN(img_width, img_height);
    int im_size_max = MAX(img_width, img_height);

    float scale = float(INPUT_IMG_MIN_SIZE) / float(im_size_min);
    if (scale * im_size_max > INPUT_IMG_MAX_SIZE) {
      scale = (float)INPUT_IMG_MAX_SIZE / (float)im_size_max;
    }

    if (scale != 1.0) {
      cv::Mat outputMat;
      cv::resize(src_img, outputMat,
                 cv::Size(img_width * scale, img_height * scale));
      src_img = outputMat;
    }

    // cv::resize(src_img, src_img, cv::Size(320, 240));
    dlib::cv_image<dlib::bgr_pixel> cimg(src_img);

    double thresh = 0.5;
    mRets0 = mObjectDetector(cimg, thresh);
    return mRets0.size();
  }

  inline std::vector<dlib::rectangle> getResult() { return mRets0; }

  virtual ~DLibHOGDetector() {}

 protected:
  std::vector<dlib::rectangle> mRets0;
  std::string mModelPath;
  const int INPUT_IMG_MAX_SIZE = 800;
  const int INPUT_IMG_MIN_SIZE = 600;
};







/* 	the function is re-written by You Lyu in 2018/8/10 to improve processing efficiency
	opencv classifier is used instead as it gives faster face landmarks extraction speed */
class DLibHOGFaceDetector : public DLibHOGDetector {
 protected:
  std::vector<cv::Rect> mRets;
 private:
  std::string mLandMarkModel;
  std::string mClassifier;
  cv::CascadeClassifier face_cascade;
  dlib::shape_predictor msp;
  std::unordered_map<int, dlib::full_object_detection> mFaceShapeMap;
  dlib::frontal_face_detector mFaceDetector;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = dlib::get_frontal_face_detector();
  }

 public:
  DLibHOGFaceDetector() { init(); }

  DLibHOGFaceDetector(const std::string& landmarkmodel, const std::string& classifier)
      : mLandMarkModel(landmarkmodel), mClassifier(classifier) {
    init();
    if (!mLandMarkModel.empty() && jniutils::fileExists(mLandMarkModel) &&
		!mClassifier.empty() && jniutils::fileExists(mClassifier)) {
      face_cascade.load(classifier);
      dlib::deserialize(mLandMarkModel) >> msp;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
	else {
      LOG(INFO) << "Landmark or classifer model not found";
	}
  }

  virtual inline int det(const std::string& path) {
    LOG(INFO) << "Read path from " << path;
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    return det(src_img);
  }

  // The format of mat should be BGR or Gray
  // If converting 4 channels to 3 channls because the format could be BGRA or
  // ARGB
  /* this function is re-written by You Lyu in 2018/08/10 to improve processing efficiency */	
  virtual inline int det(const cv::Mat& image) {
    if (image.empty())
      return 0;
    LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";
	LOG(INFO) << "Mat channels: " << image.channels();
	
	CHECK(image.channels() == 1);
	
	clock_t curr = clock();
//================================================================================================================	
	//cv::equalizeHist(img, img);
	face_cascade.detectMultiScale(image, mRets, 1.2, 2, 0|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(20, 20));
//================================================================================================================	
	LOG(INFO) << "interval 10: " << clock() - curr;
	curr = clock();
	LOG(INFO) << "Dlib HOG face det size : " << mRets.size();
	dlib::cv_image<uchar> img(image);
	LOG(INFO) << "get image";
    mFaceShapeMap.clear();
    // Process shape
    if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
      for (unsigned long j = 0; j < mRets.size(); ++j) {
		dlib::rectangle det;
		det.set_left(mRets[j].x);
        det.set_top(mRets[j].y);
        det.set_right(mRets[j].x+mRets[j].width);
        det.set_bottom(mRets[j].y+mRets[j].height);
        dlib::full_object_detection shape = msp(img, det);
        LOG(INFO) << "face index:" << j
                  << " number of parts: " << shape.num_parts();
        mFaceShapeMap[j] = shape;
      }
    }
	LOG(INFO) << "interval 11: " << clock() - curr;
	curr = clock();
    return mRets.size();
  }
  
  
  inline std::vector<cv::Rect> getResultCV() { return mRets; }
  
  
  std::unordered_map<int, dlib::full_object_detection>& getFaceShapeMap() {
    return mFaceShapeMap;
  }
};






