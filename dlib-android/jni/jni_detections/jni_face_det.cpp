/*
 * jni_pedestrian_det.cpp using google-style
 *
 *  Created on: Oct 20, 2015
 *      Author: Tzutalin
 *
 *  Copyright (c) 2015 Tzutalin. All rights reserved.
 */
#include <android/bitmap.h>
#include <jni_common/jni_bitmap2mat.h>
#include <jni_common/jni_primitives.h>
#include <jni_common/jni_fileutils.h>
#include <jni_common/jni_utils.h>
#include <detector.h>
#include <jni.h>
#include <time.h>


using namespace cv;

extern JNI_VisionDetRet* g_pJNI_VisionDetRet;

namespace {

#define JAVA_NULL 0
using DetectorPtr = DLibHOGFaceDetector*;

class JNI_FaceDet {
 public:
  JNI_FaceDet(JNIEnv* env) {
    jclass clazz = env->FindClass(CLASSNAME_FACE_DET);
    mNativeContext = env->GetFieldID(clazz, "mNativeFaceDetContext", "J");
    env->DeleteLocalRef(clazz);
  }

  DetectorPtr getDetectorPtrFromJava(JNIEnv* env, jobject thiz) {
    DetectorPtr const p = (DetectorPtr)env->GetLongField(thiz, mNativeContext);
    return p;
  }

  void setDetectorPtrToJava(JNIEnv* env, jobject thiz, jlong ptr) {
    env->SetLongField(thiz, mNativeContext, ptr);
  }

  jfieldID mNativeContext;
};

// Protect getting/setting and creating/deleting pointer between java/native
std::mutex gLock;

std::shared_ptr<JNI_FaceDet> getJNI_FaceDet(JNIEnv* env) {
  static std::once_flag sOnceInitflag;
  static std::shared_ptr<JNI_FaceDet> sJNI_FaceDet;
  std::call_once(sOnceInitflag, [env]() {
    sJNI_FaceDet = std::make_shared<JNI_FaceDet>(env);
  });
  return sJNI_FaceDet;
}

DetectorPtr const getDetectorPtr(JNIEnv* env, jobject thiz) {
  std::lock_guard<std::mutex> lock(gLock);
  return getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
}

// The function to set a pointer to java and delete it if newPtr is empty
void setDetectorPtr(JNIEnv* env, jobject thiz, DetectorPtr newPtr) {
  std::lock_guard<std::mutex> lock(gLock);
  DetectorPtr oldPtr = getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
  if (oldPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager delete old ptr : " << oldPtr;
    delete oldPtr;
  }

  if (newPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager set new ptr : " << newPtr;
  }

  getJNI_FaceDet(env)->setDetectorPtrToJava(env, thiz, (jlong)newPtr);
}

}  // end unnamespace

#ifdef __cplusplus
extern "C" {
#endif


#define DLIB_FACE_JNI_METHOD(METHOD_NAME) \
  Java_mango_whale_dlib_FaceDet_##METHOD_NAME

void JNIEXPORT
    DLIB_FACE_JNI_METHOD(jniNativeClassInit)(JNIEnv* env, jclass _this) {}

jobjectArray getDetectResult(JNIEnv* env, DetectorPtr faceDetector,
                             const int& size) {

  clock_t curr = clock();
  LOG(INFO) << "getFaceRet";
  jobjectArray jDetRetArray = JNI_VisionDetRet::createJObjectArray(env, size);
  for (int i = 0; i < size; i++) {
    jobject jDetRet = JNI_VisionDetRet::createJObject(env);
    env->SetObjectArrayElement(jDetRetArray, i, jDetRet);
	
	LOG(INFO) << "interval 1: " << clock() - curr;
	curr = clock();
	
    ////dlib::rectangle rect = faceDetector->getResult()[i];
	cv::Rect rect = faceDetector->getResultCV()[i];
	
	LOG(INFO) << "interval 2: " << clock() - curr;
	curr = clock();
	
    g_pJNI_VisionDetRet->setRect(env, jDetRet, rect.x, rect.y,
                                 rect.x+rect.width, rect.y+rect.height);
    g_pJNI_VisionDetRet->setLabel(env, jDetRet, "face");
	
	LOG(INFO) << "interval 3: " << clock() - curr;
	curr = clock();
	
    std::unordered_map<int, dlib::full_object_detection>& faceShapeMap =
        faceDetector->getFaceShapeMap();
		
	LOG(INFO) << "interval 4: " << clock() - curr;
	curr = clock();
		
    if (faceShapeMap.find(i) != faceShapeMap.end()) {
      dlib::full_object_detection shape = faceShapeMap[i];
      for (unsigned long j = 0; j < shape.num_parts(); j++) {
        int x = shape.part(j).x();
        int y = shape.part(j).y();
        // Call addLandmark
        g_pJNI_VisionDetRet->addLandmark(env, jDetRet, x, y);
      }
    }
	
	LOG(INFO) << "interval 5: " << clock() - curr;
	curr = clock();
	
  }
  return jDetRetArray;
}

JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniDetect)(JNIEnv* env, jobject thiz,
                                    jstring imgPath) {
  LOG(INFO) << "jniFaceDet";
  const char* img_path = env->GetStringUTFChars(imgPath, 0);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  int size = detPtr->det(std::string(img_path));
  env->ReleaseStringUTFChars(imgPath, img_path);
  LOG(INFO) << "det face size: " << size;
  return getDetectResult(env, detPtr, size);
}

#if 0
JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniBitmapDetect)(JNIEnv* env, jobject thiz,
                                          jobject bitmap) {
  
  clock_t curr = clock();											  
  LOG(INFO) << "jniBitmapFaceDet";
  cv::Mat rgbaMat;
  cv::Mat bgrMat;
  jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
  LOG(INFO) << "interval 6: " << clock() - curr;
  curr = clock();
  cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
  LOG(INFO) << "interval 7: " << clock() - curr;
  curr = clock();
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  LOG(INFO) << "interval 8: " << clock() - curr;
  curr = clock();
  jint size = detPtr->det(bgrMat);
#if 0
  cv::Mat rgbMat;
  cv::cvtColor(bgrMat, rgbMat, cv::COLOR_BGR2RGB);
  cv::imwrite("/sdcard/ret.jpg", rgbaMat);
#endif
  LOG(INFO) << "det face size: " << size;
  return getDetectResult(env, detPtr, size);
}
#else
/* 	this function is re-written by You Lyu in 2018/8/10 to improve processing efficiency
	instead of using Bitmap, passing memory address of Mat object and use gray scale format
	as it only contains one channel and it is unneccessary to use a BGR (3 channels) image */
JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniGRAYMatDetect)(JNIEnv* env, jobject thiz,
                                          jlong addrGRAYMat) {
  
  clock_t curr = clock();											  
  LOG(INFO) << "jniBitmapFaceDet";
  cv::Mat& bgrMat = *(cv::Mat*)addrGRAYMat;
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  LOG(INFO) << "interval 8: " << clock() - curr;
  curr = clock();
  jint size = detPtr->det(bgrMat);
  LOG(INFO) << "det face size: " << size;
  LOG(INFO) << "interval 9: " << clock() - curr;
  curr = clock();
  return getDetectResult(env, detPtr, size);
}
#endif






/* this function is re-written by You Lyu in 2018/08/10 to improve the processing efficiency */
jint JNIEXPORT JNICALL DLIB_FACE_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz,
                                                     jstring jLandmarkPath, jstring jClassifierPath) {
  LOG(INFO) << "jniInit";
  std::string landmarkPath = jniutils::convertJStrToString(env, jLandmarkPath);
  std::string classifierPath = jniutils::convertJStrToString(env, jClassifierPath);
  DetectorPtr detPtr = new DLibHOGFaceDetector(landmarkPath, classifierPath);
  setDetectorPtr(env, thiz, detPtr);
  
  return JNI_OK;
}




jint JNIEXPORT JNICALL
    DLIB_FACE_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
  LOG(INFO) << "jniDeInit";
  setDetectorPtr(env, thiz, JAVA_NULL);
  return JNI_OK;
}

#ifdef __cplusplus
}
#endif
