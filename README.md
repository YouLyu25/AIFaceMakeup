# This is a readme file for AIMakeup app.

## **Project Overview**
This project is a sample project that uses openCV and dlib to add makeups (like
deeper colored lips, whiter and smoother skin etc.) to both real-time video stream
and still image.

Note that dlib does not have Android platform supported, hence some porting work
should be carried out. There are ported Android version of dlib available online,
and the java/C++ calls are carried out via JNI.
please check out: https://github.com/tzutalin/dlib-android

Be sure to have a certain level of understanding of what is JNI and how it works
in Android as for some adjustment and optimization may have to be implemented in
C++ (as in native dlib).

OpenCV itself has Android support, check out official website to get a new and
steady version of this library if necessary.

The app has been installed in a pad and its name is "Makeup".


## **Porting**
As mentioned above, dlib is ported in this project for face landmark detection
purpose. The folder named "dlib-android" contains all the files and libs necessary
for this project. Note that some changes are made in the jni .cpp files to improve
the detection efficiency. For further improvement, refer to files under dlib-android/jni
directory as they contain the implementations of java/C++ interface.

After making any adjustment to the native .cpp files, make sure to re-compile it
and replace the contents in libs directory (be sure to keep the already existed 
opencv lib intact) of your app with the newly generated libs under dlib-android/libs.

The approaches of how the libs are generated can be found in:
https://github.com/tzutalin/dlib-android

Here are reminders:
- 1. The libs used are under dlib-android/libs and are classified according to differerent ABIs (x86, arm etc.), use them just as ordinary .so files in Android.
- 2. The files in package mango.whale.dlib of this app are java classes that call the native jni functions, changes can be made for improvement.
- 3. Be sure to also include openCV as a module of this project, there are tones of resources online of how to do this.


## **Project Structure**
The Makeup directory is the android app root directory and contains all the files
for the app project.

Note that as the dlib-android package is exceedingly large (> 3 GB), only the
java/C++ interface files are uploaded, they are under dlib-android/jni directory.

The data/ directory and AIMakeup.py file are for makeup reference using dlib and
opencv. the .dat file under data/ are the pre-tranied face landmark model used for
face detection. To further implement the makeup project, refer to the .py file
for general methodologies.

If you open the project in Android Studio and in project view, you can see that
there are app and openCVLibrary342 modules.

app module contains the source files and layouts for the app. There are three activities in this project:
MainActivity, RealTimeMakeupActivity and ImageMakeupActivity.

- MainActivity contains buttons for directing to the other 2 activities.
- RealTimeMakeupActivity displays real-time image data after processed and added with "makeups". The camera is used for capturing the real-time image and the result is shown in a preview surface.
- ImageMakeupActivity displays both the original image and the image after processed for comparison. The images are loaded from the storage device of pad using a fixed path (which can be altered for more flexible use of this function, e.g. add makeup to the picture just taken).


## **Methodology**
In this project, the real-time image captured from the camera are used as the
source to be processed. When a image is ready, its data can be obtained by the callback
function "onImageAvailable()".

The data will be used as the resource of face detection. after that, the face rectangle
will ge obtained and the landmarks of face features can be extracted. Note that the
model used in this project will be loaded from the device storage space when RealTimeMakeupActivity
is created.

After that, the landmark points will be used to get the "slice" of each feature part
of face, as a result, the boxes which encompass features can be obtained.

Using opencv and landmark points, the convex hull of each feature can be obtained,
then by manipulating the convex hull and matrix of zeros and ones, the mask of
each feature will be generated. Then by using the mask, the pixels within the shape
of a feature can be manipulated. For the deeper lip color operation, simply change
the pixels' channel representing brightness in HSV format of the source image Mat.
Other makeup operations can be implemented with similar idea, although some further
research and investigation should be carried out for better quality.
check out the .py file named AIMaleup.py, it will give a general idea of how makeup
can be added by manipulating matrices.



## **Performace and Future Work**
The overall performance of this app in terms of accuracy is acceptable, while that of
processing speed is questionable.
To improve the effciency, some changes are made to the ported dlib java/C++ interface:
- Gray scale image is used as the source of face detection, the prototype of the JNI call is changed to use a Mat as the input image instead of a Bitmap object.
- Dlib provides a more accurate result of face detection (i.e. getting face rectangle) in a compromise of the processing speed, hence the opencv cascade classifier used to detect face and the result is used as the resource dlib's feature landmark extraction.

Note that the most time-consuming operation is the face detection, it may consume
over 85% of the total image processing time. Future improvement of the processing
speed may focus on speed up the face detection operation.

Possible improvement may be made by:
- Reasonably reducing searching space (e.g. only use a part of the captured image as the detection resource or simply reduce the resolution...), if no face is detected, expand the searching space under certain rules.
- Using better model (the model used now is a 68 points model from opencv), a "liter" model with acceptable deviation may be used to speed up the processing as the makeup may not have to be very accurate.
- Using GPU in someway (probably?) as it provides high parallism as suitable for processing data like pixels. Or use another way to render the image (instead of process the capture one and return the result for preview), maybe check out OpenGL?
