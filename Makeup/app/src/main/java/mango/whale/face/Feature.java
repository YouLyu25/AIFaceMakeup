package mango.whale.face;

import android.graphics.Point;
import android.support.annotation.NonNull;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;


/**
 * Created by You Lyu on 2018/8/6.
 * Implementations of Feature class
 */
@SuppressWarnings("ALL")
public class Feature {
    private static final String TAG = "Feature DEBUG";
    private static final int GAUSSIAN_KERNEL_RATE = 15;
    private static final int RED_CHANNEL = 1;
    private static final int BRIGHT_CHANNEL = 1;
    private static final int CHANNEL_NUM = 3;
    private static final int MAX_COLOR_VAL = 255;
    private static final int MAX_BRIGHT_VAL = 255;
    private ArrayList<Point> featureLandmarks;
    private Mat imageRGB;
    private Mat imageHSV;
    public Mat featureRGB;
    public Mat featureHSV;
    public Mat mask;
    public Mat featureRelativeMask;
    private Map<String, Integer> featureParams;
    private Size ksize;



    public Feature(Mat RGB, Mat HSV, ArrayList<Point> landmarks) {
        imageRGB = RGB;
        imageHSV = HSV;
        featureLandmarks = landmarks;
        featureParams = getFeatureParams();
        getFeature();
        getFeatureRelativeMask();
    }



    private Map<String, Integer> getFeatureParams() {
        Map<String, Integer> params = new HashMap<>();
        int xMax = 0;
        int yMax = 0;
        int xMin = imageRGB.rows();
        int yMin = imageRGB.cols();

        for (int i = 0; i < featureLandmarks.size(); ++i) {
            Point point = featureLandmarks.get(i);
            if (point.x >= xMax) {
                xMax = point.x;
            }
            if (point.x <= xMin) {
                xMin = point.x;
            }

            if (point.y >= yMax) {
                yMax = point.y;
            }
            if (point.y <= yMin) {
                yMin = point.y;
            }
        }

        params.put("top", yMin);
        params.put("bottom", yMax);
        params.put("left", xMin);
        params.put("right", xMax);
        params.put("area", (yMax-yMin)*(xMax-xMin)*3);
        params.put("adjustment", (int)(sqrt(params.get("area")/3)/20));
        params.put("boundaryYLow", max(params.get("top")-params.get("adjustment"), 0));
        params.put("boundaryYUp", min(params.get("bottom")+params.get("adjustment"), imageRGB.rows()));
        params.put("boundaryXLow", max(params.get("left")-params.get("adjustment"), 0));
        params.put("boundaryXUp", min(params.get("right")+params.get("adjustment"), imageRGB.cols()));
        Log.d("DEBUG", "top: " + params.get("top"));
        Log.d("DEBUG", "bottom: " + params.get("bottom"));
        Log.d("DEBUG", "left: " + params.get("left"));
        Log.d("DEBUG", "right: " + params.get("right"));
        Log.d("DEBUG", "area: " + params.get("area"));
        Log.d("DEBUG", "adjustment: " + params.get("adjustment"));
        Log.d("DEBUG", "boundaryYLow: " + params.get("boundaryYLow"));
        Log.d("DEBUG", "boundaryYUp: " + params.get("boundaryYUp"));
        Log.d("DEBUG", "boundaryXLow: " + params.get("boundaryXLow"));
        Log.d("DEBUG", "boundaryXUp: " + params.get("boundaryXUp"));


        return params;
    }



    @NonNull
    private Size getGaussianKernelSize(int area, int rate) {
        int size = max((int)(sqrt(area/3)/rate), 1);
        if (size % 2 != 1) {
            size = size + 1;
        }
        return new Size(size, size);
    }



    /*
        get the Mat of a certain feature
        e.g. if the Feature object being processed is jaw
        then featureBGR and featureHSV are color representations
        of a box which encompass the jaw on the face
    */
    private void getFeature() {
        int xLow = featureParams.get("boundaryXLow");
        int xUp = featureParams.get("boundaryXUp");
        int yLow = featureParams.get("boundaryYLow");
        int yUp = featureParams.get("boundaryYUp");

        featureRGB = imageRGB.submat(yLow, yUp, xLow, xUp);
        featureHSV = imageHSV.submat(yLow, yUp, xLow, xUp);
    }


    /*
        get the masks necessary for processing face features
        TODO: the sample implementation of getting face feature is coded as method brightening()
        TODO: the future manipulation of face features, like widening the eyes,
        TODO: whitening/smoothening the skin can be implemented with simlilar principle
    */
    private void getFeatureRelativeMask() {
        List<org.opencv.core.Point> landmarkPoints = new ArrayList<>();
        int xOffset = max(featureParams.get("left")-featureParams.get("adjustment"), 0);
        int yOffset = max(featureParams.get("top")-featureParams.get("adjustment"), 0);
        for (Point element: featureLandmarks) {
            double x = element.x - xOffset;
            double y = element.y - yOffset;
            org.opencv.core.Point point = new org.opencv.core.Point(x, y);
            landmarkPoints.add(point);
        }
        MatOfPoint landmarkMat = new MatOfPoint();
        landmarkMat.fromList(landmarkPoints);
        for (int i = 0; i < landmarkMat.rows(); ++i) {
            for (int j = 0; j < landmarkMat.cols(); ++j) {
                double point = landmarkMat.get(i, j)[0];
            }
        }

        //mask = Mat.zeros(featureRGB.rows(), featureRGB.cols(), CvType.CV_8U);
        mask = new Mat(featureRGB.rows(), featureRGB.cols(), CvType.CV_8UC1, new Scalar(0));
        MatOfInt hull = new MatOfInt();
        //------------------------------------------------------------------------------------------
        Imgproc.convexHull(landmarkMat, hull);
        //------------------------------------------------------------------------------------------

        org.opencv.core.Point[] landmarkArray = landmarkMat.toArray();
        org.opencv.core.Point[] hullPoints = new org.opencv.core.Point[hull.rows()];
        List<Integer> hullContourIdxList = hull.toList();
        for (int i = 0; i < hullContourIdxList.size(); i++) {
            hullPoints[i] = landmarkArray[hullContourIdxList.get(i)];
        }
        MatOfPoint hullMat = new MatOfPoint(hullPoints);

        //------------------------------------------------------------------------------------------
        Imgproc.fillConvexPoly(mask, hullMat, new Scalar(1));
        //------------------------------------------------------------------------------------------

        featureRelativeMask = new Mat(mask.rows(), mask.cols(), CvType.CV_8UC3);
        for (int i = 0; i < mask.rows(); ++i) {
            for (int j = 0; j < mask.cols(); ++j) {
                double pixel = mask.get(i, j)[0];
                double[] pixel3 = new double[3];
                try {
                    for (int c = 0; c < CHANNEL_NUM; ++c) {
                        pixel3[c] = pixel;
                    }
                }
                catch (Exception e) {
                    Log.d(TAG, "error pixel: " + pixel);
                }
                featureRelativeMask.put(i, j, pixel3);
            }
        }
        ksize = getGaussianKernelSize(featureParams.get("area"), GAUSSIAN_KERNEL_RATE);

        Imgproc.GaussianBlur(featureRelativeMask, featureRelativeMask, ksize, 0);
        for (int i = 0; i < featureRelativeMask.rows(); ++i) {
            for (int j = 0; j < featureRelativeMask.cols(); ++j) {
                double[] pixel = featureRelativeMask.get(i, j);
                try {
                    for (int c = 0; c < CHANNEL_NUM; ++c) {
                        if (pixel[c] > 0) {
                            pixel[c] = 1;
                        }
                        featureRelativeMask.put(i, j, pixel);
                    }
                }
                catch (Exception e) {
                    break;
                }
            }
        }

        Imgproc.GaussianBlur(featureRelativeMask, featureRelativeMask, ksize, 0);
    }



    public void brightening(double rate) {
        Mat HSV = new Mat(featureHSV.rows(), featureHSV.cols(), CvType.CV_8U);
        for (int i = 0; i < featureHSV.rows(); ++i) {
            for (int j = 0; j < featureHSV.cols(); ++j) {
                // alter the channel representing saturation in feature HSV Mat
                // in this case the lips' red will get deeper
                double res = featureHSV.get(i, j)[BRIGHT_CHANNEL] * featureRelativeMask.get(i, j)[BRIGHT_CHANNEL] * rate;
                double[] pixel = new double[1];
                pixel[0] = res;
                HSV.put(i, j, pixel);
            }
        }

        // do some Gaussian blur to make the change smoother and more natral
        Imgproc.GaussianBlur(HSV, HSV, new Size(3, 3), 0);


        for (int i = 0; i < featureHSV.rows(); ++i) {
            for (int j = 0; j < featureHSV.cols(); ++j) {
                double res = featureHSV.get(i, j)[BRIGHT_CHANNEL] + HSV.get(i, j)[0];
                double[] pixel = featureHSV.get(i, j);
                pixel[BRIGHT_CHANNEL] = Math.min(res, MAX_BRIGHT_VAL);
                featureHSV.put(i, j, pixel);
            }
        }
        // update the change in HSV to RGB (as the display will be in RGB format)
        Imgproc.cvtColor(featureHSV, featureRGB, Imgproc.COLOR_HSV2RGB);
    }
}





