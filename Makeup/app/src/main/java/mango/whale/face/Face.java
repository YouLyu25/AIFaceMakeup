package mango.whale.face;

import android.graphics.Point;
import android.util.Log;

import org.opencv.core.Mat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by You Lyu on 2018/8/6.
 * Implementations of Face class
 */
@SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
public class Face {
    private static final int JAW_RANGE_LOW = 0;
    private static final int JAW_RANGE_UP = 17;
    private static final int MOUTH_RANGE_LOW = 48;
    private static final int MOUTH_RANGE_UP = 61;
    private static final int NOSE_RANGE_LOW = 27;
    private static final int NOSE_RANGE_UP = 35;
    private static final int LEFT_EYE_RANGE_LOW = 42;
    private static final int LEFT_EYE_RANGE_UP = 48;
    private static final int RIGHT_EYE_RANGE_LOW = 36;
    private static final int RIGHT_EYE_RANGE_UP = 42;
    private static final int LEFT_BROW_RANGE_LOW = 22;
    private static final int LEFT_BROW_RANGE_UP = 27;
    private static final int RIGHT_BROW_RANGE_LOW = 17;
    private static final int RIGHT_BROW_RANGE_UP = 22;
    private Map<String, Feature> features = new HashMap<>();

    // constructor
    @SuppressWarnings("SuspiciousSystemArraycopy")
    public Face(Mat imageRBG, Mat imageHSV, ArrayList<Point> landmarkPoints) {
        ArrayList<int[]> featureLandmarksRange = new ArrayList<>();
        featureLandmarksRange.add(new int[]{JAW_RANGE_LOW, JAW_RANGE_UP});
        featureLandmarksRange.add(new int[]{MOUTH_RANGE_LOW, MOUTH_RANGE_UP});
        featureLandmarksRange.add(new int[]{NOSE_RANGE_LOW, NOSE_RANGE_UP});
        featureLandmarksRange.add(new int[]{LEFT_EYE_RANGE_LOW, LEFT_EYE_RANGE_UP});
        featureLandmarksRange.add(new int[]{RIGHT_EYE_RANGE_LOW, RIGHT_EYE_RANGE_UP});
        featureLandmarksRange.add(new int[]{LEFT_BROW_RANGE_LOW, LEFT_BROW_RANGE_UP});
        featureLandmarksRange.add(new int[]{RIGHT_BROW_RANGE_LOW, RIGHT_BROW_RANGE_UP});

        ArrayList<String> featureNames = new ArrayList<>();
        featureNames.add("jaw");
        featureNames.add("mouth");
        featureNames.add("nose");
        featureNames.add("leftEye");
        featureNames.add("rightEye");
        featureNames.add("leftEyeBrow");
        featureNames.add("rightEyeBrow");

        // for each feature, get its landmark points and initiate a new Feature class
        for (int i = 0; i < featureLandmarksRange.size(); ++i) {
            int low = featureLandmarksRange.get(i)[0];
            int up = featureLandmarksRange.get(i)[1];
            ArrayList<Point> featureLandmarks = new ArrayList<>();
            featureLandmarks.addAll(landmarkPoints.subList(low, up));
            features.put(featureNames.get(i), new Feature(imageRBG, imageHSV, featureLandmarks));
        }
    }


    public Feature getFeature(String featureName) {
        return features.get(featureName);
    }
}
