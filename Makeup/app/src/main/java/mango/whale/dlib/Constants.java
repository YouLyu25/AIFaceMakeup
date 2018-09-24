package mango.whale.dlib;

import android.os.Environment;
import android.util.Log;

import java.io.File;

/**
 * Created by Darrenl on 2016/4/22.
 *
 */
public final class Constants {
    private Constants() {
        // Constants should be prive
    }

    /**
     * getFaceShapeModelPath
     * @return default face shape model path
     */
    public static String getFaceShapeModelPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        // TODO: for debugging purpose, return a relative path of pre-trained model data, may be changed later
        Log.d("DEBUG", sdcard.getAbsolutePath());
        return sdcard.getAbsolutePath() + "/Download/shape_predictor_68_face_landmarks.dat";
        //return "/sdcard/Download/shape_predictor_68_face_landmarks.dat"; // this path can be used when using emulator
    }


    // added by You Lyu in 2018/08/10
    public static String getClassifierPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        // TODO: for debugging purpose, return a relative path of pre-trained model data, may be changed later
        Log.d("DEBUG", sdcard.getAbsolutePath());
        return sdcard.getAbsolutePath() + "/Download/koestinger_cascade_aflw_lbp.xml";
    }
}
