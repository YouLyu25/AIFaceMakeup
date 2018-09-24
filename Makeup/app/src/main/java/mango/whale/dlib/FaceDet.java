package mango.whale.dlib;

import android.graphics.Bitmap;
import android.support.annotation.Keep;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.annotation.WorkerThread;
import android.util.Log;

import org.opencv.core.Mat;

import java.util.Arrays;
import java.util.List;

/**
 * Created by houzhi on 16-10-20.
 * Modified by tzutalin on 16-11-15
 */
public class FaceDet {
    private static final String TAG = "dlib";

    // accessed by native methods
    @SuppressWarnings("unused")
    private long mNativeFaceDetContext;
    private String mLandMarkPath = "";
    private String mClassifierPath = "";

    static {
        try {
            Log.d(TAG, "loading library");
            System.loadLibrary("android_dlib");
            Log.d(TAG, "successfully loaded library");
            jniNativeClassInit();
            Log.d(TAG, "jniNativeClassInit success");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "library not found, " + e);
        }
    }

    @SuppressWarnings("unused")
    public FaceDet() {
        jniInit(mLandMarkPath, mClassifierPath);
    }

    public FaceDet(String landMarkPath, String classifierPath) {
        mLandMarkPath = landMarkPath;
        mClassifierPath = classifierPath;
        jniInit(mLandMarkPath, mClassifierPath);
    }

    @Nullable
    @WorkerThread
    public List<VisionDetRet> detect(@NonNull String path) {
        VisionDetRet[] detRets = jniDetect(path);
        return Arrays.asList(detRets);
    }

    /*
    @Nullable
    @WorkerThread
    public List<VisionDetRet> detect(@NonNull Bitmap bitmap) {
        VisionDetRet[] detRets = jniBitmapDetect(bitmap);
        return Arrays.asList(detRets);
    }
    */

    @Nullable
    @WorkerThread
    public List<VisionDetRet> detect(@NonNull Mat BGRMat) {
        VisionDetRet[] detRets = jniGRAYMatDetect(BGRMat.getNativeObjAddr());
        return Arrays.asList(detRets);
    }


    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        release();
    }

    public void release() {
        jniDeInit();
    }

    @Keep
    private native static void jniNativeClassInit();

    @Keep
    private synchronized native int jniInit(String landmarkModelPath, String classifierPath);

    @Keep
    private synchronized native int jniDeInit();

    @Keep
    private synchronized native VisionDetRet[] jniBitmapDetect(Bitmap bitmap);

    @Keep
    private synchronized native VisionDetRet[] jniGRAYMatDetect(long mNativeFaceDetContext);

    @Keep
    private synchronized native VisionDetRet[] jniGetLandmarks(long mNativeFaceDetContext);

    @Keep
    private synchronized native VisionDetRet[] jniDetect(String path);
}