package mango.whale.makeup;

import android.Manifest;
import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.media.Image;
import android.media.ImageReader;
import android.support.annotation.NonNull;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import mango.whale.dlib.Constants;
import mango.whale.dlib.FaceDet;
import mango.whale.dlib.VisionDetRet;
import mango.whale.face.Face;
import mango.whale.face.Feature;


@SuppressLint("LongLogTag")
public class RealTimeMakeupActivity extends AppCompatActivity {
    static private final boolean DEBUG = true;
    static private final String TAG = "mango.whale.makeup.RealTimeMakeupActivity";
    static private final int PERMISSION_REQUEST_CODE = 1;
    static private final int LANDMARK_POINTS_NUM = 68;
    static private final int MAX_SIMULTANEOUS_IMAGE = 35;
    static private final int IMAGE_WIDTH = 320;
    static private final int IMAGE_HEIGHT = 240;
    static private final int DISPLAY_WIDTH = 800;
    static private final int DISPLAY_HEIGHT = 600;
    static private final double BRIGHTENING_RATE = 1.8;

    private FaceDet mFaceDet;
    private List<MatOfPoint2f> landmarks;
    private SurfaceHolder mSurfaceHolder;
    private CameraManager manager;
    private String mCameraId;
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCaptureSession;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private CaptureRequest mPreviewRequest;
    private ImageReader mImageReader;
    private boolean processImage = true;
    List<VisionDetRet> visionDetRets;

    // load shared library
    static {
        System.loadLibrary("opencv_java3");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_real_time_makeup);

        // load pre-trained face detection model
        mFaceDet = new FaceDet(Constants.getFaceShapeModelPath(), Constants.getClassifierPath());
        Log.d(TAG, "successfully loaded model");
        landmarks = new ArrayList<>(LANDMARK_POINTS_NUM);


        SurfaceView mSurfaceView = findViewById(R.id.mainSurface);
        mImageReader = ImageReader.newInstance(IMAGE_WIDTH, IMAGE_HEIGHT, 0x2, MAX_SIMULTANEOUS_IMAGE);
        mSurfaceHolder = mSurfaceView.getHolder();
        mSurfaceHolder.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {}

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {}

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {}
        });

//-----------------------------------------------------------------------------------------------------------------------------------------
        manager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
        mCameraId = String.valueOf(getIntent().getIntExtra("CameraInfo", 0));

        // configure and open camera
        try {
            // get available camera list
            if (manager != null) {
                for (String cameraId : manager.getCameraIdList()) {
                    CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                    if (characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) == null) {
                        continue;
                    }
                    // TODO: for pad, use fixed camera ID for debugging, may dynamically obtain ID later
                    //mCameraId = cameraId;
                }

                // open camera
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CODE);
                    return;
                }
                manager.openCamera(mCameraId, mStateCallback, null);
                Log.d(TAG, "camera is ready");
            }
            else {
                Log.d(TAG,"camera manager is null");
            }
        } catch (CameraAccessException | NullPointerException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getApplicationContext(), "camera is opened!", Toast.LENGTH_SHORT).show();
            }
        }
        else {
            Toast.makeText(getApplicationContext(), "open camera permission denied...", Toast.LENGTH_SHORT).show();
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


    @Override
    protected void onPause() {
        super.onPause();
        Log.d("PAUSE", "pausing");
    }



    // set camera state callback functions
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            // create preview session to start the preview of processed frame
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
        }


        private void createCameraPreviewSession() {
            try {
                mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                //mPreviewRequestBuilder.addTarget(mSurfaceHolder.getSurface());
                mPreviewRequestBuilder.addTarget(mImageReader.getSurface());

                // create a capture session for preview
                //mCameraDevice.createCaptureSession(Arrays.asList(mSurfaceHolder.getSurface(), mImageReader.getSurface()),
                mCameraDevice.createCaptureSession(Collections.singletonList(mImageReader.getSurface()),
                        new CameraCaptureSession.StateCallback() {
                            @Override
                            public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                                // camera is closed
                                if (mCameraDevice == null) {
                                    return;
                                }
                                // start preview when session is set
                                mCaptureSession = cameraCaptureSession;
                                try {
                                    // auto focus
                                    mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                                    // send preview request
                                    mPreviewRequest = mPreviewRequestBuilder.build();
                                    mCaptureSession.setRepeatingRequest(mPreviewRequest,
                                            null, null);
                                    Log.d(TAG,"open camera and start preview");
                                } catch (CameraAccessException e) {
                                    e.printStackTrace();
                                }
                            }

                            @Override
                            public void onConfigureFailed(
                                    @NonNull CameraCaptureSession cameraCaptureSession) {
                                Log.e(TAG,"onConfigureFailed: fail to start preview");
                            }
                        }, null);

//-------------------------------------------------------------------------------------------------------------------------------
                // callback function when image data is ready from the camera
                ImageReader.OnImageAvailableListener mOnImageAvailableListener
                        = new ImageReader.OnImageAvailableListener() {
                    // process available frame, i.e. add makeup and display the processed frame on preview surface
                    @Override
                    public void onImageAvailable(ImageReader reader) {
                        try {
                            Image image = reader.acquireLatestImage();
                            //Image image = reader.acquireNextImage();
                            Bitmap bitmap = addMakeup(image);
                            Canvas canvas = mSurfaceHolder.lockCanvas();
                            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                            canvas.drawBitmap(bitmap, null, new Rect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), null);
                            mSurfaceHolder.unlockCanvasAndPost(canvas);
                            image.close();
                        }
                        catch (Exception ignored) {}
                    }
                };
//-------------------------------------------------------------------------------------------------------------------------------

                mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, null);
            } catch (CameraAccessException e) {
                Log.e(TAG,"CameraAccessException: fail to start preview");
                e.printStackTrace();
            }
        }
    };



    @SuppressWarnings("NumericOverflow")
    public Bitmap convert2GrayImage(Bitmap img) {
        int width = img.getWidth();
        int height = img.getHeight();

        int []pixels = new int[width * height];

        img.getPixels(pixels, 0, width, 0, 0, width, height);
        int alpha = 0xFF << 24;
        for(int i = 0; i < height; i++)  {
            for(int j = 0; j < width; j++) {
                int grey = pixels[width * i + j];

                int red = ((grey  & 0x00FF0000 ) >> 16);
                int green = ((grey & 0x0000FF00) >> 8);
                int blue = (grey & 0x000000FF);

                grey = (int)((float) red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
                grey = alpha | (grey << 16) | (grey << 8) | grey;
                pixels[width * i + j] = grey;
            }
        }
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        result.setPixels(pixels, 0, width, 0, 0, width, height);
        return result;
    }



    public Bitmap addMakeup(Image image) {
        /*
            1. load and decode image, get its BGR and HSV matrix
        */
        // convert Image object to Bitmap object
        int width = image.getWidth();
        int height = image.getHeight();
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * width;
        Bitmap bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);

        Mat imageBGR = new Mat(bitmap.getWidth(), bitmap.getHeight(), CvType.CV_8UC3);
        Utils.bitmapToMat(bitmap, imageBGR);
        Mat imageGRAY = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC1);
        Mat imageRGB = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC3);
        Mat imageHSV = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC3);
        Imgproc.cvtColor(imageBGR, imageGRAY, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(imageBGR, imageHSV, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(imageBGR, imageRGB, Imgproc.COLOR_BGR2RGB);


        /*
            2. detect face and get face landmarks
         */
        // use grey scale image to increase the efficiency (minor though...)
//---------------------------------------------------------------------------------------------------------------------------------------------------
        // TODO: this step alone may take up to 210 ms, try to find a way to optimize the use...
        // TODO: maybe try to use openCV directly with classifier and Facemark from opencv_contrib
        // TODO: could be a compromise between accuracy and efficiency
        // TODO: for now, use grey scale input and smaller resolution may accelerate the processing speed
        // TODO: if ArFace API provides a better face detection result (a box encompasses the face)
        // TODO: one possible solution would be to use the result as the input of getting face landmarks
        // TODO: however, this requires change in JNI .cpp file (jni_face_det.cpp, check out function getDetectResult())
        // TODO: improving the efficiency requires a significant amount of work in the future
        // TODO: while the sample face feature extraction has been implemented, other operations like
        // TODO: whitening and smoothing are relatively easy to implement
        // TODO: and there are many resources online using a different language (more likely in C++ or Python) to carry out a same implementation
        visionDetRets = mFaceDet.detect(imageGRAY);
//---------------------------------------------------------------------------------------------------------------------------------------------------

        assert visionDetRets != null;
        ArrayList<ArrayList<Point>> landmarkPoints = new ArrayList<>();
        for (VisionDetRet visionDetRet:visionDetRets) {
            landmarkPoints.add(visionDetRet.getFaceLandmarks());
        }
        /*
            3. initialize Face and Feature class, add makeup
         */

        @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
        ArrayList<Face> faces = new ArrayList<>(visionDetRets.size());
        for (ArrayList<Point> landmarkPoint: landmarkPoints) {
            faces.add(new Face(imageRGB, imageHSV, landmarkPoint));
            Feature mouth = faces.get(faces.size()-1).getFeature("mouth");
            // TODO: may need to add more makeup options, like whitening, smoothing and so forth
            mouth.brightening(BRIGHTENING_RATE);
        }

        Imgproc.cvtColor(imageRGB, imageBGR, Imgproc.COLOR_RGB2BGR);
        Bitmap result = Bitmap.createBitmap(imageBGR.cols(), imageBGR.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageBGR, result);

        return result;
    }
}
