package mango.whale.makeup;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import mango.whale.dlib.Constants;
import mango.whale.dlib.FaceDet;
import mango.whale.dlib.VisionDetRet;
import mango.whale.face.Face;
import mango.whale.face.Feature;

public class ImageMakeupActivity extends AppCompatActivity {
    static private final boolean DEBUG = true;
    static private final String TAG = "mango.whale.makeup.MainActivity";
    static private final String imageRelativePath = "/Download/1.jpg";
    private FaceDet mFaceDet;

    static{
        System.loadLibrary("opencv_java3");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_makeup);
        mFaceDet = new FaceDet(Constants.getFaceShapeModelPath(), Constants.getClassifierPath());
        addMakeup2StillImage();
    }


    private void addMakeup2StillImage() {
        File externalStorageDirectory = Environment.getExternalStorageDirectory();
        /*
            1. load and decode image, get its BGR and HSV matrix
         */
        Mat imageBGR = Imgcodecs.imread(externalStorageDirectory.getAbsolutePath() + imageRelativePath);
        Mat imageGRAY = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC1);
        Mat imageRGB = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC3);
        Mat imageHSV = new Mat(imageBGR.rows(), imageBGR.cols(), CvType.CV_8UC3);
        Imgproc.cvtColor(imageBGR, imageGRAY, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(imageBGR, imageHSV, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(imageBGR, imageRGB, Imgproc.COLOR_BGR2RGB);


        /*
            2. detect face and get face landmarks
         */
        Bitmap decodedImage = Bitmap.createBitmap(imageRGB.cols(), imageRGB.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageRGB, decodedImage);
        List<VisionDetRet> visionDetRets = mFaceDet.detect(imageGRAY);

        assert visionDetRets != null;
        ArrayList<ArrayList<Point>> landmarkPoints = new ArrayList<>();
        for (VisionDetRet visionDetRet:visionDetRets) {
            landmarkPoints.add(visionDetRet.getFaceLandmarks());
        }


        /*
            3. initialize Face class
         */
        @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
        ArrayList<Face> faces = new ArrayList<>(visionDetRets.size());
        for (ArrayList<Point> landmarkPoint: landmarkPoints) {
            faces.add(new Face(imageRGB, imageHSV, landmarkPoint));
            if (DEBUG) {
                Feature mouth = faces.get(faces.size()-1).getFeature("mouth");


                mouth.brightening(3);


                Bitmap bitmap = Bitmap.createBitmap(imageRGB.cols(), imageRGB.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(imageRGB, bitmap);

                // display both the original and processed images
                ImageView imageView1 = findViewById(R.id.imageView1);
                imageView1.setImageBitmap(decodedImage);
                ImageView imageView2 = findViewById(R.id.imageView2);
                imageView2.setImageBitmap(bitmap);

            }
        }
    }
}
