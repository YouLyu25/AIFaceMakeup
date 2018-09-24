package mango.whale.makeup;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;


/*
    Note:
    1.  download openCV and configure it to correct path
    2.  change .cpp files according to project's package name:
        line 15, 18, 19 in jni_primitives.h and line 88 in jni_face_det.cpp
    3.  download NDK and use it to rebuild libraries
    4.  add libs (change build.gradle to map jniLibs and libs)
    5.  add FaceDet.java, VisionDet.java and Constants.java to proper directory
    6.  add pre-trained landmark data to pad and change directory in Constants.java
    7.  add openCV as module and update module dependencies of app
    8.  add shared libs of openCV to libs of app
    9.  use System.loadLibrary before calling openCV java class
*/

public class MainActivity extends AppCompatActivity {
    static private final String TAG = "mango.whale.makeup.MainActivity";
    static private final int BACK_CAMERA = 0;
    static private final int FRONT_CAMERA = 1;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void goToRealTimeMakeupActivity(View view) {
        Intent intent = new Intent(this, RealTimeMakeupActivity.class);
        intent.putExtra("CameraInfo", BACK_CAMERA);
        startActivity(intent);
    }


    public void goToImageMakeupActivity(View view) {
        Intent intent = new Intent(this, ImageMakeupActivity.class);
        startActivity(intent);
    }
}
