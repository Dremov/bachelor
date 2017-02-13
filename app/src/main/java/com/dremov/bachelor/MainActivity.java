package com.dremov.bachelor;

import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import com.dremov.bachelor.util.Config;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();

    private Mat bufferImg;
    private Mat mask;

    private BackgroundSubtractorMOG2 bgMOG;// = Video.createBackgroundSubtractorMOG2();

    private CameraBridgeViewBase mOpenCvCameraView;
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;

    //    private Camera mCamera;
    private CameraManager mCameraManager;
    private CameraPreview mPreview;

    List<MatOfPoint> squares = new ArrayList<MatOfPoint>();
    int thresh = 50, N = 11;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.open_cv_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(640, 480);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    bgMOG = Video.createBackgroundSubtractorMOG2();
                    mask = new Mat();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };



    @Override
    protected void onResume() {
        super.onResume();

        openBackgroundThread();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);

        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

            bgMOG = Video.createBackgroundSubtractorMOG2();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        if (bufferImg == null) {
            bufferImg = inputFrame.gray();
        }

        final Mat frame = inputFrame.gray();

        Thread processingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                drawContours(frame);

            }
        });
        processingThread.setName("Processing Thread");
        if(getThreadByName("Processing Thread") == null) {
            processingThread.start();
        }

        bgMOG.apply(frame, mask, 0.8);

        return mask;
    }

    @Override
    public void onPause() {

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        closeBackgroundThread();
        super.onPause();
    }

    private void openBackgroundThread() {
        mBackgroundThread = new HandlerThread("Image processing background thread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void closeBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private Thread getThreadByName(String threadName)
    {
        Thread __tmp = null;
        Set<Thread> threadSet = Thread.getAllStackTraces().keySet();
        Thread[] threadArray = threadSet.toArray(new Thread[threadSet.size()]);
        for (int i = 0; i < threadArray.length; i++)
            if (threadArray[i].getName().equals(threadName))
                __tmp =  threadArray[i];
        return __tmp;
    }

    private void drawContours(Mat frame) {

        Imgproc.findContours(mask, squares, new Mat(), Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        int maxAreaIdx = -1;
        Rect r = null;
        Vector<Rect> rect_array = new Vector<Rect>();

        for (int idx = 0; idx < squares.size(); idx++) {
            Mat contour = squares.get(idx);
            double contourarea = Imgproc.contourArea(contour);
            if (contourarea > Config.MIN_BLOB_AREA && contourarea < Config.MAX_BLOB_AREA) {
                // MIN_BLOB_AREA = contourarea;
                maxAreaIdx = idx;
                r = Imgproc.boundingRect(squares.get(maxAreaIdx));
                rect_array.add(r);
//                 Imgproc.drawContours(mask, squares, maxAreaIdx, new
//                 Scalar(255, 255, 255));
            }

        }

        for (int i=0; i<squares.size() ; i++) {
            Imgproc.drawContours(frame, squares, i, new Scalar(255, 0, 0), 1);
        }
    }

    public static Vector<Rect> detectionContours(Mat outmat) {
        Mat v = new Mat();
        Mat vv = outmat.clone();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(vv, contours, v, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        int maxAreaIdx = -1;
        Rect r = null;
        Vector<Rect> rect_array = new Vector<Rect>();

        for (int idx = 0; idx < contours.size(); idx++) {
            Mat contour = contours.get(idx);
            double contourarea = Imgproc.contourArea(contour);
            if (contourarea > Config.MIN_BLOB_AREA && contourarea < Config.MAX_BLOB_AREA) {
                // MIN_BLOB_AREA = contourarea;
                maxAreaIdx = idx;
                r = Imgproc.boundingRect(contours.get(maxAreaIdx));
                rect_array.add(r);
                // Imgproc.drawContours(imag, contours, maxAreaIdx, new
                // Scalar(255, 255, 255));
            }

        }

        v.release();
        return rect_array;
    }
}
