package tw.geosense.treedetection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.ContentFrameLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.Kwarg;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

/*
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;*/

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.os.Build.VERSION.SDK_INT;
/*import static org.opencv.core.Core.BORDER_CONSTANT;
import static org.opencv.core.Core.copyMakeBorder;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;*/

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private final int READWRITE_EXTERNAL_STORAGE = 1;
    private final int INSTALL_PACKAGES = 2;
    private final int ACCESS_NETWORK_STATE = 3;

    private final int PERMISSION_REQUEST_CODE = 100;

    private AlertDialog dialog;
    Handler handler = new Handler();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TextView tv = findViewById(R.id.sample);
        /*if (OpenCVLoader.initDebug()) {
            Log.i("CV", "load OpenCV Library Successful.");
            tv.setText("load OpenCV Library successfully.");
        } else {
            Log.i("CV", "load OpenCV Library Failed.");
            tv.setText("load OpenCV Library failed.");;
        }*/

        doInit();
    }

    private void doInit()
    {
        checkPermission();
        requestPermission();
        initPython();
        chkDBtoSDCard();
        setLayout();
    }

    private void setLayout()
    {
        ((Button)findViewById(R.id.identify)).setOnClickListener(this);
    }

    private void initPython(){
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
    }

    private void chkDBtoSDCard()
    {
        String tSDCardPath = android.os.Environment.getExternalStorageDirectory().getAbsolutePath();
        File rootPath = new File(tSDCardPath+ File.separator +config.appDir + File.separator);

        File file = rootPath;
        if (!file.exists())
            file.mkdir();

        loadImage(rootPath);
    }

    private void loadImage(File rootPath) {
        //File imageFile = new File(rootPath + File.separator + "test.jpg");

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);

        ImageView imgOrg = findViewById(R.id.imgOrg);
        imgOrg.setImageBitmap(bitmap);


        File file = new File(rootPath, "test.jpg");

        FileOutputStream fOut = null;
        try {
            fOut = new FileOutputStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
        try {
            fOut.flush();
            fOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void doOpenCV(String fileName, boolean tf)
    {
        setProgressDialog();

        new Thread(new Runnable()
        {
            public void run()
            {
                String tSDCardPath = android.os.Environment.getExternalStorageDirectory().getAbsolutePath();
                File rootPath = new File(tSDCardPath+ File.separator +config.appDir + File.separator);

                Python py = Python.getInstance();
                PyObject obj = py.getModule("treeDetection").callAttr("doRemoveBG", rootPath.getAbsolutePath(), fileName);

                Integer result = obj.toJava(Integer.class);
                if (result == 0) {
                    File imageFile = new File(rootPath + File.separator + fileName + "_no_bg.png");

                    Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());

                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            ImageView imgAfter = findViewById(R.id.imgAfter);
                            imgAfter.setImageBitmap(bitmap);
                            LinearLayout.LayoutParams param = new LinearLayout.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    1.0f
                            );
                            imgAfter.setLayoutParams(param);
                            dialog.dismiss();
                        }
                    });
                }
            }
        }).start();
    }

    /*
    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }*/

    public void setProgressDialog() {

        int llPadding = 30;
        LinearLayout ll = new LinearLayout(this);
        ll.setOrientation(LinearLayout.HORIZONTAL);
        ll.setPadding(llPadding, llPadding, llPadding, llPadding);
        ll.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams llParam = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        llParam.gravity = Gravity.CENTER;
        ll.setLayoutParams(llParam);

        ProgressBar progressBar = new ProgressBar(this);
        progressBar.setIndeterminate(true);
        progressBar.setPadding(0, 0, llPadding, 0);
        progressBar.setLayoutParams(llParam);

        llParam = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        llParam.gravity = Gravity.CENTER;
        TextView tvText = new TextView(this);
        tvText.setText(getString(R.string.loaadingIdentify));
        tvText.setTextColor(Color.parseColor("#000000"));
        tvText.setTextSize(20);
        tvText.setLayoutParams(llParam);

        ll.addView(progressBar);
        ll.addView(tvText);

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setCancelable(true);
        builder.setView(ll);

        dialog = builder.create();
        dialog.show();
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams();
            layoutParams.copyFrom(dialog.getWindow().getAttributes());
            layoutParams.width = LinearLayout.LayoutParams.WRAP_CONTENT;
            layoutParams.height = LinearLayout.LayoutParams.WRAP_CONTENT;
            dialog.getWindow().setAttributes(layoutParams);
        }
    }

    private boolean checkPermission() {
        if (SDK_INT >= Build.VERSION_CODES.R) {
            return Environment.isExternalStorageManager();
        } else {
            int result = ContextCompat.checkSelfPermission(MainActivity.this, READ_EXTERNAL_STORAGE);
            int result1 = ContextCompat.checkSelfPermission(MainActivity.this, WRITE_EXTERNAL_STORAGE);
            return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED;
        }
    }

    private void requestPermission() {
        if (SDK_INT >= Build.VERSION_CODES.R) {
            try {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.addCategory("android.intent.category.DEFAULT");
                intent.setData(Uri.parse(String.format("package:%s",getApplicationContext().getPackageName())));
                startActivityForResult(intent, 2296);
            } catch (Exception e) {
                Intent intent = new Intent();
                intent.setAction(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivityForResult(intent, 2296);
            }
        } else {
            //below android 11
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        }
    }

    public void onClick(View v) {
        switch(v.getId())
        {
            case R.id.identify:
                doOpenCV("test.jpg", true);
                break;
            default:
                break;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 2296) {
            if (SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    // perform action when allow permission success
                } else {
                    Toast.makeText(this, "Allow permission for storage access!", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0) {
                    boolean READ_EXTERNAL_STORAGE = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean WRITE_EXTERNAL_STORAGE = grantResults[1] == PackageManager.PERMISSION_GRANTED;

                    if (READ_EXTERNAL_STORAGE && WRITE_EXTERNAL_STORAGE) {
                        // perform action when allow permission success
                    } else {
                        Toast.makeText(this, "Allow permission for storage access!", Toast.LENGTH_SHORT).show();
                    }
                }
                break;
        }
    }
}