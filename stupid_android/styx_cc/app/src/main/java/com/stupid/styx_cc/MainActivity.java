package com.stupid.styx_cc;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.TextView;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class MainActivity extends AppCompatActivity {

    class Tensor {
        int[] shape;
        float[] data;

        @Override
        public String toString() {
            String output = "";
            output += "shape: " + Arrays.toString(shape);
            output += " data: " + Arrays.toString(data);
            return output;
        }
    }

    // Read SVD model from assets and write it to a temp file because C++ cannot access the assets.
    File getSvdModelFile() {
        File downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File model_file = new File(downloads, "svd.tflite");
        if (!model_file.exists()) {
            throw new RuntimeException("No such file: " + model_file.getAbsolutePath());
        }
        return model_file;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        File modelPath = getSvdModelFile();
        String init_message = initSvd(modelPath.getAbsolutePath());

        String message;
        if (init_message.isEmpty()) {
            Tensor input = new Tensor();
            input.data = new float[]{1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f};
            input.shape = new int[]{1, 3, 3};
            Tensor s = new Tensor();
            Tensor u = new Tensor();
            message = runSvd(input, s, u);
            if (message.isEmpty()) {
                message += "input: " + input.toString() + "\n";
                message += "s: " + s.toString() + "\n";
                message += "u: " + u.toString() + "\n";
            }
        } else {
            message = "Init failed: " + init_message;
        }

        TextView tv = new TextView(this);
        tv.setText(message);
        setContentView(tv);
    }

    public native String initSvd(String model_path);

    public native String runSvd(Tensor input, Tensor s, Tensor u);

    static {
        System.loadLibrary("register-svd");
    }


}
