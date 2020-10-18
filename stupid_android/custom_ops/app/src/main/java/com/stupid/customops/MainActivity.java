package com.stupid.customops;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

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
        try {
            InputStream is = getAssets().open("svd.tflite");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            File outputDir = getApplicationContext().getCacheDir();
            File outputFile = File.createTempFile("svd", "tflite", outputDir);
            FileOutputStream stream = new FileOutputStream(outputFile);
            stream.write(buffer);
            stream.close();
            return outputFile;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
