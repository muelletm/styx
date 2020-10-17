package com.stupid.customops;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.Buffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    static String CallTF(File modelFile) {
        Interpreter interpreter = new Interpreter(modelFile);

        assert interpreter.getInputTensorCount() == 1;
        Tensor input = interpreter.getInputTensor(0);
        assert input.numDimensions() == 3;

        interpreter.resizeInput(0, new int[]{1 ,3, 3}, true);

        interpreter.allocateTensors();

        Object[] inputs =  {new float[]{0, 0, 1, 0, 1, 0, 1, 0, 0 }};


        Map outputs = new HashMap<>();
        outputs.put(0, FloatBuffer.allocate(3));
        outputs.put(1, FloatBuffer.allocate(9));
        outputs.put(2, FloatBuffer.allocate(9));

        interpreter.runForMultipleInputsOutputs(inputs, outputs);
        interpreter.close();

        String output_message = outputs.toString();
        return output_message;
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
            message = runSvd();
        } else {
            message = "Init failed: " + init_message;
        }

        TextView tv = new TextView(this);
        tv.setText(message);
        setContentView(tv);
    }

    public native String initSvd(String model_path);
    public native String runSvd();

    static {
        System.loadLibrary("register-svd");
    }



}
