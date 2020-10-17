/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.hellolibs;

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

/*
 * Simple Java UI to trigger jni function. It is exactly same as Java code
 * in hello-jni.
 */
public class MainActivity extends AppCompatActivity {

    static String CallTF(File modelFile) {
        Interpreter interpreter = new Interpreter(modelFile);

        interpreter.allocateTensors();

        Log.i("Input tensor count: ", Integer.toString(interpreter.getInputTensorCount()));

        if (interpreter.getInputTensorCount() == 0) {
            return "No input";
        }

        Tensor input_tensor = interpreter.getInputTensor(0);

        Object[] inputs = {0};
        Map map_of_indices_to_outputs = new HashMap<>();
        FloatBuffer output = FloatBuffer.allocate(5);
        map_of_indices_to_outputs.put(0, output);

        interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);

        String output_message = output.array().toString();

        interpreter.close();

        return output_message;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);

        try {
            InputStream is = getAssets().open("sin.tflite");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();

            File outputDir = getApplicationContext().getCacheDir();
            File outputFile = File.createTempFile("sin", "tflite", outputDir);

            try {
                FileOutputStream stream = new FileOutputStream(outputFile);
                stream.write(buffer);
                stream.close();
            }
            catch (java.io.FileNotFoundException e) {
                throw new RuntimeException(e);
            }
            String c_message = stringFromJNI(outputFile.getAbsolutePath());

            String j_message = CallTF(outputFile);

            tv.setText(c_message + "\n" + j_message);

        } catch (IOException e) {throw new RuntimeException(e);}

        setContentView(tv);
    }

    public native String  stringFromJNI(String filePath);
    static {
        System.loadLibrary("hello-libs");
    }

}
