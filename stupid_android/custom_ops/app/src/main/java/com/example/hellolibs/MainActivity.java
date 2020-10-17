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
import android.widget.TextView;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.Buffer;

/*
 * Simple Java UI to trigger jni function. It is exactly same as Java code
 * in hello-jni.
 */
public class MainActivity extends AppCompatActivity {

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
            tv.setText( stringFromJNI(outputFile.getAbsolutePath()));
        } catch (IOException e) {throw new RuntimeException(e);}
        setContentView(tv);
    }

    public native String  stringFromJNI(String filePath);
    static {
        System.loadLibrary("hello-libs");
    }

}
