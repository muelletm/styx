package com.stupid.styx_cc;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    final static int PermRequestCode = 1;

    TextView tv_;
    ImageView image_;

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
        File model_file = new File(downloads, "fake.tflite");
        if (!model_file.exists()) {
            throw new RuntimeException("No such file: " + model_file.getAbsolutePath());
        }
        return model_file;
    }

    Tensor LoadResourceImageAsTensor(int resource_index) {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.gilbert);

        // TODO(thomas) Remove this temporary hack.
        int new_size = Math.max(bitmap.getWidth(), bitmap.getHeight());
        if (new_size > 64) {
            new_size = 64;
        }
        bitmap = Bitmap.createScaledBitmap(bitmap, new_size,new_size, false);

        int size = bitmap.getHeight() * bitmap.getWidth() * 3;
        Tensor tensor = new Tensor();
        tensor.shape = new int[]{1, bitmap.getHeight(), bitmap.getWidth(), 3};
        tensor.data = new float[size];
        int index = 0;
        for (int x = 0; x < bitmap.getWidth(); ++x) {
            for (int y = 0; y < bitmap.getHeight(); ++y) {
                int pixel = bitmap.getPixel(x, y);

                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = pixel & 0xFF;

                if (y == 0 && x == 0) {
                    Log.i("pixel", Integer.toBinaryString(pixel));
                    Log.i("pixel", Integer.toBinaryString(red));
                    Log.i("pixel", Integer.toBinaryString(green));
                    Log.i("pixel", Integer.toBinaryString(blue));
                }

                assert red >= 0 && red <= 255;
                assert green >= 0 && green <= 255;
                assert blue >= 0 && blue <= 255;

                tensor.data[index] = red / 255.0f;
                index++;
                tensor.data[index] = green / 255.0f;
                index++;
                tensor.data[index] = blue / 255.0f;
                index++;
            }
        }
        return tensor;
    }

    Bitmap TensorToBitmap(Tensor tensor, boolean red_eye) {
        int height = tensor.shape[1];
        int width = tensor.shape[2];

        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap image = Bitmap.createBitmap(width, height, conf);

        int index = 0;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                int red = (int) (tensor.data[index] * 255.);
                index++;
                int green = (int) (tensor.data[index] * 255.);
                index++;
                int blue = (int) (tensor.data[index] * 255.);
                if (red_eye) {
                    green = 0;
                    blue = 0;
                }

                index++;
                int color = Color.rgb(
                        red,
                        green,
                        blue
                );
                image.setPixel(x, y, color);
            }
        }

        int current_size = Math.max(image.getWidth(), image.getHeight());
        if (current_size < 512) {
            float scale_factor = 512 / current_size;
            int new_width = (int)(scale_factor * image.getWidth());
            int new_height = (int)(scale_factor * image.getHeight());
            image = Bitmap.createScaledBitmap(image, new_width, new_height, true);
        }
        return image;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        assert requestCode == PermRequestCode;
        assert permissions.length == 1;
        assert grantResults.length == 1;
        assert permissions[0].equals(Manifest.permission.READ_EXTERNAL_STORAGE);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            runModel();
        } else {
            tv_.setText("Permission denied.");
        }
    }

    void runModel() {
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            tv_.setText("Permission denied.");
            return;
        }
        File modelPath = getSvdModelFile();
        String init_message = initSvd(modelPath.getAbsolutePath());
        if (!init_message.isEmpty()) {
            tv_.setText("Init failed: " + init_message);
            return;
        }

        Tensor content = LoadResourceImageAsTensor(R.drawable.gilbert);
        Tensor style = LoadResourceImageAsTensor(R.drawable.style5);
        Tensor result = new Tensor();
        String error = runStyleTransfer(content, style, result);
        if (!error.isEmpty()) {
            tv_.setText("Transfer failed: " + error);
            return;
        }
        image_.setImageBitmap(TensorToBitmap(result, false));
        tv_.setText("Success");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        image_ = findViewById(R.id.imageView);
        tv_ = findViewById(R.id.textView);

        image_.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                image_.setImageBitmap(
                        TensorToBitmap(LoadResourceImageAsTensor(R.drawable.gilbert), true));
                runModel();
            }
        });

        image_.setImageBitmap(
                TensorToBitmap(LoadResourceImageAsTensor(R.drawable.gilbert), false));

        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PermRequestCode);
            tv_.setText("Waiting for permissions");
        }
    }

    public native String initSvd(String model_path);

    public native String runStyleTransfer(Tensor content, Tensor style, Tensor result);

    static {
        System.loadLibrary("register-svd");
    }


}
