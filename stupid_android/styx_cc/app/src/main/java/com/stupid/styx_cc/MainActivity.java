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

    private final static int PERM_REQUEST_CODE_ = 1;
    private final static int MAX_IMAGE_SIZE_ = 512;
    // fake.tflite, fake_svd.tflite, stupid_relu4.tflite
    private final static String MODEL_NAME_ = "stupid_relu4.tflite";

    private TextView tv_;
    private ImageView image_;

    private class Timer {
        public Timer() {
            time_ = System.currentTimeMillis();
        }

        public long getTimeDelta() {
            return System.currentTimeMillis() - time_;
        }

        private long time_;
    }

    private class Tensor {
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

    private File getModelFile() {
        File downloads = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS);
        File model_file = new File(downloads, MODEL_NAME_);
        if (!model_file.exists()) {
            throw new RuntimeException("No such file: " + model_file.getAbsolutePath());
        }
        return model_file;
    }

    private Tensor LoadResourceImageAsTensor(int resource_index) {
        Timer timer = new Timer();
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), resource_index);

        // TODO(thomas) Remove this temporary hack.
        int new_size = Math.min(Math.min(bitmap.getWidth(), bitmap.getHeight()), MAX_IMAGE_SIZE_);
        bitmap = Bitmap.createScaledBitmap(bitmap, new_size, new_size, false);

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
        Log.i("LoadResourceImage", "Took " + timer.getTimeDelta() + "ms");
        return tensor;
    }

    private Bitmap TensorToBitmap(Tensor tensor, boolean red_eye) {
        Timer timer = new Timer();
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


        int new_height = image_.getDrawable().getIntrinsicHeight();
        int new_width = image_.getDrawable().getIntrinsicWidth();
        image = Bitmap.createScaledBitmap(image, new_width, new_height, true);
        Log.i("LoadResourceImage", "Took " + timer.getTimeDelta() + "ms");
        return image;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        assert requestCode == PERM_REQUEST_CODE_;
        assert permissions.length == 1;
        assert grantResults.length == 1;
        assert permissions[0].equals(Manifest.permission.READ_EXTERNAL_STORAGE);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            runModel();
        } else {
            tv_.setText("Permission denied.");
        }
    }

    private String runModel() {
        Timer timer = new Timer();
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            return "Permission denied.";
        }

        try {
            File modelPath = getModelFile();
            String init_message = prepareInterpreter(modelPath.getAbsolutePath());
            if (!init_message.isEmpty()) {
                return "Init failed: " + init_message;
            }
        } catch (RuntimeException e) {
            return e.getMessage();
        }

        Tensor content = LoadResourceImageAsTensor(R.drawable.gilbert);
        Tensor style = LoadResourceImageAsTensor(R.drawable.style5);
        Tensor result = new Tensor();
        String error = runStyleTransfer(content, style, result);
        if (!error.isEmpty()) {
            return "Transfer failed: " + error;
        }
        image_.setImageBitmap(TensorToBitmap(result, false));
        tv_.setText("Success (" + (timer.getTimeDelta()) + " milliseconds)");
        return "";
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
                String error = runModel();
                if (!error.isEmpty()) {
                    tv_.setText(error);
                    image_.setImageBitmap(
                            TensorToBitmap(LoadResourceImageAsTensor(R.drawable.gilbert), true));
                }
            }
        });

        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERM_REQUEST_CODE_);
            tv_.setText("Waiting for permissions");
        }
    }

    private native String prepareInterpreter(String model_path);

    private native String runStyleTransfer(Tensor content, Tensor style, Tensor result);

    static {
        System.loadLibrary("register-svd");
    }
}
