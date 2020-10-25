package com.stupid.styx_cc;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

public class Model {

    private final static int MIN_IMAGE_SIZE_ = 32;

    static {
        System.loadLibrary("register-svd");
    }

    public String name;
    public ModelConfig preview_;
    public ModelConfig full_;

    public Model(String name, ModelConfig preview, ModelConfig full) {
        this.name = name;
        this.preview_ = preview;
        this.full_ = full;
    }

    public String Init(String model_path) {
        return prepareInterpreter(model_path);
    }

    public Bitmap Run(
            boolean preview, Bitmap content_map, Bitmap style_map) throws ExecutionError {
        ModelConfig config = getModelConfig(preview);
        Tensor content = LoadResourceImageAsTensor(
                config.max_image_size,
                content_map);
        Tensor style = LoadResourceImageAsTensor(
                config.max_image_size,
                style_map);
        final Tensor result = new Tensor();
        String error = runStyleTransfer(config.svd_rank, content, style, result);
        if (!error.isEmpty()) {
            throw new ExecutionError(error);
        }
        return TensorToBitmap(result);
    }

    ModelConfig getModelConfig(boolean preview) {
        return (preview) ? preview_ : full_;
    }

    private Tensor LoadResourceImageAsTensor(int max_image_size,
                                             Bitmap bitmap) {
        Log.i("main", "LoadResourceImageAsTensor");
        Timer timer = new Timer();
        int max_size = Math.max(bitmap.getWidth(), bitmap.getHeight());
        int new_size = Math.max(Math.min(max_size, max_image_size), MIN_IMAGE_SIZE_);

        double scale_factor = (double) new_size / (double) max_size;

        int new_height = (int) (bitmap.getHeight() * scale_factor);
        int new_width = (int) (bitmap.getWidth() * scale_factor);

        bitmap = Bitmap.createScaledBitmap(bitmap, new_width, new_height, false);

        int size = bitmap.getHeight() * bitmap.getWidth() * 3;
        Tensor tensor = new Tensor();
        tensor.shape = new int[]{1, bitmap.getWidth(), bitmap.getHeight(), 3};
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

                tensor.data[index] = red;
                index++;
                tensor.data[index] = green;
                index++;
                tensor.data[index] = blue;
                index++;
            }
        }
        Log.i("LoadResourceImage", "Took " + timer.getTimeDelta() + "ms");
        return tensor;
    }

    private int toColorValue(float value) {
        int int_value = (int) (value);
        if (int_value < 0) {
            return 0;
        }
        if (int_value > 255) {
            return 255;
        }
        return int_value;
    }

    private Bitmap TensorToBitmap(Tensor tensor) {
        Timer timer = new Timer();
        int width = tensor.shape[1];
        int height = tensor.shape[2];

        Bitmap.Config conf = Bitmap.Config.ARGB_8888;
        Bitmap image = Bitmap.createBitmap(width, height, conf);

        int index = 0;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                int red = toColorValue(tensor.data[index]);
                index++;
                int green = toColorValue(tensor.data[index]);
                index++;
                int blue = toColorValue(tensor.data[index]);
                index++;
                int color = Color.rgb(
                        red,
                        green,
                        blue
                );
                image.setPixel(x, y, color);
            }
        }
        Log.i("TensorToBitmap", "Took " + timer.getTimeDelta() + "ms");
        return image;
    }

    private native String prepareInterpreter(String model_path);

    private native String runStyleTransfer(int svd_rank,
                                           Tensor content,
                                           Tensor style,
                                           Tensor result);

    public class ExecutionError extends RuntimeException {
        public ExecutionError(String error) {
            super(error);
        }
    }
}
