package com.stupid.styx;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import static java.io.File.separator;

public class MainActivity extends AppCompatActivity {

    private final static int PERM_REQUEST_CODE_ = 1;
    private final static int[] STYLES = new int[]{
            R.drawable.art_2092530_640,
            R.drawable.art_2108118_640,
            R.drawable.art_3125816_640,
            R.drawable.art_4178302_640,
            R.drawable.background_2719576_640,
            R.drawable.background_2734972_640,
            R.drawable.background_2743842_640,
            R.drawable.camel_5674406_640,
            R.drawable.cartoon_5544856_640,
            R.drawable.color_4287692_640,
            R.drawable.elephant_5671866_640,
            R.drawable.eye_2555760_640,
            R.drawable.forest_5656930_640,
//            R.drawable.fox_5617008_640, results in a completely black image
            R.drawable.girl_2242858_640,
            R.drawable.golden_gate_bridge_5673315_640,
            R.drawable.halloween_5658809_640,
            R.drawable.heart_5677354_640,
            R.drawable.image_1247354_640,
//            R.drawable.lace_5674462_640, causes a SIGSEGV
            R.drawable.landscape_4258253_640,
            R.drawable.loveourplanet_4851331_640,
            R.drawable.man_5631295_640,
            R.drawable.moon_5659196_640,
            R.drawable.painting_3995999_640,
            R.drawable.pair_2028068_640,
            R.drawable.parrot_5658203_640,
//            R.drawable.porcupine_5677365_640, causes a SIGSEGV
            R.drawable.reading_5173530_640,
            R.drawable.stars_5673499_640,
            R.drawable.turtle_5674360_640,
            R.drawable.virus_5672362_640,
            R.drawable.watercolour_2109383_640,
            R.drawable.woman_5644555_640,
            R.drawable.woman_5658211_640,
            R.drawable.woman_5668428_640,
    };

    private final Model model_ = new Model("big4321_dq.tflite",
            new ModelConfig(25000, 64, 256),
            new ModelConfig(85000, 128, 512));
    private ImageState image_state_ = ImageState.STYLE;
    private ModelState model_state_ = ModelState.UNINITIALIZED;
    private TextView tv_;
    private ImageView image_view_;
    private SeekBar stylebar_;
    private ImageView thumbnail_;
    private Bitmap image_;

    Model getModel() {
        return model_;
    }

    private void InitModel(Model model) {
        Log.i("main", "InitModel");
        if (model_state_ != ModelState.UNINITIALIZED) {
            // The model has already been initialized.
            return;
        }
        try {
            String init_message = model.Init();
            if (!init_message.isEmpty()) {
                setStatus("Init failed: " + init_message);
                return;
            }
        } catch (RuntimeException e) {
            setStatus(e.getMessage());
            return;
        }
        setModelState(ModelState.IDLE);
        setStatus("Waiting...");
    }

    private Bitmap GetBitmapResource(int resource_index) {
        return BitmapFactory.decodeResource(getResources(), resource_index);
    }

    private ContentValues contentValues() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
        values.put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis());
        return values;
    }

    private void saveImageToStream(Bitmap bitmap, OutputStream outputStream) throws IOException {
        if (outputStream != null) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
            outputStream.flush();
            outputStream.close();
        }
    }

    private void saveImage(Bitmap bitmap, Context context, String name) throws IOException {
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            Log.i("saveMediaImage", "New API");
            ContentValues values = contentValues();
            values.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures" + separator + "styx");
            values.put(MediaStore.Images.Media.IS_PENDING, true);
            values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
            Uri uri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri != null) {
                saveImageToStream(bitmap, context.getContentResolver().openOutputStream(uri));
                values.put(MediaStore.Images.Media.IS_PENDING, false);
                context.getContentResolver().update(uri, values, null, null);
            }
        } else {
            Log.i("saveMediaImage", "Old API");
            File directory = new File(Environment.getExternalStorageDirectory().toString(), "styx");
            if (!directory.exists()) {
                if (!directory.mkdirs()) {
                    throw new FileNotFoundException("Cannot create directory: " + directory.getAbsolutePath());
                }
            }
            String fileName = name + ".png";
            File file = new File(directory, fileName);
            saveImageToStream(bitmap, new FileOutputStream(file));
            if (file.getAbsolutePath() != null) {
                ContentValues values = contentValues();
                values.put(MediaStore.Images.Media.DATA, file.getAbsolutePath());
                context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            }
        }
    }

    private Thread saveImageThread(final Bitmap image, final int style_id) {
        Thread thread = new Thread(new Runnable() {
            public void run() {
                if (image == null) {
                    setStatus("Image is null.");
                }
                if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    setStatus("No permission to write files.");
                }
                try {
                    saveImage(image, getApplicationContext(), getResources().getResourceEntryName(style_id));
                } catch (IOException e) {
                    setStatus(e.getMessage());
                }
                setStatus("Image saved.");
            }
        });
        thread.start();
        return thread;
    }

    private String runModel(Model model, boolean preview) {
        Log.i("main", "runModel");
        final Timer timer = new Timer();
        Bitmap content = GetBitmapResource(R.drawable.gilbert);
        int style_id = STYLES[stylebar_.getProgress()];
        Bitmap style = GetBitmapResource(style_id);
        try {
            image_ = model.Run(preview, content, style);
            setImage(image_);
        } catch (Model.ExecutionError error) {
            return "Transfer failed: " + error.getMessage();
        }
        setThumbnail(style_id);
        if (preview) {
            setStatus("Preview (" + (timer.getTimeDelta()) + " milliseconds)");
        } else {
            setStatus("Full (" + (timer.getTimeDelta()) + " milliseconds)");
        }
        return "";
    }

    private Thread startModelThread(final Model model, final boolean preview) {
        Thread thread = new Thread(new Runnable() {
            public void run() {
                String error = runModel(model, preview);
                image_state_ = ImageState.PREVIEW;
                if (!error.isEmpty()) {
                    setStatus(error);
                } else if (preview) {
                    image_state_ = ImageState.PREVIEW;
                } else {
                    image_state_ = ImageState.FULL;
                }
                setModelState(ModelState.IDLE);
            }
        });
        thread.start();
        return thread;
    }

    private void setStatus(final String status_message) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tv_.setText(status_message);
            }
        });
    }

    private void setImage(final Bitmap bitmap) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                image_view_.setImageBitmap(bitmap);
            }
        });
    }

    private void setImage(final int resource) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                image_view_.setImageResource(resource);
            }
        });
    }

    private void setThumbnail(final int resource) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                thumbnail_.setImageResource(resource);
            }
        });
    }

    private Thread startProgressThread(final Model model, final boolean preview) {
        Thread thread = new Thread(new Runnable() {
            final Timer timer = new Timer();

            public void run() {
                while (true) {
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                    }
                    if (model_state_ != ModelState.RUNNING) {
                        return;
                    }
                    double run_time = timer.getTimeDelta();
                    double total_run_time = model.getModelConfig(preview).runtime_ins_ms;
                    final int progress = Math.min((int) ((run_time / total_run_time) * 100.0), 100);
                    if (preview) {
                        setStatus("Creating preview (" + progress + "%)");
                    } else {
                        setStatus("Creating image (" + progress + "%)");
                    }
                    if (progress >= 100) {
                        return;
                    }
                }
            }
        });
        thread.start();
        return thread;
    }

    private void SetProgress(int progress) {
        Log.i("Stylebar", "Style: " + progress);
        setImage(STYLES[progress]);
        setThumbnail(R.drawable.gilbert);
        image_state_ = ImageState.STYLE;
        image_ = null;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        stylebar_ = findViewById(R.id.styleBar);
        stylebar_.setMin(0);
        stylebar_.setMax(STYLES.length - 1);
        stylebar_.setOnSeekBarChangeListener(
                new SeekBar.OnSeekBarChangeListener() {
                    @Override
                    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                        SetProgress(progress);
                    }

                    @Override
                    public void onStartTrackingTouch(SeekBar seekBar) {

                    }

                    @Override
                    public void onStopTrackingTouch(SeekBar seekBar) {

                    }
                }
        );


        image_view_ = findViewById(R.id.styleView);
        thumbnail_ = findViewById(R.id.Thumbnail);
        tv_ = findViewById(R.id.textView);

        image_view_.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Model model = getModel();
                if (model_state_ == ModelState.UNINITIALIZED) {
                    InitModel(model);
                }
                if (model_state_ != ModelState.IDLE) {
                    return;
                }
                setModelState(ModelState.RUNNING);
                switch (image_state_) {
                    case STYLE:
                    case PREVIEW:
                        final boolean preview = image_state_ == ImageState.STYLE;
                        startModelThread(model, preview);
                        startProgressThread(model, preview);
                        break;
                    case FULL:
                        saveImageThread(image_, STYLES[stylebar_.getProgress()]);
                        break;
                }
            }
        });

        if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            setStatus("Waiting for permissions");
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERM_REQUEST_CODE_);
        } else {
            setStatus("Good to go!");
        }

        thumbnail_.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        runAndSaveAll(false);
                    }
                }
        );

        SetProgress(0);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        Log.i("main", "onRequestPermissionsResult");
        assert requestCode == PERM_REQUEST_CODE_;
        assert permissions.length == 1;
        assert grantResults.length == 1;
        assert permissions[0].equals(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            setStatus("Permission granted.");
        } else {
            setStatus("Permission denied.");
        }
    }

    private void setStyleBarEnabled(final boolean enabled) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                stylebar_.setEnabled(enabled);
            }
        });
    }

    private void setModelState(ModelState state) {
        setStyleBarEnabled(state != ModelState.RUNNING);
        model_state_ = state;
    }

    void runAndSaveAll(final boolean preview) {
        new Thread(new Runnable() {
            public void run() {
                Log.i("runAndSaveAll", "Disable UI");
                stylebar_.setEnabled(false);
                image_view_.setEnabled(false);
                Model model = getModel();
                if (model_state_ == ModelState.UNINITIALIZED) {
                    InitModel(model);
                }
                for (int i = stylebar_.getMin(); i <= stylebar_.getMax(); ++i) {
                    Log.i("runAndSaveAll", "Processing: " + i);
                    try {
                        stylebar_.setProgress(i);
                        model_state_ = ModelState.RUNNING;
                        Thread model_thread = startModelThread(model, preview);
                        Thread progress_thread = startProgressThread(model, preview);
                        model_thread.join();
                        model_state_ = ModelState.IDLE;
                        progress_thread.join();
                        Log.i("runAndSaveAll", "Storing model: " + i);
                        Thread image_thread = saveImageThread(image_, STYLES[stylebar_.getProgress()]);
                        synchronized(image_thread) {
                            image_thread.join();
                        }
                    } catch (InterruptedException e) {
                        setStatus(e.getMessage());
                    }
                }
                Log.i("runAndSaveAll", "Enable UI");
                stylebar_.setEnabled(true);
                image_view_.setEnabled(true);
            }
        }).start();
    }

    private enum ImageState {
        STYLE,
        PREVIEW,
        FULL,
    }

    private enum ModelState {
        UNINITIALIZED,
        RUNNING,
        IDLE,
    }
}
