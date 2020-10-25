package com.stupid.styx_cc;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    private final static int PERM_REQUEST_CODE_ = 1;
    private final static int[] STYLES = new int[]{
            R.drawable.style1,
            R.drawable.style2,
            R.drawable.style3,
            R.drawable.style4,
            R.drawable.style5,
            R.drawable.style_1,
            R.drawable.style_2,
            R.drawable.style_3,
            R.drawable.style_4,
            R.drawable.style_5,
            R.drawable.style_6,
            R.drawable.style_7,
            R.drawable.style_8,
            R.drawable.style_9,
            R.drawable.style_10,
            R.drawable.style_11,
            R.drawable.style_12,
            R.drawable.style_13,
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
            R.drawable.fox_5617008_640,
            R.drawable.girl_2242858_640,
            R.drawable.golden_gate_bridge_5673315_640,
            R.drawable.halloween_5658809_640,
            R.drawable.heart_5677354_640,
            R.drawable.image_1247354_640,
            R.drawable.lace_5674462_640,
            R.drawable.landscape_4258253_640,
            R.drawable.loveourplanet_4851331_640,
            R.drawable.man_5631295_640,
            R.drawable.moon_5659196_640,
            R.drawable.painting_3995999_640,
            R.drawable.pair_2028068_640,
            R.drawable.parrot_5658203_640,
            R.drawable.porcupine_5677365_640,
            R.drawable.reading_5173530_640,
            R.drawable.stars_5673499_640,
            R.drawable.turtle_5674360_640,
            R.drawable.virus_5672362_640,
            R.drawable.watercolour_2109383_640,
            R.drawable.woman_5644555_640,
            R.drawable.woman_5658211_640,
            R.drawable.woman_5668428_640,
    };

    private Model model_ = new Model("big4321.tflite",
            new ModelConfig(25000, 64, 256),
            new ModelConfig(85000, 128, 512));
    private ImageState image_state_ = ImageState.STYLE;
    private ModelState model_state_ = ModelState.UNINITIALIZED;
    private TextView tv_;
    private ImageView image_;
    private SeekBar stylebar_;
    private ImageView thumbnail_;

    Model getModel() {
        return model_;
    }

    private File getModelFile(Model model) {
        Log.i("main", "getModelFile");
        File downloads = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS);
        File model_file = new File(downloads, model.name);
        if (!model_file.exists()) {
            throw new RuntimeException("No such file: " + model_file.getAbsolutePath());
        }
        return model_file;
    }

    private void InitModel(Model model) {
        Log.i("main", "InitModel");
        if (model_state_ != ModelState.UNINITIALIZED) {
            // The model has already been initialized.
            return;
        }
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            setStatus("Storage permission denied.");
            return;
        }
        try {
            File modelPath = getModelFile(model);
            String init_message = model.Init(modelPath.getAbsolutePath());
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

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        Log.i("main", "onRequestPermissionsResult");
        assert requestCode == PERM_REQUEST_CODE_;
        assert permissions.length == 1;
        assert grantResults.length == 1;
        assert permissions[0].equals(Manifest.permission.READ_EXTERNAL_STORAGE);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            setStatus("Permission granted.");
        } else {
            setStatus("Permission denied.");
        }
    }

    private Bitmap GetBitmapResource(int resource_index) {
        return BitmapFactory.decodeResource(getResources(), resource_index);
    }

    private String runModel(Model model, boolean preview) {
        Log.i("main", "runModel");
        final Timer timer = new Timer();
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            return "Permission denied.";
        }
        Bitmap content = GetBitmapResource(R.drawable.gilbert);
        Bitmap style = GetBitmapResource(STYLES[stylebar_.getProgress()]);
        try {
            Bitmap image = model.Run(preview, content, style);
            int new_height = content.getHeight();
            int new_width = content.getWidth();
            image = Bitmap.createScaledBitmap(image, new_width, new_height, true);
            setImage(image);
        } catch (Model.ExecutionError error) {
            return "Transfer failed: " + error.getMessage();
        }
        setThumbnail(STYLES[stylebar_.getProgress()]);
        if (preview) {
            setStatus("Preview (" + (timer.getTimeDelta()) + " milliseconds)");
        } else {
            setStatus("Full (" + (timer.getTimeDelta()) + " milliseconds)");
        }
        return "";
    }

    private void startModelThread(final Model model, final boolean preview) {
        new Thread(new Runnable() {
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
        }).start();
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
                image_.setImageBitmap(bitmap);
            }
        });
    }

    private void setImage(final int resource) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                image_.setImageResource(resource);
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

    private void startProgressThread(final Model model, final boolean preview) {
        new Thread(new Runnable() {
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
        }).start();
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
                        Log.i("Stylebar", "Style: " + progress);
                        setImage(STYLES[progress]);
                        setThumbnail(R.drawable.gilbert);
                        image_state_ = ImageState.STYLE;
                    }

                    @Override
                    public void onStartTrackingTouch(SeekBar seekBar) {

                    }

                    @Override
                    public void onStopTrackingTouch(SeekBar seekBar) {

                    }
                }
        );


        image_ = findViewById(R.id.styleView);
        thumbnail_ = findViewById(R.id.Thumbnail);
        tv_ = findViewById(R.id.textView);

        image_.setOnClickListener(new View.OnClickListener() {
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
                final boolean preview = image_state_ == ImageState.STYLE;
                startModelThread(model, preview);
                startProgressThread(model, preview);
            }
        });

        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            setStatus("Waiting for permissions");
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERM_REQUEST_CODE_);
        } else {
            setStatus("Good to go!");
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
        if (state == ModelState.RUNNING) {
            setStyleBarEnabled(false);
        } else {
            setStyleBarEnabled(true);
        }
        model_state_ = state;
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
