#include <android/log.h>
#include <cinttypes>
#include <cstring>
#include <jni.h>
#include <string>
#include <optional>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

#include "jni_utils.h"
#include "svd_op.h"
#include "timing.h"

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "register_svd::", __VA_ARGS__))

#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "register_svd::", __VA_ARGS__))

static tflite::FlatBufferModel *model_ = nullptr;
static tflite::Interpreter *interpreter_ = nullptr;

std::string PrepareInterpreter(const std::string &modelPath) {
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str()).release();
    if (model_ == nullptr) {
        return "No such file: '" + std::string(modelPath) + "'";
    }

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Svd", tflite::ops::custom::Register_SVD());
    if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter) != kTfLiteOk) {
        return "Cannot build interpreter";
    }
    interpreter_ = interpreter.release();
    return "";
}

void populateOutput(JNIEnv *env, int index, jobject output) {
    LOGI("populateOutput");
    long start_time = currentTimeMillis();
    TfLiteTensor *tensor = interpreter_->output_tensor(index);
    float *data = interpreter_->typed_output_tensor<float>(index);

    int data_length = tflite::NumElements(tensor);
    jfloatArray data_array = env->NewFloatArray(data_length);
    env->SetFloatArrayRegion(data_array, 0, data_length, (jfloat *) data);

    jclass tensor_class = env->GetObjectClass(output);
    jfieldID dataID = env->GetFieldID(tensor_class, "data", "[F");
    env->SetObjectField(output, dataID, data_array);

    int shape_length = tensor->dims->size;
    jintArray shape_array = env->NewIntArray(shape_length);
    env->SetIntArrayRegion(shape_array, 0, shape_length, (jint *) tensor->dims->data);

    jfieldID shapeID = env->GetFieldID(tensor_class, "shape", "[I");
    env->SetObjectField(output, shapeID, shape_array);
    LOGI("populateOutput (took %ldms)", currentTimeMillis() - start_time);
}

std::string
runTransfer(JNIEnv *env,
            const std::vector<float> &content,
            const std::vector<int> &content_shape,
            const std::vector<float> &style,
            const std::vector<int> &style_shape,
            jobject result) {
    long start_time = currentTimeMillis();
    LOGI("runTransfer");

    if (interpreter_ == nullptr) {
        return "ERROR: Interpreter is null.";
    }

    if (interpreter_->inputs().size() != 2) {
        return "ERROR: Unexpected number of inputs.";
    }

    interpreter_->ResizeInputTensor(0, content_shape);
    interpreter_->ResizeInputTensor(1, style_shape);

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        return "ERROR: Cannot allocate tensors";
    }

    float *content_data = interpreter_->typed_input_tensor<float>(0);
    for (int i = 0; i < tflite::NumElements(interpreter_->input_tensor(0)); i++) {
        content_data[i] = content[i];
    }

    float *style_data = interpreter_->typed_input_tensor<float>(1);
    for (int i = 0; i < tflite::NumElements(interpreter_->input_tensor(1)); i++) {
        style_data[i] = style[i];
    }

    {
        long start_time = currentTimeMillis();
        if (interpreter_->Invoke() != kTfLiteOk) {
            return "ERROR: Cannot invoke";
        }
        LOGI("Invoke (took %ldms)", currentTimeMillis() - start_time);
    }

    // The graph should only use the s and u tensors.
    if (interpreter_->outputs().size() != 1) {
        LOGE("ERROR: Unexpected number of outputs: %lu",
             (unsigned long) interpreter_->outputs().size());
        return "ERROR: Unexpected number of outputs: " +
               std::to_string(interpreter_->outputs().size());
    }

    populateOutput(env, 0, result);
    LOGI("Run Transfer (took %ldms)", currentTimeMillis() - start_time);
    return "";
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_styx_1cc_Model_prepareInterpreter(JNIEnv *env, jobject thiz,
                                                  jstring modelPath) {
    std::string output = PrepareInterpreter(jstringToString(env, modelPath));
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_styx_1cc_Model_runStyleTransfer(JNIEnv *env,
                                                jobject thiz,
                                                jint svd_rank,
                                                jobject content,
                                                jobject style,
                                                jobject result) {

    LOGI("runStyleTransfer");
    tflite::ops::custom::setSvdRank((int) svd_rank);

    long start_time = currentTimeMillis();
    const std::vector<float> content_data = getFloatVecField(env, content, "data");
    const std::vector<int> content_shape = getIntVecField(env, content, "shape");
    const std::vector<float> style_data = getFloatVecField(env, style, "data");
    const std::vector<int> style_shape = getIntVecField(env, style, "shape");
    LOGI("Prepared inputs (took %ldms)", currentTimeMillis() - start_time);
    const std::string output = runTransfer(
            env, content_data, content_shape, style_data, style_shape, result);
    return env->NewStringUTF(output.c_str());
}