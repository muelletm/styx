#include <android/log.h>
#include <cinttypes>
#include <cstring>
#include <gmath.h>
#include <gperf.h>
#include <jni.h>
#include <mutex>
#include <string>
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "hello-libs::", __VA_ARGS__))

#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "oh shit::", __VA_ARGS__))

static tflite::FlatBufferModel* model_ = nullptr;
static tflite::Interpreter* interpreter_ = nullptr;


namespace tflite {
namespace ops {
namespace custom {

TfLiteStatus SvdPrepare(TfLiteContext* context, TfLiteNode* node) {
    LOGI("Prepare SVD");
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);
    const TfLiteTensor* matrix = GetInput(context, node, 0);
    TfLiteTensor* s = GetOutput(context, node, 0);
    TfLiteTensor* u = GetOutput(context, node, 1);
    TfLiteTensor* v = GetOutput(context, node, 2);

    int num_dims = NumDimensions(matrix);

    TfLiteIntArray* s_output_size = TfLiteIntArrayCreate(num_dims - 1);
    for (int i=0; i<num_dims - 1; ++i) {
        s_output_size->data[i] = matrix->dims->data[i];
    }

    TfLiteIntArray* u_output_size = TfLiteIntArrayCreate(num_dims);
    for (int i=0; i<num_dims; ++i) {
        u_output_size->data[i] = matrix->dims->data[i];
    }

    TfLiteIntArray* v_output_size = TfLiteIntArrayCreate(num_dims);
    for (int i=0; i<num_dims; ++i) {
        v_output_size->data[i] = matrix->dims->data[i];
    }

    TfLiteStatus status = context->ResizeTensor(context, s, s_output_size);
    if (status != kTfLiteOk) {
        return status;
    }
    status = context->ResizeTensor(context, u, u_output_size);
    if (status != kTfLiteOk) {
        return status;
    }
    status = context->ResizeTensor(context, v, v_output_size);
    return status;
}

void SetToZero(float* data, TfLiteTensor* tensor) {
    size_t count = 1;
    int num_dims = NumDimensions(tensor);
    for (int i = 0; i < num_dims; ++i) {
        count *= tensor->dims->data[i];
    }
    for (size_t i=0; i<count; ++i) {
        data[i] = 0.0;
    }
}

void Copy(float* src, float* trg, TfLiteTensor* tensor) {
    size_t count = 1;
    int num_dims = NumDimensions(tensor);
    for (int i = 0; i < num_dims; ++i) {
        count *= tensor->dims->data[i];
    }
    for (size_t i=0; i<count; ++i) {
        trg[i] = src[i];
    }
}

TfLiteStatus SvdEval(TfLiteContext* context, TfLiteNode* node) {
    LOGI("Eval SVD");
    using namespace tflite;
    const TfLiteTensor* matrix = GetInput(context, node,0);
    TfLiteTensor* s = GetOutput(context, node,0);
    TfLiteTensor* u = GetOutput(context, node,1);
    TfLiteTensor* v = GetOutput(context, node,2);

    float* input_data = matrix->data.f;
    float* s_data = s->data.f;
    float* u_data = u->data.f;
    float* v_data = v->data.f;

    SetToZero(s_data, s);
    Copy(input_data, u_data, u);
    Copy(input_data, v_data, v);
    return kTfLiteOk;
}

TfLiteRegistration* Register_SVD() {
    LOGI("Register SVD");
    static TfLiteRegistration r = {nullptr, nullptr, SvdPrepare, SvdEval};
    return &r;
}

std::string PrepareInterpreter(const std::string& modelPath) {
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str()).release();
    if (model_ == nullptr) {
        return "No such file: '" + std::string(modelPath) + "'";
    }

    std::unique_ptr<Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Svd", Register_SVD());
    if (InterpreterBuilder(*model_, resolver)(&interpreter) != kTfLiteOk) {
        return "Cannot build interpreter";
    }
    interpreter_ = interpreter.release();
    return "";
}

void appendOutput(const int index, const std::string& name, std::string& message) {
    TfLiteTensor* tensor = interpreter_->output_tensor(index);
    message += name;
    message += " :";

    int length = 1;
    message += "shape = (";
    for (int dim=0; dim < tensor->dims->size; ++dim) {
        message += " " + std::to_string(tensor->dims->data[dim]);
        length *= tensor->dims->data[dim];
    }
    message += ")";

    float* data = interpreter_->typed_output_tensor<float>(index);
    message += "[";
    for (int i=0; i < length; ++i) {
        message += " " + std::to_string(data[i]);
    }
    message += "]";
}

std::string runSvD() {
    if (interpreter_ == nullptr) {
        return "ERROR: Interpreter is null.";
    }
    interpreter_->ResizeInputTensor(0, {1, 3, 3});

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        return "ERROR: Cannot allocate tensors";
    }

    float inputs[] = {0., 0., 1., 0., 1., 0., 1., 0., 0. };

    float* input_data = interpreter_->typed_input_tensor<float>(0);
    for (int i=0; i<9; i++) {
        input_data[i] = inputs[i];
    }

    if (interpreter_->Invoke() != kTfLiteOk) {
        return "ERROR: Cannot invoke";
    }

    // The graph should only use the s and u tensors.
    if (interpreter_->outputs().size() != 2) {
        LOGE("ERROR: Unexpected number of outputs: %d", interpreter_->outputs().size());
        return "ERROR: Unexpected number of outputs: " + std::to_string(interpreter_->outputs().size());
    }

    std::string message;
    appendOutput(0, "s", message);
    message += "\n";
    appendOutput(1, "u", message);
    message += "\n";
    return message;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite

namespace {

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte* pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *)pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_customops_MainActivity_initSvd(JNIEnv *env, jobject thiz, jstring modelPath) {
    std::string output = tflite::ops::custom::PrepareInterpreter(jstring2string(env, modelPath));
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_customops_MainActivity_runSvd(JNIEnv *env, jobject thiz) {
    std::string output = tflite::ops::custom::runSvD();
    return env->NewStringUTF(output.c_str());
}