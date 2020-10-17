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
 *
 */
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

namespace tflite {
namespace ops {
namespace custom {

TfLiteStatus SinPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);

    int num_dims = NumDimensions(input);

    TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
    for (int i=0; i<num_dims; ++i) {
        output_size->data[i] = input->dims->data[i];
    }

    return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
    using namespace tflite;
    const TfLiteTensor* input = GetInput(context, node,0);
    TfLiteTensor* output = GetOutput(context, node,0);

    float* input_data = input->data.f;
    float* output_data = output->data.f;

    size_t count = 1;
    int num_dims = NumDimensions(input);
    for (int i = 0; i < num_dims; ++i) {
        count *= input->dims->data[i];
    }

    for (size_t i=0; i<count; ++i) {
        output_data[i] = sin(input_data[i]);
    }
    return kTfLiteOk;
}

TfLiteRegistration* Register_SIN() {
    static TfLiteRegistration r = {nullptr, nullptr, SinPrepare, SinEval};
    return &r;
}

void Init() {
    LOGI("Registering Sin.");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Sin", Register_SIN());
}

}  // namespace custom
}  // namespace ops

std::string RunInterpreter(std::string modelFile) {
    // Create model from file. Note that the model instance must outlive the
    // interpreter instance.
    auto model = tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());
    if (model == nullptr) {
        LOGE("No such file: %s", modelFile.c_str());
        return "ERROR: No such file: '" + std::string(modelFile) + "'";
    }
    // Create an Interpreter with an InterpreterBuilder.
    std::unique_ptr<Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        LOGE("Cannot build interpreter");
        return "ERROR: Cannot build interpreter";
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOGE("Cannot allocate tensors");
        return "ERROR: Cannot allocate tensors";
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        LOGE("Cannot invoke");
        return "ERROR: Cannot invoke";
    }



    std::string output_message = "SIN([-8, 0.5, 2, 2.2, 201]) = [";

    int num_outputs = interpreter->outputs().size();
    if (num_outputs != 1) {
        return "ERROR: unexpected number of outputs: " + std::to_string(num_outputs);
    }
    const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    if (output_tensor->dims->size != 1) {
        return "ERROR: unexpected output dimension " + std::to_string(output_tensor->dims->size);
    }

    int output_dim = (output_tensor->dims->data[0]);
    for (int i=0; i<output_dim; i++) {
        float output = interpreter->typed_output_tensor<float>(0)[i];
        output_message += " " + std::to_string(output);
    }

    return output_message + "].";
}

}  // namespace tflite

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

/* This is a trivial JNI example where we use a native method
 * to return a new VM String. See the corresponding Java source
 * file located at:
 *
 *   app/src/main/java/com/example/hellolibs/MainActivity.java
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_hellolibs_MainActivity_stringFromJNI(JNIEnv *env, jobject thiz, jstring filePath) {
    static std::once_flag onceFlag;
    {
        std::call_once ( onceFlag, [ ]{ tflite::ops::custom::Init(); } );
    }

    std::string message = tflite::RunInterpreter(jstring2string(env, filePath));
    return env->NewStringUTF(message.c_str());
}