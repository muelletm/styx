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

std::string RunInterpreter() {
    // Create model from file. Note that the model instance must outlive the
    // interpreter instance.
    constexpr char kModelFile[] = "";
    auto model = tflite::FlatBufferModel::BuildFromFile(kModelFile);
    if (model == nullptr) {
        LOGE("No such file: %s", kModelFile);
        return "ERROR";
    }
    // Create an Interpreter with an InterpreterBuilder.
    std::unique_ptr<Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        LOGE("Cannot build interpreter");
        return "ERROR";
    }

    if (interpreter->ResizeInputTensor(0, {1}) != kTfLiteOk) {
        LOGE("Cannot resize input tensors");
        return "ERROR";
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOGE("Cannot allocate tensors");
        return "ERROR";
    }

    interpreter->typed_tensor<float>(0)[0] = 1.0;

    if (interpreter->Invoke() != kTfLiteOk) {
        LOGE("Cannot invoke");
        return "ERROR";
    }

    float output = interpreter->typed_output_tensor<float>(0)[0];

    return "SIN(1.0) = " + std::to_string(output) + ".";
}

}  // namespace tflite

/* This is a trivial JNI example where we use a native method
 * to return a new VM String. See the corresponding Java source
 * file located at:
 *
 *   app/src/main/java/com/example/hellolibs/MainActivity.java
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_hellolibs_MainActivity_stringFromJNI(JNIEnv *env, jobject thiz) {
    static std::once_flag onceFlag;
    {
        std::call_once ( onceFlag, [ ]{ tflite::ops::custom::Init(); } );
    }

    std::string message = tflite::RunInterpreter();
    return env->NewStringUTF(message.c_str());
}