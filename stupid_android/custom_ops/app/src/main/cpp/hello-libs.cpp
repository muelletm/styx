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
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "hello-libs::", __VA_ARGS__))

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
    // Just for simplicity, we do this right away; correct way would do it in
    // another thread...
    auto ticks = GetTicks();

    for (auto exp = 0; exp < 32; ++exp) {
        volatile unsigned val = gpower(exp);
        (void) val;  // to silence compiler warning
    }
    ticks = GetTicks() - ticks;

    LOGI("calculation time: %" PRIu64, ticks);

    {
        std::call_once ( onceFlag, [ ]{ tflite::ops::custom::Init(); } );
    }

    return env->NewStringUTF("Hello from JNI LIBS!");
}