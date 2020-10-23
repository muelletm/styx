#include <android/log.h>
#include <cinttypes>
#include <cstring>
#include <gmath.h>
#include <gperf.h>
#include <jni.h>
#include <string>
#include <optional>
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "Eigen/SVD"
#include "Eigen/Dense"

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "hello-libs::", __VA_ARGS__))

#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "oh shit::", __VA_ARGS__))

static tflite::FlatBufferModel *model_ = nullptr;
static tflite::Interpreter *interpreter_ = nullptr;


namespace tflite {
    namespace ops {
        namespace custom {

            bool get2dShape(const TfLiteTensor &tensor, std::vector<int> *shape) {
                if (tensor.dims->size == 2) {
                    *shape = {tensor.dims->data[0], tensor.dims->data[1]};
                    return true;
                }
                if (tensor.dims->size == 3) {
                    if (tensor.dims->data[0] == 1) {
                        *shape = {tensor.dims->data[1], tensor.dims->data[2]};
                        return true;
                    } else {
                        LOGE("Invalid first dimension.");
                    }
                } else {
                    LOGE("Invalid number of dimensions.");
                }
                return false;
            }

            TfLiteStatus SvdPrepare(TfLiteContext *context, TfLiteNode *node) {
                LOGI("Prepare SVD");
                TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
                TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);
                const TfLiteTensor *input = GetInput(context, node, 0);

                std::vector<int> input_shape;
                if (!get2dShape(*input, &input_shape)) {
                    return kTfLiteError;
                }



                int num_rows = input_shape[0];
                int num_cols = input_shape[1];
                if (num_rows != num_cols) {
                    LOGE("Non-quadratic shapes are not supported!");
                    return kTfLiteError;
                }

                TfLiteTensor *s = GetOutput(context, node, 0);
                TfLiteTensor *u = GetOutput(context, node, 1);
                TfLiteTensor *v = GetOutput(context, node, 2);

                int num_dims = NumDimensions(input);

                TfLiteIntArray *s_output_size = TfLiteIntArrayCreate(num_dims - 1);
                for (int i = 0; i < num_dims - 1; ++i) {
                    s_output_size->data[i] = input->dims->data[i];
                }

                TfLiteIntArray *u_output_size = TfLiteIntArrayCreate(num_dims);
                for (int i = 0; i < num_dims; ++i) {
                    u_output_size->data[i] = input->dims->data[i];
                }

                TfLiteIntArray *v_output_size = TfLiteIntArrayCreate(num_dims);
                for (int i = 0; i < num_dims; ++i) {
                    v_output_size->data[i] = input->dims->data[i];
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

            TfLiteStatus SvdEval(TfLiteContext *context, TfLiteNode *node) {
                LOGI("Eval SVD");
                using namespace tflite;
                const TfLiteTensor *input = GetInput(context, node, 0);
                TfLiteTensor *s = GetOutput(context, node, 0);
                TfLiteTensor *u = GetOutput(context, node, 1);
                TfLiteTensor *v = GetOutput(context, node, 2);

                float *input_data = input->data.f;
                float *s_data = s->data.f;
                float *u_data = u->data.f;
                float *v_data = v->data.f;

                std::vector<int> input_shape;
                if (!get2dShape(*input, &input_shape)) {
                    return kTfLiteError;
                }
                int num_rows = input_shape[0];
                int num_cols = input_shape[1];

                Eigen::MatrixXf input_eigen(num_rows, num_cols);
                for (int index = 0; index < NumElements(input); ++index) {
                    int row = index / input_shape[0];
                    int col = index % input_shape[0];
                    input_eigen(row, col) = input_data[index];
                }

                Eigen::BDCSVD<Eigen::MatrixXf> svd(input_eigen,
                                                   Eigen::ComputeFullU | Eigen::ComputeFullV);

                const Eigen::VectorXf &s_eigen = svd.singularValues();
                for (int i = 0; i < num_rows; i++) {
                    s_data[i] = s_eigen(i);
                }

                const Eigen::MatrixXf &u_eigen = svd.matrixU();
                for (int r = 0; r < u_eigen.rows(); r++) {
                    for (int c = 0; c < u_eigen.cols(); c++) {
                        int index = r * input_shape[0] + c;
                        u_data[index] = u_eigen(r, c);
                        LOGI("%i %i %f", r, c, u_eigen(r, c));
                    }
                }
                const Eigen::MatrixXf &v_eigen = svd.matrixV();
                for (int r = 0; r < v_eigen.rows(); r++) {
                    for (int c = 0; c < v_eigen.cols(); c++) {
                        int index = r * input_shape[0] + c;
                        v_data[index] = v_eigen(r, c);
                    }
                }

                return kTfLiteOk;
            }

            TfLiteRegistration *Register_SVD() {
                LOGI("Register SVD");
                static TfLiteRegistration r = {nullptr, nullptr, SvdPrepare, SvdEval};
                return &r;
            }

            std::string PrepareInterpreter(const std::string &modelPath) {
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

            void populateOutput(JNIEnv *env, int index, jobject output) {
                TfLiteTensor *tensor = interpreter_->output_tensor(index);
                float *data = interpreter_->typed_output_tensor<float>(index);

                int data_length = NumElements(tensor);
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
            }

            std::string
            runTransfer(JNIEnv *env,
                   const std::vector<float> &content,
                   const std::vector<int> &content_shape,
                   const std::vector<float> &style,
                   const std::vector<int> &style_shape,
                   jobject result) {
                if (interpreter_ == nullptr) {
                    return "ERROR: Interpreter is null.";
                }

                if (interpreter_->inputs().size() != 2) {
                    return "ERROR: Unexpected number of inputs.";
                }

                auto c_dims = interpreter_->input_tensor(0)->dims;
                for (int i=0; i<c_dims->size; ++i) {
                    LOGI("c_dim[%d]: %d", i, c_dims->data[i]);
                }
                auto s_dims = interpreter_->input_tensor(1)->dims;
                for (int i=0; i<s_dims->size; ++i) {
                    LOGI("s_dim[%d]: %d", i, s_dims->data[i]);
                }

                interpreter_->ResizeInputTensor(0, content_shape);
                interpreter_->ResizeInputTensor(1, style_shape);

                if (interpreter_->AllocateTensors() != kTfLiteOk) {
                    return "ERROR: Cannot allocate tensors";
                }

                float* content_data = interpreter_->typed_input_tensor<float>(0);
                for (int i = 0; i < 9; i++) {
                    content_data[i] = content[i];
                }

                float* style_data = interpreter_->typed_input_tensor<float>(1);
                for (int i = 0; i < 9; i++) {
                    style_data[i] = style[i];
                }

                if (interpreter_->Invoke() != kTfLiteOk) {
                    return "ERROR: Cannot invoke";
                }

                // The graph should only use the s and u tensors.
                if (interpreter_->outputs().size() != 1) {
                    LOGE("ERROR: Unexpected number of outputs: %d", interpreter_->outputs().size());
                    return "ERROR: Unexpected number of outputs: " +
                           std::to_string(interpreter_->outputs().size());
                }

                populateOutput(env, 0, result);
                return "";
            }

        }  // namespace custom
    }  // namespace ops
}  // namespace tflite

namespace {

    std::string jstring2string(JNIEnv *env, jstring jStr) {
        if (!jStr)
            return "";

        const jclass stringClass = env->GetObjectClass(jStr);
        const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes",
                                                    "(Ljava/lang/String;)[B");
        const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                           env->NewStringUTF(
                                                                                   "UTF-8"));

        size_t length = (size_t) env->GetArrayLength(stringJbytes);
        jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

        std::string ret = std::string((char *) pBytes, length);
        env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

        env->DeleteLocalRef(stringJbytes);
        env->DeleteLocalRef(stringClass);
        return ret;
    }

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_styx_1cc_MainActivity_initSvd(JNIEnv *env, jobject thiz, jstring modelPath) {
    std::string output = tflite::ops::custom::PrepareInterpreter(jstring2string(env, modelPath));
    return env->NewStringUTF(output.c_str());
}

std::vector<float> toFloatVec(JNIEnv *env, jfloatArray input) {
    int input_length = env->GetArrayLength(input);
    jfloat *body = env->GetFloatArrayElements(input, nullptr);
    std::vector<float> input_vec(input_length);
    for (int i = 0; i < input_length; i++)
        input_vec[i] += body[i];
    env->ReleaseFloatArrayElements(input, body, 0);
    return input_vec;
}

std::vector<int> toIntVec(JNIEnv *env, jintArray input) {
    int input_length = env->GetArrayLength(input);
    jint *body = env->GetIntArrayElements(input, nullptr);
    std::vector<int> input_vec(input_length);
    for (int i = 0; i < input_length; i++)
        input_vec[i] += body[i];
    env->ReleaseIntArrayElements(input, body, 0);
    return input_vec;
}

std::vector<int> GetIntVecField(JNIEnv *env, const jobject &input, const std::string &field_name) {
    jclass tensor = env->GetObjectClass(input);
    jfieldID fieldID = env->GetFieldID(tensor, field_name.c_str(), "[I");
    jobject object = env->GetObjectField(input, fieldID);
    jintArray *array = reinterpret_cast<jintArray *>(&object);
    return toIntVec(env, *array);
}

std::vector<float>
GetFloatVecField(JNIEnv *env, const jobject &input, const std::string &field_name) {
    jclass tensor = env->GetObjectClass(input);
    jfieldID fieldID = env->GetFieldID(tensor, field_name.c_str(), "[F");
    jobject object = env->GetObjectField(input, fieldID);
    jfloatArray *array = reinterpret_cast<jfloatArray *>(&object);
    return toFloatVec(env, *array);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_styx_1cc_MainActivity_runStyleTransfer(JNIEnv *env,
        jobject thiz,
        jobject content,
        jobject style,
        jobject result) {
    const std::vector<float> content_data = GetFloatVecField(env, content, "data");
    const std::vector<int> content_shape = GetIntVecField(env, content, "shape");
    const std::vector<float> style_data = GetFloatVecField(env, style, "data");
    const std::vector<int> style_shape = GetIntVecField(env, style, "shape");
    const std::string output = tflite::ops::custom::runTransfer(
            env, content_data, content_shape, style_data, style_shape, result);
    return env->NewStringUTF(output.c_str());
}