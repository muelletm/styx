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
                if(!get2dShape(*input, &input_shape)) {
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

                Eigen::BDCSVD <Eigen::MatrixXf> svd(input_eigen,
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

            void appendTensor(const TfLiteTensor &tensor, float data[], const std::string &name,
                              std::string &message) {
                message += name;
                message += " :";

                int length = 1;
                message += "shape = (";
                for (int dim = 0; dim < tensor.dims->size; ++dim) {
                    message += " " + std::to_string(tensor.dims->data[dim]);
                    length *= tensor.dims->data[dim];
                }
                message += ")";

                message += "[";
                for (int i = 0; i < length; ++i) {
                    message += " " + std::to_string(data[i]);
                }
                message += "]";
            }

            void appendOutput(const int index, const std::string &name, std::string &message) {
                TfLiteTensor *tensor = interpreter_->output_tensor(index);
                float *data = interpreter_->typed_output_tensor<float>(index);
                appendTensor(*tensor, data, name, message);
            }

            void appendInput(const int index, const std::string &name, std::string &message) {
                TfLiteTensor *tensor = interpreter_->input_tensor(index);
                float *data = interpreter_->typed_input_tensor<float>(index);
                appendTensor(*tensor, data, name, message);
            }

            std::string runSvD() {
                if (interpreter_ == nullptr) {
                    return "ERROR: Interpreter is null.";
                }
                interpreter_->ResizeInputTensor(0, {1, 3, 3});

                if (interpreter_->AllocateTensors() != kTfLiteOk) {
                    return "ERROR: Cannot allocate tensors";
                }

                float inputs[] = {1., 0., 1., 0., 1., 0., 1., 0., 0.};

                float *input_data = interpreter_->typed_input_tensor<float>(0);
                for (int i = 0; i < 9; i++) {
                    input_data[i] = inputs[i];
                }

                std::string message;
                appendInput(0, "input: ", message);

                if (interpreter_->Invoke() != kTfLiteOk) {
                    return "ERROR: Cannot invoke";
                }

                // The graph should only use the s and u tensors.
                if (interpreter_->outputs().size() != 2) {
                    LOGE("ERROR: Unexpected number of outputs: %d", interpreter_->outputs().size());
                    return "ERROR: Unexpected number of outputs: " +
                           std::to_string(interpreter_->outputs().size());
                }

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
Java_com_stupid_customops_MainActivity_initSvd(JNIEnv *env, jobject thiz, jstring modelPath) {
    std::string output = tflite::ops::custom::PrepareInterpreter(jstring2string(env, modelPath));
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_stupid_customops_MainActivity_runSvd(JNIEnv *env, jobject thiz) {
    std::string output = tflite::ops::custom::runSvD();
    return env->NewStringUTF(output.c_str());
}