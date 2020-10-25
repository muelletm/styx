#include "svd_op.h"

#include <android/log.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include "rsvd/RandomizedSvd.hpp"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

#include "timing.h"

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "svd_op::", __VA_ARGS__))

#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "svd_op::", __VA_ARGS__))


namespace tflite {
    namespace ops {
        namespace custom {
            static int svd_rank_;

            void setSvdRank(int svd_rank) {
                svd_rank_ = svd_rank;
            }

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

            void CopyVector(const Eigen::VectorXf &vector, float *data) {
                for (int i = 0; i < vector.size(); i++) {
                    data[i] = (float) vector(i);
                }
            }

            void CopyMatrix(const Eigen::MatrixXf &matrix,
                            const std::vector<int> &input_shape,
                            float *data) {
                for (int r = 0; r < matrix.rows(); r++) {
                    for (int c = 0; c < matrix.cols(); c++) {
                        int index = r * input_shape[0] + c;
                        data[index] = (float) matrix(r, c);
                    }
                }
            }

            TfLiteStatus SvdEval(TfLiteContext *context, TfLiteNode *node) {
                LOGI("SvdEval");
                long start_time = currentTimeMillis();
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
                const int max_rank = std::min(input_eigen.cols(), input_eigen.rows());
                int rank = svd_rank_;
                if (rank >= max_rank) {
                    LOGI(
                            "Setting rank to full rank (svd rank %d, max rank: %d).",
                            svd_rank_,
                            max_rank);
                    rank = -1;
                }
                if (rank == -1) {
                    Eigen::BDCSVD <Eigen::MatrixXf> svd(input_eigen,
                                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
                    const Eigen::VectorXf &s_eigen = svd.singularValues();
                    CopyVector(s_eigen, s_data);
                    const Eigen::MatrixXf &u_eigen = svd.matrixU();
                    CopyMatrix(u_eigen, input_shape, u_data);
                    const Eigen::MatrixXf &v_eigen = svd.matrixV();
                    CopyMatrix(v_eigen, input_shape, v_data);
                } else {
                    std::mt19937_64 randomEngine{};
                    randomEngine.seed(777);
                    Rsvd::RandomizedSvd <Eigen::MatrixXf, std::mt19937_64, Rsvd::SubspaceIterationConditioner::Lu> svd(
                            randomEngine);
                    svd.compute(input_eigen, rank);
                    std::fill(s_data, s_data + NumElements(s), 0.0f);
                    std::fill(u_data, u_data + NumElements(u), 0.0f);
                    std::fill(v_data, v_data + NumElements(v), 0.0f);
                    const Eigen::VectorXf s_eigen = svd.singularValues();
                    CopyVector(s_eigen, s_data);
                    const Eigen::MatrixXf u_eigen = svd.matrixU();
                    CopyMatrix(u_eigen, input_shape, u_data);
                    const Eigen::MatrixXf v_eigen = svd.matrixV();
                    CopyMatrix(v_eigen, input_shape, v_data);
                }
                LOGI("SvdEval (took %ldms)", currentTimeMillis() - start_time);
                return kTfLiteOk;
            }

            TfLiteRegistration *Register_SVD() {
                LOGI("Register SVD");
                static TfLiteRegistration r = {nullptr, nullptr, SvdPrepare, SvdEval};
                return &r;
            }

        }  // namespace custom
    }  // namespace ops
}  // namespace tflite