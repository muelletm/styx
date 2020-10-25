#ifndef STYX_CC_SVD_OP_H
#define STYX_CC_SVD_OP_H

#include "tensorflow/lite/context.h"

namespace tflite {
    namespace ops {
        namespace custom {

            void setSvdRank(int svd_rank);

            TfLiteRegistration *Register_SVD();

        }  // namespace custom
    }  // namespace ops
}  // namespace tflite

#endif //STYX_CC_SVD_OP_H
