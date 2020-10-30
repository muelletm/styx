
#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


cmake_minimum_required(VERSION 3.4.1)

# configure import libs
set(distribution_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../distribution)
set(HOME_DIR                $ENV{HOME})
set(TF_DIR                  ${HOME_DIR}/projects/tensorflow)
set(FLATBUFFER_DIR          ${HOME_DIR}/projects/flatbuffers/include)
set(EIGEN_DIR          ${HOME_DIR}/projects/eigen-3.3.8)
set(TFLITE_LIB_DIR          ${CMAKE_CURRENT_SOURCE_DIR}/tflite/lib/${ANDROID_ABI})
set(RSVD ${HOME_DIR}/projects/rsvd/include)

include_directories(
${FLATBUFFER_DIR}
${TF_DIR}
${EIGEN_DIR}
${RSVD}
)

link_directories(
        ${TFLITE_LIB_DIR}
)
find_library(libtensorflowlite tensorflowlite)

# build application's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_library(styx SHARED timing.h jni_utils.h svd_op.h timing.cpp jni_utils.cpp svd_op.cpp big4321_dq.h big4321_dq.cpp model.cpp)

target_link_libraries(styx
                      android
                      log
                      tensorflowlite)
