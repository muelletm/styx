//
// Created by thomas on 25.10.20.
//

#ifndef STYX_CC_JNI_UTILS_H
#define STYX_CC_JNI_UTILS_H

#include <string>
#include <vector>
#include <jni.h>

long currentTimeMillis(void);

std::string jstringToString(JNIEnv *env, jstring jStr);

std::vector<float> toFloatVec(JNIEnv *env, jfloatArray input);

std::vector<int> toIntVec(JNIEnv *env, jintArray input);

std::vector<int>
getIntVecField(JNIEnv *env, const jobject &input, const std::string &field_name);

std::vector<float>
getFloatVecField(JNIEnv *env, const jobject &input, const std::string &field_name);

#endif //STYX_CC_JNI_UTILS_H
