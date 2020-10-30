#include "jni_utils.h"

std::string jstringToString(JNIEnv *env, jstring jStr) {
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

std::vector<int>
getIntVecField(JNIEnv *env, const jobject &input, const std::string &field_name) {
    jclass tensor = env->GetObjectClass(input);
    jfieldID fieldID = env->GetFieldID(tensor, field_name.c_str(), "[I");
    jobject object = env->GetObjectField(input, fieldID);
    jintArray *array = reinterpret_cast<jintArray *>(&object);
    return toIntVec(env, *array);
}

std::vector<float>
getFloatVecField(JNIEnv *env, const jobject &input, const std::string &field_name) {
    jclass tensor = env->GetObjectClass(input);
    jfieldID fieldID = env->GetFieldID(tensor, field_name.c_str(), "[F");
    jobject object = env->GetObjectField(input, fieldID);
    jfloatArray *array = reinterpret_cast<jfloatArray *>(&object);
    return toFloatVec(env, *array);
}