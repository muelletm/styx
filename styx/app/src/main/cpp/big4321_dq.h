/*
Based on
 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/model.cc
 https://www.tensorflow.org/lite/microcontrollers/build_convert

Created with:

xxd -i big4321_dq.tflite > styx/app/src/main/cpp/big4321_dq.cpp

*/

extern const unsigned int big4321_dq_length;
extern const unsigned char big4321_dq_model[];
