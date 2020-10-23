Custom Ops
=========

Based on https://developer.android.com/studio/projects/add-native-code.

## Install instructions


* Import the project into Android Studio
* Clone flatbuffers and tflite set the paths in 'CMakeLists.txt'
  * `git clone https://github.com/google/flatbuffers.git flatbuffers`
  * `git clone https://github.com/tensorflow/tensorflow.git tensorflow`
* Download a content and style images
  * (into `stupid_android/styx_cc/app/src/main/res/drawable`)
  * Update the definition of `STYLES` in `MainActivity.java`
* Build the APK and install it on your phone
* Download the style transfer tflite model
  * [stupid_relu4.tflite](https://drive.google.com/drive/u/0/folders/1GWfb4EcM7-WFKCCM7ZHlimE_J6bS4ExW)
* Launch the app and give it permission to read storage files


Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements.  See the NOTICE file distributed with this work for
additional information regarding copyright ownership.  The ASF licenses this
file to you under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy of
the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations under
the License.

