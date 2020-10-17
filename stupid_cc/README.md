# TF-Lite C++ Custom Ops

### Get Android NDK

Download `android-ndk-r21d-linux-x86_64.zip` from https://developer.android.com/ndk/downloads.

### Get Android SDK

This is part of android studio. Default path is `~/Android/Sdk`.

### Checkout Tensorflow 2.3

```bash
git clone https://github.com/tensorflow/tensorflow.git tensorflow
cd tensorflow
git checkout r2.3
```

### Get Bazel and add it to `$PATH`.

```bash
git clone https://github.com/bazelbuild/bazelisk.git bazelisk
ln -s bazelisk.py bazelisk/bazel
export "PATH=$PATH:$PWD/bazelisk"
```

### Configure:

```bash
$ python3 -m venv venv
source venv/bin/activate
./configure
```

```
You have bazel 3.1.0 installed.
Please specify the location of python. [Default is /home/thomas/projects/tensorflow/venv/bin/python3]: 


Found possible Python library paths:
  /home/thomas/projects/tensorflow/venv/lib/python3.8/site-packages
Please input the desired Python library path to use.  Default is [/home/thomas/projects/tensorflow/venv/lib/python3.8/site-packages]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: 
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: 
Clang will not be downloaded.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: y
Searching for NDK and SDK installations.

Please specify the home path of the Android NDK to use. [Default is /home/thomas/Android/Sdk/ndk-bundle]: /home/thomas/projects/android-ndk-r21d


WARNING: The NDK version in /home/thomas/projects/android-ndk-r21d is 21, which is not supported by Bazel (officially supported versions: [10, 11, 12, 13, 14, 15, 16, 17, 18]). Please use another version. Compiling Android targets may result in confusing errors.

Please specify the (min) Android NDK API level to use. [Available levels: ['16', '17', '18', '19', '21', '22', '23', '24', '26', '27', '28', '29', '30']] [Default is 21]:   


Please specify the home path of the Android SDK to use. [Default is /home/thomas/Android/Sdk]: 


Please specify the Android SDK API level to use. [Available levels: ['29']] [Default is 29]: 


Please specify an Android build tools version to use. [Available versions: ['29.0.2']] [Default is 29.0.2]: 
```

### Build TF Lite

```
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```