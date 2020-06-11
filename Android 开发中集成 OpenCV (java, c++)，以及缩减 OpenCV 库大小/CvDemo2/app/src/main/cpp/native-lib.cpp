#include <jni.h>
#include <string>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_cv_cvdemo2_MainActivity_convertToGray(JNIEnv *env, jobject thiz, jintArray img_data,
                                               jint width, jint height) {
    // TODO: implement convertToGray()
    jint* cbuf;
    cbuf = env->GetIntArrayElements(img_data, JNI_FALSE);

    Mat inp_img(height, width, CV_8UC4, (unsigned char *)cbuf);

    Mat gray_img;
    cvtColor(inp_img, gray_img, CV_BGRA2GRAY);

    Mat ret_img;
    cvtColor(gray_img, ret_img, CV_GRAY2BGRA);
    int size = width * height;
    jintArray result = env->NewIntArray(size);
    uchar *ptr = ret_img.data;
    env->SetIntArrayRegion(result, 0, size, (const jint *) ptr);
    env->ReleaseIntArrayElements(img_data, cbuf, 0);
    return result;
}