////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "kernels.cuh"
#include "CustomTimer.hpp"

////////////////////////////////////////////////////////////////////////////////
#define MAX(a,b) ((a > b) ? a : b)
int iDivUp(int a, int b){   return (a % b != 0) ? (a / b + 1) : (a / b);}
float3 CameraPosition = make_float3(.0f,.0f,.0f);
float3 CameraVector = make_float3(1.0f, .0f, .0f);

typedef unsigned int uint;
typedef unsigned char uchar;

//const char* heightfieldfile = "BricksH.ppm";//"imtest.ppm";// "height.ppm"; //"flower.ppm";
//const char* colorfile = "BricksCol.ppm"; // "imtest.ppm";

// constants
unsigned int window_width = 1024;
unsigned int window_height = 1024;
int intensitySetting = 0;
int mode = 1;
int sceneNumber = 0;
dim3 block(16, 16, 1);
//dim3 block(32, 32, 1);
//unsigned int window_width = 100;
//unsigned int window_height = 100;
//dim3 block(1, 1, 1);
dim3 grid(iDivUp(window_width, block.x), iDivUp(window_height, block.y), 1);

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
//float rotate_x = 0.0, rotate_y = 0.0;
//float translate_z = 3.0;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0f, -2.0f, -8.0f);

StopWatchInterface* timer = NULL;

// fps
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;

int* pArgc = NULL;
char** pArgv = NULL;

CustomTimer timerC = CustomTimer();

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runProgram(int argc, char** argv, int scene, int mode);
void cleanup();
void initPixelBuffer();
void loadImageData();
void loadImageDataCube(int scene);
void idle();

// GL functionality
bool initGL(int* argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource** vbo_resource);

const char* sSDKsample = "simpleGL (VBO)";

bool checkHW(char* name, const char* gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    /*
    char* ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char**)argv, "file", (char**)&ref_file);
        }
    }

    printf("\n");
*/
    std::cout << "command line arguments: x y \nwhere x = {1,2} - scene number,\n y = {1,2,3} - mode: "<<
        "\n1 - visual ray marching, \n2 - \'sound\' raymarching, \n3 - \'sound\' cube raymarching\n";
    
    int sceneNumber = 1;
    //int mode = 1;
    if (argc > 1)
        sceneNumber = std::stoi(argv[1]);
    if (argc > 2)
        mode = std::stoi(argv[2]);
    sceneNumber = ((sceneNumber - 1) % 2) + 1;
    mode = ((mode - 1) % 3) + 1;
    runProgram(argc, argv, sceneNumber, mode);

}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
   
    if (frameCount % 100 == 1) {
        TimerInfo timerinfo = timerC.getInfo();
        std::cout << "mean: " << timerinfo.mean << " microseconds, Sigma = " << sqrt(timerinfo.dispersion)<<std::endl; 
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("test");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    initPixelBuffer();

    return true;
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, window_width * window_height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool runProgram(int argc, char** argv, int scene, int mode)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);


    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }
    //initCuda(h_volume, volumeSize);
    // run the cuda part
    //runCuda(&cuda_vbo_resource);

    //loadImageData();
    loadImageDataCube(scene);

    // start rendering mainloop
    glutMainLoop();

    return true;
}


template <class T>
bool LoadPPM4(const char* file, const char* filedepth, T** data, unsigned int* w, unsigned int* h);

#include "EXRLoader.h"

/*
void loadImageData()
{
    int imgWidth = 0;
    int imgHeight = 0;
    //uchar* imgData = NULL;
    float* imgData = NULL;
    const char* imgFilename = "images/0001.exr";// colorfile;
    const char* DepthFilename = heightfieldfile;

    //LoadPPM4(imgFilename, DepthFilename, (unsigned char**)&imgData, &imgWidth, &imgHeight);

    //readRgba(imgFilename, imgWidth, imgHeight, imgData);
    readGZ(imgFilename, imgWidth, imgHeight, imgData);

    if (!imgData)
    {
        printf("Error opening file '%s'\n", imgFilename);
        exit(EXIT_FAILURE);
    }

    printf("Loaded '%s', %d x %d pixels\n", imgFilename, imgWidth, imgHeight);

    Image img;
    img.size = make_cudaExtent(imgWidth, imgHeight, 0);
    img.h_data = imgData;
   // images.push_back(img);

    //initCuda(img);
}
*/
void loadImageDataCube(int scene)
{
    int imgWidth = 0;
    int imgHeight = 0;
    //uchar* imgData = NULL;
    //float* imgData[6];// = NULL;
    char* imgFilenames1[] = { "images/C0001.exr", "images/C0002.exr","images/C0003.exr","images/C0004.exr","images/C0005.exr","images/C0006.exr",
    "images/N0001.exr", "images/N0002.exr","images/N0003.exr","images/N0004.exr","images/N0005.exr","images/N0006.exr"};
    char* imgFilenames2[] = { "images2/C0001.exr", "images2/C0002.exr","images2/C0003.exr","images2/C0004.exr","images2/C0005.exr","images2/C0006.exr",
    "images2/N0001.exr", "images2/N0002.exr","images2/N0003.exr","images2/N0004.exr","images2/N0005.exr","images2/N0006.exr" };
    //ImageCube imgcube;
    std::vector<Image> images;

    char** filenames;
    if (scene == 1) filenames = imgFilenames1;
    if (scene == 2) filenames = imgFilenames2;

    for (int i=0;i<12;++i){
        int imgWidth = 0;
        int imgHeight = 0;
        float* imgData=nullptr;
        readGZ(filenames[i], imgWidth, imgHeight, imgData, (i>=6));

        if (!imgData)
        {
            printf("Error opening file '%s'\n", filenames[i]);
            exit(EXIT_FAILURE);
        }

        printf("Loaded '%s', %d x %d pixels\n", filenames[i], imgWidth, imgHeight);

        Image img;
        img.size = make_cudaExtent(imgWidth, imgHeight, 0);
        img.h_data = imgData;
        images.push_back(img);
    }
    //initCuda(img);
    initCuda(images);
}

void runCuda()
{
    // map PBO to get CUDA device pointer
    uchar4* d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, window_width * window_height * 4));

    float modelviewtmp[16];
    //float projectviewtmp[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelviewtmp);
    //glGetFloatv(GL_PROJECTION_MATRIX, projectviewtmp);
    glm::mat4 modelview = glm::make_mat4(modelviewtmp);
    //glm::mat4 projectview = glm::make_mat4(projectviewtmp);
    //glm::mat4 inv_matrix = modelview * projectview;
    glm::mat4 inv_matrix = glm::transpose(modelview);
    //inv_matrix = glm::inverse(inv_matrix);

    //launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
    // execute the kernel
    copyInvViewMatrix((float*)glm::value_ptr(inv_matrix), 12*sizeof(float));
//    copyInvViewMatrix(inverseviewtmp, 3 * sizeof(float4));
    //raymarchkernel << < grid, block >> > (d_output, window_width, window_height);
    raymarchkernelStart(d_output, window_width, window_height,grid, block, nullptr, intensitySetting, mode);
    cudaDeviceSynchronize();

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);
    // run CUDA kernel to generate vertex positions


    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

   // gluLookAt(  CameraPosition.x, CameraPosition.y, CameraPosition.z,
   //             CameraPosition.x + CameraVector.x, CameraPosition.y + CameraVector.y, CameraPosition.z + CameraVector.z,
   //             0.0f, 1.0f, 0.0f);

    
    //glRotatef(rotate_x, 1.0, 0.0, 0.0);
    //glRotatef(rotate_y, 0.0, 1.0, 0.0);
    //glTranslatef(0.0, 0.0, translate_z);
    glRotatef(-viewRotation.x, 0.0, 1.0, 0.0);
    glRotatef(-viewRotation.y, 1.0, 0.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    //glTranslatef(0, 0, -1);
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();

    timerC.tic();
    runCuda();
    timerC.toc();

    glPopMatrix();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // draw using texture

// copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    //freeCudaBuffers();
    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

void idle()
{
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27):
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
        break;
    case '=':
    case '+':
        ++intensitySetting;
        break;
    case '-':
        --intensitySetting;
        break;
    case '1':
        mode = 1;
        break;
    case '2':
        mode = 2;
        break;
    case '3':
        mode = 4;
        break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dx / 5.0f;
        viewRotation.y += dy / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    // предупредим деление на ноль
        // если окно сильно перетянуто будет
        if (y == 0)
            y = 1;
    float ratio = 1.0f * x / y;
    window_width = x;
    window_height = y;

    initPixelBuffer();

    grid = dim3(iDivUp(window_width, block.x), iDivUp(window_height, block.y));

    // определяем окно просмотра
    glViewport(0, 0, x, y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // используем матрицу проекции
    glMatrixMode(GL_PROJECTION);
    // Reset матрицы
    glLoadIdentity();

    // установить корректную перспективу.
    //gluPerspective(60.0, ratio, 0.1, 10.0);
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    //glm::mat4 proj = glm::perspective(60.0f, ratio, 0.1f, 10.0f);
    
    // вернуться к модели
    //glMatrixMode(GL_MODELVIEW);

}


template <class T>
bool LoadPPM4(const char* file, const char* filedepth, T** data, unsigned int* w, unsigned int* h)
{
    unsigned char* idata = 0;
    unsigned char* idataD = 0;
    unsigned int channels;
    unsigned int channelsD;

    if (__loadPPM(file, &idata, w, h, &channels)  && __loadPPM(filedepth, &idataD, w, h, &channelsD)) {
        // pad 4th component
        int size = *w * *h;
        // keep the original pointer
        unsigned char* idata_orig = idata;
        unsigned char* idata_origD = idataD;
        *data = reinterpret_cast<T*>(malloc(sizeof(T) * size * 4));
        unsigned char* ptr = *data;

        for (int i = 0; i < size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idataD;
            idataD += 3;
        }

        free(idata_orig);
        free(idata_origD);
        return true;
    }
    else {
        free(idata);
        free(idataD);
        return false;
    }
}
