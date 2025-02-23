/**********************************************************************************/
/* Copyright (c) 2025 Markus Geiger                                               */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <signal.h>
#include <time.h>
#include <windows.h>
#include <vfw.h>
#include <conio.h>

#include "..\\..\\mbednn\\src\\mbednn.h"

#define TRAINING
#define REAL_VIDEO

#define ROWS               128ul
#define COLS               128ul
#define CLASSES            2ul
#define RAW_FILENAME       "snapshot.raw"

mbednn_t *mbednn;
char     full_path[MAX_PATH];
real     outputs[CLASSES];
char     *train_path = ".\\PetImages\\Train128x128";
uint32_t pred_class;
HWND     hWndCap;
BOOL     isCameraConnected = FALSE;
BOOL     saveNextFrame = FALSE;

void signal_handler(int signal)
{
    if (signal == SIGFPE)
    {
        printf("Caught SIGFPE: Floating-point exception occurred!\n");
    }
}

int64_t get_time_us(void)
{
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;

    // get frequency
    QueryPerformanceFrequency(&freq);
    // get current time
    QueryPerformanceCounter(&counter);
    // convert to microseconds
    return (counter.QuadPart * 1000000LL) / freq.QuadPart;
}

void print_current_time(void)
{
    // get the current time
    time_t current_time;
    struct tm* time_info;

    time(&current_time);
    time_info = localtime(&current_time);

    // print the time in a readable format
    printf("Current time: %s\n", asctime(time_info));
}

// clamp a value to 0-255 range
static inline uint8_t clamp(int value)
{
    return (value < 0) ? 0 : (value > 255) ? 255 : value;
}

// convert YUY2 to RGB24
void YUY2toRGB24(const uint8_t* yuy2, uint8_t* rgb, int width, int height)
{
    int frameSize = width * height * 2;  // YUY2 uses 2 bytes per pixel

    for (int i = 0, j = 0; i < frameSize; i += 4, j += 6)
    {
        // extract YUY2 components
        uint8_t Y0 = yuy2[i + 0];
        uint8_t U = yuy2[i + 1];
        uint8_t Y1 = yuy2[i + 2];
        uint8_t V = yuy2[i + 3];

        // convert YUV to RGB
        int C0 = Y0 - 16;
        int C1 = Y1 - 16;
        int D = U - 128;
        int E = V - 128;

        // YUV to RGB conversion formulas
        int R0 = (298 * C0 + 409 * E + 128) >> 8;
        int G0 = (298 * C0 - 100 * D - 208 * E + 128) >> 8;
        int B0 = (298 * C0 + 516 * D + 128) >> 8;

        int R1 = (298 * C1 + 409 * E + 128) >> 8;
        int G1 = (298 * C1 - 100 * D - 208 * E + 128) >> 8;
        int B1 = (298 * C1 + 516 * D + 128) >> 8;

        // store RGB24 (RGB format)
        rgb[j + 0] = clamp(R0);
        rgb[j + 1] = clamp(G0);
        rgb[j + 2] = clamp(B0);

        rgb[j + 3] = clamp(R1);
        rgb[j + 4] = clamp(G1);
        rgb[j + 5] = clamp(B1);
    }
}

// crop 640x480 RGB24 image to center 480x480
void cropRGB24_640x480_to_480x480(const uint8_t* src, uint8_t* dst)
{
    int srcWidth = 640, dstWidth = 480;
    int xOffset = (srcWidth - dstWidth) / 2;   // 80 pixels removed from each side

    for (int y = 0; y < 480; y++)
    {
        int srcIndex = (y * srcWidth + xOffset) * 3;          // source row start
        int dstIndex = (y * dstWidth) * 3;                    // destination row start

        memcpy(&dst[dstIndex], &src[srcIndex], dstWidth * 3); // copy one row
    }
}

// resizes an RGB24 image to 128x128 using nearest-neighbor
void resizeRGB24(const uint8_t* src, uint8_t* dst, int srcWidth, int srcHeight)
{
    const int dstWidth = 128, dstHeight = 128;
    float x_ratio = (float)srcWidth / dstWidth;
    float y_ratio = (float)srcHeight / dstHeight;

    for (int y = 0; y < dstHeight; y++)
    {
        for (int x = 0; x < dstWidth; x++)
        {
            // nearest-neighbor scaling
            int srcX = (int)(x * x_ratio);
            int srcY = (int)(y * y_ratio);

            int srcIndex = (srcY * srcWidth + srcX) * 3;  // RGB24 (3 bytes per pixel)
            int dstIndex = (y * dstWidth + x) * 3;

            dst[dstIndex + 0] = src[srcIndex + 0]; // R
            dst[dstIndex + 1] = src[srcIndex + 1]; // G
            dst[dstIndex + 2] = src[srcIndex + 2]; // B
        }
    }
}

// callback function to receive the video frame
LRESULT CALLBACK FrameCallback(HWND hWnd, LPVIDEOHDR lpVHdr)
{
    if (saveNextFrame == TRUE)
    {
        saveNextFrame = FALSE;

        if (lpVHdr && lpVHdr->lpData)
        {
            int rawSize = 640 * 480 * 3;
            FILE *file = fopen(RAW_FILENAME, "wb");
            if (file != NULL)
            {
                uint8_t *rgbBuffer = malloc(rawSize);
                uint8_t *croppedBuffer = malloc(480 * 480 * 3);
                uint8_t *finalBuffer = malloc(128 * 128 * 3);

                YUY2toRGB24(lpVHdr->lpData, rgbBuffer, 640, 480);
                cropRGB24_640x480_to_480x480(rgbBuffer, croppedBuffer);
                resizeRGB24(croppedBuffer, finalBuffer, 480, 480);

                if (finalBuffer != NULL)
                {
                    fwrite(finalBuffer, 1, 128 * 128 * 3, file);
                }
                fclose(file);
                free(finalBuffer);
                free(croppedBuffer);
                free(rgbBuffer);
                printf("Snapshot saved as %s (RAW 24-bit RGB/YUV data)\n", RAW_FILENAME);

                // make a prediction of the snapshot
                mbednn_predict_file(mbednn, 1ul, RAW_FILENAME, outputs);
                pred_class = mbednn_class_predict(outputs, CLASSES);
                printf("Predicted class is (0 - dog, 1 - cat): %ld\n", pred_class);
            }
            else
            {
                printf("Error: Failed to save raw file!\n");
            }
        }
    }

    return (LRESULT)TRUE;
}

// function to set up live video streaming with frame capture
void SetupCamera(HWND hWnd)
{
    hWndCap = capCreateCaptureWindow(L"Capture Window", WS_CHILD | WS_VISIBLE, 0, 0, 640, 480, hWnd, 1);
    if (!hWndCap)
    {
        printf("Error: Failed to create capture window!\n");
        return;
    }

    if (capDriverConnect(hWndCap, 0))
    {
        capPreviewScale(hWndCap, TRUE);
        capPreviewRate(hWndCap, 33);
        capPreview(hWndCap, TRUE);
        isCameraConnected = TRUE;

        // set frame callback to grab raw data
        capSetCallbackOnFrame(hWndCap, FrameCallback);

        printf("Live video started! Press 'S' to capture RAW frame, 'Q' to quit.\n");
    }
    else
    {
        printf("Error: No webcam found!\n");
        DestroyWindow(hWnd);
    }
}

// windows procedure
LRESULT CALLBACK WinProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
        SetupCamera(hWnd);
        break;
    case WM_DESTROY:
        if (isCameraConnected)
        {
            capDriverDisconnect(hWndCap);
        }
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

// function to create and display the webcam window
DWORD WINAPI StartVideoWindow(LPVOID lpParam)
{
    MSG msg;
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WinProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = L"VideoCaptureClass";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    if (!RegisterClass(&wc))
    {
        printf("Error: Could not register window class!\n");
        return 1;
    }

    HWND hWndMain = CreateWindow(L"VideoCaptureClass", L"Live Video - Console App",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        640, 520, NULL, NULL, wc.hInstance, NULL);

    if (!hWndMain)
    {
        printf("Error: Could not create main window!\n");
        return 1;
    }

    ShowWindow(hWndMain, SW_SHOW);
    UpdateWindow(hWndMain);

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

int get_prefix_number_from_filename(char* filename)
{
    char filename_copy[260];
    int prefix_number;
    char* underscore_pos;

    strcpy(filename_copy, filename);
    underscore_pos = strchr(filename_copy, '_');

    if (underscore_pos != NULL)
    {
        // isolate the prefix
        *underscore_pos = '\0';

        // convert the prefix (before the underscore) to an integer
        prefix_number = atoi(filename_copy);
    }
    else
    {
        prefix_number = -1;
    }

    return prefix_number;
}

real epoch_callback(void)
{
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind;
    char search_path[MAX_PATH];
    uint32_t datasets = 0ul;
    uint32_t correct = 0ul;
    uint32_t act_class;
    real accuracy;

    snprintf(search_path, MAX_PATH, "%s\\*", train_path);
    hFind = FindFirstFileA(search_path, &findFileData);
    if (hFind == INVALID_HANDLE_VALUE)
    {
        printf("Error: Unable to open directory\n");
        return REAL_C(0.0);
    }

    do
    {
        if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            datasets++;
            snprintf(full_path, MAX_PATH, "%s\\%s", train_path, findFileData.cFileName);
            mbednn_predict_file(mbednn, 1ul, full_path, outputs);
            pred_class = mbednn_class_predict(outputs, CLASSES);

            // first character of file name is the class
            act_class = (uint32_t)get_prefix_number_from_filename(findFileData.cFileName);

            if (pred_class == act_class)
            {
                correct++;
            }
        }
    }
    while (FindNextFileA(hFind, &findFileData) != 0);

    accuracy = (real)correct * REAL_C(100.0) / (real)datasets;
    printf(" Accuracy=%7.3f%%\n", accuracy);

    return accuracy;
}

int main(int argc, char* argv[])
{
#ifdef TRAINING
    real loss;

    // floating point exception activation
    signal(SIGFPE, signal_handler);
    unsigned int control_word;
    _controlfp_s(&control_word, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW, _MCW_EM);

    // create a new empty fully connected neural network
    mbednn = mbednn_create();

    // define network layers
    (void)mbednn_add_layer_input_2d(mbednn, ROWS, COLS, 3ul);
    (void)mbednn_add_layer_conv_2d(mbednn, 8ul, 3ul, 3ul, 1ul, MBEDNN_ACTIVATION_RELU);
    (void)mbednn_add_layer_maxpooling_2d(mbednn);
    (void)mbednn_add_layer_conv_2d(mbednn, 16ul, 3ul, 3ul, 1ul, MBEDNN_ACTIVATION_RELU);
    (void)mbednn_add_layer_maxpooling_2d(mbednn);
    (void)mbednn_add_layer_conv_2d(mbednn, 32ul, 3ul, 3ul, 1ul, MBEDNN_ACTIVATION_RELU);
    (void)mbednn_add_layer_maxpooling_2d(mbednn);
    (void)mbednn_add_layer_flatten_2d(mbednn);
#if 0
    (void)mbednn_add_layer_dropout(mbednn, REAL_C(0.2));
#endif
    (void)mbednn_add_layer_dense(mbednn, 512ul, MBEDNN_ACTIVATION_RELU, MBEDNN_RANDOM_UNIFORM);
    (void)mbednn_add_layer_output(mbednn, CLASSES, MBEDNN_ACTIVATION_SOFTMAX, MBEDNN_RANDOM_UNIFORM);

    // print the network summary
    mbednn_summary(mbednn);

    // define network hyper parameters
    mbednn_compile(mbednn, MBEDNN_OPTIMIZER_DEFAULT, MBEDNN_LOSS_CATEGORICAL_CROSS_ENTROPY, REAL_C(0.0001), REAL_C(0.01));

#if 0
    // define user filters, e.g. Sobel, Gaussian, Laplacian
    // filters not explicitely defined here are intialized with random values
    real filter_values_sobel1[] =    { REAL_C(-1.0), REAL_C( 0.0), REAL_C( 1.0),
                                       REAL_C(-2.0), REAL_C( 0.0), REAL_C( 2.0),
                                       REAL_C(-1.0), REAL_C( 0.0), REAL_C( 1.0) };

    real filter_values_sobel2[] =    { REAL_C( 1.0), REAL_C( 2.0), REAL_C( 1.0),
                                       REAL_C( 0.0), REAL_C( 0.0), REAL_C( 0.0),
                                       REAL_C(-1.0), REAL_C(-2.0), REAL_C(-1.0) };

    real filter_values_gauss[] =     { REAL_C( 1.0), REAL_C( 2.0), REAL_C( 1.0),
                                       REAL_C( 2.0), REAL_C( 4.0), REAL_C( 2.0),
                                       REAL_C( 1.0), REAL_C( 2.0), REAL_C( 1.0) };

    real filter_values_laplacian[] = { REAL_C(-1.0), REAL_C(-1.0), REAL_C(-1.0),
                                       REAL_C(-1.0), REAL_C( 8.0), REAL_C(-1.0),
                                       REAL_C(-1.0), REAL_C(-1.0), REAL_C(-1.0) };

    mbednn_set_filter(mbednn, 1ul, 0ul, 0ul, filter_values_laplacian);
    mbednn_set_filter(mbednn, 1ul, 0ul, 1ul, filter_values_laplacian);
    mbednn_set_filter(mbednn, 1ul, 0ul, 2ul, filter_values_laplacian);

    mbednn_set_filter(mbednn, 1ul, 1ul, 0ul, filter_values_sobel1);
    mbednn_set_filter(mbednn, 1ul, 1ul, 1ul, filter_values_sobel1);
    mbednn_set_filter(mbednn, 1ul, 1ul, 2ul, filter_values_sobel1);

    mbednn_set_filter(mbednn, 1ul, 2ul, 0ul, filter_values_sobel2);
    mbednn_set_filter(mbednn, 1ul, 2ul, 1ul, filter_values_sobel2);
    mbednn_set_filter(mbednn, 1ul, 2ul, 2ul, filter_values_sobel2);

    mbednn_set_filter(mbednn, 1ul, 3ul, 0ul, filter_values_gauss);
    mbednn_set_filter(mbednn, 1ul, 3ul, 1ul, filter_values_gauss);
    mbednn_set_filter(mbednn, 1ul, 3ul, 2ul, filter_values_gauss);
#endif

    // print the current time
    print_current_time();

    // train the network
    loss = mbednn_fit_files(mbednn, 100ul, 32ul, 1ul, train_path, epoch_callback);
    printf("Training loss: %7.3f\n\n", loss);

    // print the current time
    print_current_time();

    // test the network
    mbednn_predict_file(mbednn, 1ul, ".\\PetImages\\Test\\dog.raw", outputs);
    pred_class = mbednn_class_predict(outputs, CLASSES);
    printf("Predicted class of the dog (0) is: %ld\n", pred_class);
    mbednn_predict_file(mbednn, 1ul, ".\\PetImages\\Test\\cat.raw", outputs);
    pred_class = mbednn_class_predict(outputs, CLASSES);
    printf("Predicted class of the cat (1) is: %ld\n", pred_class);

    // save the neural network into binary file
    mbednn_save_binary(mbednn, ".\\net\\net.bin");
#else
#ifdef REAL_VIDEO
    mbednn = mbednn_load_binary(".\\net\\net.bin");

    // start the video window in a separate thread
    HANDLE hThread = CreateThread(NULL, 0, StartVideoWindow, NULL, 0, NULL);
    if (!hThread)
    {
        printf("Error: Could not create video window thread!\n");
        return 1;
    }

    // console loop: wait for key input
    while (1)
    {
        if (_kbhit())
        {
            // check if a key is pressed
            char ch = _getch();
            if ((ch == 's') || (ch == 'S'))
            {
                // set flag to save next frame
                saveNextFrame = TRUE;
            }
            else if ((ch == 'q') || (ch == 'Q'))
            {
                printf("Exiting program...\n");
                PostThreadMessage(GetThreadId(hThread), WM_QUIT, 0, 0);
                WaitForSingleObject(hThread, INFINITE);
                CloseHandle(hThread);
                break;
            }
        }
        Sleep(10);
    }
#else
    int64_t start;
    int64_t end;

    mbednn = mbednn_load_binary(".\\net\\net.bin");

    start = get_time_us();
    mbednn_predict_file(mbednn, 1ul, ".\\PetImages\\Test\\dog.raw", outputs);
    pred_class = mbednn_class_predict(outputs, CLASSES);
    end = get_time_us();
    printf("Predicted class of the dog (0) is: %ld\n", pred_class);
    printf("Execution time: %lld us\n", (end - start));

    start = get_time_us();
    mbednn_predict_file(mbednn, 1ul, ".\\PetImages\\Test\\cat.raw", outputs);
    pred_class = mbednn_class_predict(outputs, CLASSES);
    end = get_time_us();
    printf("Predicted class of the cat (1) is: %ld\n", pred_class);
    printf("Execution time: %lld us\n", (end - start));
#endif
#endif

    // free the network
    mbednn_free(mbednn);

    return 0;
}
