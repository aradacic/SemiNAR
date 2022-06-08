#include <iostream>
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include "device_launch_parameters.h"
const int N = 10;

using namespace std;
using namespace cv;

#define filterWidth 3
#define filterHeight 3



//kernel: funkcija koju izvtrsavaju niti graficke kartice
__global__ void sharpenAMM(unsigned char* dev_input, unsigned char* dev_output, double** dev_filter, int width, int height, int origigiStep, int istoOrigigi) {

    //int tid = threadIdx.x;
    //int bid = blockIdx.x;

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  
        const int color_tid = yIndex * origigiStep + (3 * xIndex);

        const unsigned char blue = dev_input[color_tid];
        const unsigned char green = dev_input[color_tid + 1];
        const unsigned char red = dev_input[color_tid + 2];

        int b = 0;
        int g = 0;
        int r = 0;

        if (color_tid >= width * 3 && color_tid <= height * 3 * width)
        {
            for (int filterY = 0; filterY < filterHeight; filterY++) {
                for (int filterX = 0; filterX < filterWidth; filterX++) {
                    int imageX = (xIndex - filterWidth / 2 + filterX + width) % width;
                    int imageY = (yIndex - filterHeight / 2 + filterY + height) % height;
                    //cout << imageX << " " << imageY << " " << filterX << " " << filterY<<endl;
                    b += blue * dev_filter[filterX][filterY];
                    g += green * dev_filter[filterX][filterY];
                    r += red * dev_filter[filterX][filterY];
                    //b = 0;
                    //g = 0;
                    //r = 255;
                    printf("%f\n", dev_filter[2][2]);
                }
            }
        }
       
        dev_output[color_tid] = b;//min(max(int(b), 0), 255);
        dev_output[color_tid + 1] = g;// min(max(int(g), 0), 255);
        dev_output[color_tid + 2] = r;// min(max(int(r), 0), 255);

        
       
        
 
   
    
    //printf("% d\n", bid);

}


int main() {
    //inicijaliziramo sat da možemo pratiti vrijeme izvrsavanja
    std::clock_t start;
    double duration;
    start = std::clock();

    //ime slike
    string image_name = "blur_kernel.jpg";
    Mat image = imread(image_name);
    Mat image_sharpen = imread(image_name);


    //dimenzije slike
    int width = image.cols;
    int height = image.rows;

    //filter
    double filter[3][3] =
    {
      0, -1,  0,
     -1,  5, -1,
      0, -1,  0,
    };


    unsigned char* dev_input, * dev_output;//pokazivaci na memoriju na grafickoj kartici
    double **dev_filter;

    const int colorBytes = image.step * image.rows;
    
    cudaMalloc<unsigned char>(&dev_input, colorBytes);
    cudaMalloc<unsigned char>(&dev_output, colorBytes);
    cudaMalloc(&dev_filter, 9 * sizeof(double));
    

    // Create a window for display.
    namedWindow("a", WINDOW_AUTOSIZE);
    imshow("a", image);

    cudaMemcpy(dev_input, image.ptr(), colorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter, filter, 9 * sizeof(double), cudaMemcpyHostToDevice);
    


    

    //int image_arr[height][width][3];
   /* int* image_arr1[512];
    for(int i = 0; i < 512; i++)
        image_arr1[i] = (int*)malloc(sizeof(int) * height * width);
    int* image_arr2[512];
    for (int i = 0; i < 512; i++)
        image_arr2[i] = (int*)malloc(sizeof(int) * height * width);
    int* image_arr3[512];
    for (int i = 0; i < 512; i++)
        image_arr3[i] = (int*)malloc(sizeof(int) * height * width);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            image_arr1[i][j] = image.at<Vec3b>(i, j)[0];
            image_arr2[i][j] = image.at<Vec3b>(i, j)[1];
            image_arr3[i][j] = image.at<Vec3b>(i, j)[2];
            
        }
    }
    */

    

    //alociranje memorije na grafickoj kartici
    //cudaMalloc((void**)&dev_red, width * height * sizeof(int));
    //cudaMalloc((void**)&dev_green, width * height * sizeof(int));
    //cudaMalloc((void**)&dev_blue, width * height* sizeof(int));
    //cudaMalloc((void**)&dev_filter, 9 * sizeof(double));

    //popunjavanje nizova a i b u glavnoj memoriji
   

    
    //kopiranje vrijednosti sa glavne memorije na memoriju grafièke kartice
    //cudaMemcpy(dev_red, image_arr3, width * height * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_green, image_arr2, width * height * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_blue, image_arr1, width * height * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_filter, filter, 9 * sizeof(double), cudaMemcpyHostToDevice);

    

    //poziv kernela
    ///<<< broj blokova, broj niti po bloku>>>
    sharpenAMM <<< height, width >>> (dev_input, dev_output, dev_filter, image.cols, image.rows, image.step, image_sharpen.step);

    cudaDeviceSynchronize();

    cudaMemcpy(image_sharpen.ptr(), dev_output, colorBytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
    //cudaMemcpy(image_arr3, dev_red, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(image_arr2, dev_green, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(image_arr1, dev_blue, width * height * sizeof(int), cudaMemcpyDeviceToHost);


    //cudaFree(dev_red);
    //cudaFree(dev_green);
    //cudaFree(dev_blue);
    //cudaFree(dev_filter);

    namedWindow("ab", WINDOW_AUTOSIZE);
    imshow("ab", image_sharpen);


    //sat 
    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "Time  taken " << duration << '\n';

    waitKey(0);//za cekanje
    return 0;

}


