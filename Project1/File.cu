//
// Sharpening Filter using CUDA
//

/*
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"


using namespace std;
using namespace cv;

#define BLOCK_SIZE      16
#define FILTER_HEIGHT    3       
#define FILTER_WIDTH   3     

// Run Sharpening Filter on GPU
__global__ void sharpeningFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    // only threads inside image will write results
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        for (int c = 0; c < channel; c++)
        {
            // Sum of pixel values 
            float sum = 0;
            // Loop inside the filter to average pixel values
            for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
                for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                    float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
                    sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
                }
            }
            dstImage[(y * width + x) * channel + c] = min(max(int(sum), 0), 255);;
        }
    }
}

// The wrapper is used to call sharpening filter 
void sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
    // Use cuda event to catch time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calculate number of image channels
    int channel = input.step / input.cols;

    // Calculate number of input & output bytes in each block
    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    unsigned char* d_input, * d_output;

    // Allocate device memory
    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    // Specify block size
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid size to cover the whole image
    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    // Start time
    cudaEventRecord(start);

    // Run BoxFilter kernel on CUDA 
    sharpeningFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

    // Stop time
    cudaEventRecord(stop);

    //Copy data from device memory to output image
    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    //Free the device memory
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    // Calculate elapsed time in milisecond  
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}


// Program main
int main(int argc, char** argv) {

    // name of image
    string image_name = "brdo";

    // input & output file names
    string input_file = image_name + ".png";
    string output_file_cpu = image_name + "_cpu.png";
    string output_file_gpu = image_name + "_gpu.png";

    // Read input image 
    Mat srcImage = cv::imread(input_file);
    if (srcImage.empty())
    {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return -1;
    }
    cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";
    namedWindow("a", WINDOW_AUTOSIZE);
    imshow("a", srcImage);
    // Declare the output image  
    Mat dstImage(srcImage.size(), srcImage.type());

    // run median filter on GPU  
   sharpeningFilter_GPU_wrapper(srcImage, dstImage);
    // Output image
    imwrite(output_file_gpu, dstImage);
    namedWindow("ab", WINDOW_AUTOSIZE);
    imshow("ab", dstImage);
 
    waitKey(0);
    return 0;
}

*/


