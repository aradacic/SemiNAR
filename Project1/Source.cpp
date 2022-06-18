

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

#define filterWidth 3
#define filterHeight 3

double filter[filterHeight][filterWidth] =
{
  0, -1,  0,
 -1,  5, -1,
  0, -1,  0,
};

int resizeDimension(Mat);
void showPicture(string, Mat);
Mat sharpening(Mat, Mat);

int main(int argc, char** argv)
{
    int input = 0;
    string image_name = "";

    cout << "Choose a picture to sharpen\n1. brdo.png \n2. test.jpg \n3. blur_kernel.jpg \n4. trava.png \n5. tica.jpg \n6. lines.jpg\n7. mini.jpg\n";
    cout << "Picture: ";
    cin >> input;

    switch (input)
    {
        case 1:
            image_name = "brdo.png";
            break;
        case 2:
            image_name = "test.jpg";
            break;
        case 3:
            image_name = "blur_kernel.jpg";
            break;
        case 4:
            image_name = "trava.png";
            break;
        case 5:
            image_name = "tica.jpg";
            break;
        case 6:
            image_name = "lines.jpg";
            break;
        case 7:
            image_name = "mini.jpg";
            break;
        case 8:
            image_name = "blur_min.png";
            break;
        case 9:
            image_name = "edge.png";
            break;
        default:
            cout << "Invalid input";
            return 0;
            break;
    }
    
    //taking an image matrix
    //loading an image//
    Mat image = imread(image_name);
    Mat image_original = imread(image_name);

    //int original_width = image.cols;
    //int original_height = image.rows;

    auto start = chrono::high_resolution_clock::now();
    image = sharpening(image, image_original);
    auto end = chrono::high_resolution_clock::now();

    // Calculating total time taken by the program.
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    time_taken *= 1e-6;

    cout << "Time taken by program is : " << fixed << time_taken << setprecision(9) << " ms\n\n\n\n" << endl;
    
    //showPicture("Mini window", image);

    //resize(image, image, Size(original_width, original_height), INTER_AREA);
    //resize(image_original, image_original, Size(original_width, original_height), INTER_AREA);

    showPicture("Display window", image);
    showPicture("Origigi window", image_original);
    
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}

int resizeDimension(Mat img) {
    if (img.cols > img.rows)
        return img.rows;
    else
        return img.cols;
}

void showPicture(string title, Mat image) {
    namedWindow(title, WINDOW_AUTOSIZE);// Create a window for display.
    imshow(title, image);
}

Mat sharpening(Mat image, Mat image_original) {
    //int original_width = image.cols;
    //int original_height = image.rows;

    //int new_size = resizeDimension(image);

    //resize(image, image, Size(new_size, new_size), INTER_AREA);
    //resize(image_original, image_original, Size(new_size, new_size), INTER_AREA);

    //showPicture("Mini org window", image);

    int width = image.cols;
    int height = image.rows;

    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {

            red = 0.0;
            green = 0.0;
            blue = 0.0;
            //cout << i << " " << j << endl;
            for (int filterY = 0; filterY < filterHeight; filterY++) {
                for (int filterX = 0; filterX < filterWidth; filterX++) {
                    int imageX = (i - filterWidth / 2 + filterX + width) % width;
                    int imageY = (j - filterHeight / 2 + filterY + height) % height;
                    //cout << imageX << " " << imageY << " " << filterX << " " << filterY<<endl;
                    blue += image_original.at<Vec3b>(imageY, imageX)[0] * filter[filterX][filterY];
                    green += image_original.at<Vec3b>(imageY, imageX)[1] * filter[filterX][filterY];
                    red += image_original.at<Vec3b>(imageY, imageX)[2] * filter[filterX][filterY];
                }
            }

            image.at<Vec3b>(j, i)[0] = min(max(int(blue), 0), 255);
            image.at<Vec3b>(j, i)[1] = min(max(int(green), 0), 255);
            image.at<Vec3b>(j, i)[2] = min(max(int(red), 0), 255);
        }
    }
    return image;
}
