#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>


using namespace cv;
using namespace std;
using namespace std::chrono;

int largest_area = 0;
int largest_contour_index = 0;
int mini_area = 0;
int mini_contour_index = 0;


void show_histogram(string const& name, Mat1b const& image)
{
	float max = 0;
	float min = 5000;
	for (int i = 0; i < image.cols; i++)
	{
		int column_sum = 0;
		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
			if (column_sum > max) {
				max = column_sum;
				//cout << "Max " << max << endl;
			}
			if (column_sum < min) {
				min = column_sum;
				//cout << "Min " << min << endl;

			}
		}
	}



	// Set histogram bins count
	int bins = image.cols;
	// Set ranges for histogram bins
	float lranges[] = { 0, bins };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	Mat hist;
	int channels[] = { 0 };
	float maxN = max / 50;
	float const hist_height = maxN;
	Mat3b hist_image = Mat3b::zeros(hist_height + 10, bins + 20);

	int col_low = 0;
	float height_low = 200;
	int countA = 0;
	float height_A[630];
	Mat dst;

	// Loop find Low and Locate Low
	for (int i = 0; i < image.cols; i++)
	{
		float column_sum = 0;

		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
		}

		float const height = cvRound(column_sum * hist_height / max);
		line(hist_image, Point(i + 10, (hist_height - height) + 50), Point(i + 10, hist_height), Scalar::all(255));

		// Check Low graph
		if (height < height_low) {
			height_low = height;
			col_low = i;

		}

	}


	float H_AVG = 0;
	// Loop find Average low
	for (int i = 0; i < image.cols; i++)
	{
		float column_sum = 0;

		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
		}

		float const height = cvRound(column_sum * hist_height / max);

		if (i >= (col_low - 7) && i < (col_low + 6))
		{
			cout << "H--" << height << endl;
			H_AVG += height;

		}
	}
	H_AVG = H_AVG / 13;			//best value for average 
	float H_Minus = H_AVG - height_low;

	cout << "H_AVG____" << H_AVG << endl;
	cout << "low " << height_low << endl;
	cout << "H_AVG-------Minus " << H_Minus << endl;
	//cout << "locate low " << col_low << endl;



	Mat canny_output;
	Canny(hist_image, canny_output, 50, 50 * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// create hull array for convex hull points
	vector< vector<Point> > hull(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], 1);
	}

	drawContours(hist_image, contours, 0, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
	drawContours(hist_image, hull, 0, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
	cout << " AreaC: " << contourArea(contours[0]) << endl;
	cout << " AreaH: " << contourArea(hull[0]) << endl;
	float reN = 0;
	reN = contourArea(hull[0]) - contourArea(contours[0]);
	//cout << " Result: " << reN << endl;
	if (H_Minus > 5) {
		cout << " Defect Detection  " << endl;
		cout << "===================" << endl;
	}
	else {
		cout << " Non defect Detection  " << endl;
		cout << "===================" << endl;
	}
	imshow(name, hist_image);
	//This Code tell runtime this program.
	vector<int> values(10000);
	auto f = []() -> int { return rand() % 10000; };
	generate(values.begin(), values.end(), f);
	auto start = high_resolution_clock::now();
	sort(values.begin(), values.end());
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time taken by function: " << duration.count() << " milliseconds" << endl;
	//imshow(name, hist_image);
}



int Histogram_Calculate(Mat image) {
	float max = 0;
	float min = 5000;

	for (int i = 0; i < image.cols; i++)
	{
		int column_sum = 0;
		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
			if (column_sum > max) {
				max = column_sum;
				//cout << "Max " << max << endl;
			}
			if (column_sum < min) {
				min = column_sum;
				//cout << "Min " << min << endl;

			}
		}
	}



	// Set histogram bins count
	int bins = image.cols;
	// Set ranges for histogram bins
	float lranges[] = { 0, bins };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	Mat hist;
	int channels[] = { 0 };
	float maxN = max / 50;
	float const hist_height = maxN;
	Mat3b hist_image = Mat3b::zeros(hist_height + 10, bins + 20);

	int col_low = 0;
	float height_low = 200;
	int countA = 0;
	float height_A[630];
	Mat dst;

	// Loop find Low and Locate Low
	for (int i = 0; i < image.cols; i++)
	{
		float column_sum = 0;

		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
		}

		float const height = cvRound(column_sum * hist_height / max);
		line(hist_image, Point(i + 10, (hist_height - height) + 50), Point(i + 10, hist_height), Scalar::all(255));

		// Check Low graph
		if (height < height_low) {
			height_low = height;
			col_low = i;
		}
	}

	cout << "Locate" << col_low << endl;
	if (col_low < 100 || col_low >1500) {
		return 0;
	}


	return 1;
}









int Polar_Function(Mat image, Point2f center, float radiused) {
	Mat imgPoLin;

	linearPolar(image, imgPoLin, center, radiused + 10, INTER_LINEAR + WARP_FILL_OUTLIERS);
	imshow("5", imgPoLin);

	Rect myROI(380, 0, 130, 510);
	Mat croppedRef(imgPoLin, myROI);

	Mat imgCrop;
	// Copy the data into new matrix
	croppedRef.copyTo(imgCrop);
	imshow("REz", imgCrop);
	rotate(imgCrop, imgCrop, ROTATE_90_COUNTERCLOCKWISE);
	imshow("Rotate", imgCrop);

	int YoN = 0;
	YoN = Histogram_Calculate(imgCrop);
	cout << "YoN" << YoN << endl;
	if (YoN == 0)
	{
		return YoN;
	}
	show_histogram("name", imgCrop);
	return 1;

}



int Center_Circle(Mat image, Mat imageOriginal) {
	int NoY = 0;
	vector<vector<Point> > contours;
	findContours(image, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>centers(contours.size());
	vector<float>radius(contours.size());
	vector<Vec4i> hierarchy;


	// Calculate find Area of circle
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
		double area = contourArea(contours[i]);  //  Find the area of contour.
		if (area > largest_area)
		{
			largest_area = area;
			largest_contour_index = i;               //Store the index of largest contour.
		}
		cout << area << endl;
		cout << "----------------- " << endl;

	}

	cout << largest_area << endl;
	cout << "================== " << endl;
	//cout << largest_contour_index << endl;
	Mat testimg;
	// Draw circle for show Edge of lib
	circle(imageOriginal, centers[largest_contour_index], (int)radius[largest_contour_index] + 120, Scalar(0, 0, 0), 215);
	circle(imageOriginal, centers[largest_contour_index], (int)radius[largest_contour_index] - 18, Scalar(0, 0, 0), FILLED);

	imshow("2", imageOriginal);
	linearPolar(imageOriginal, testimg, centers[largest_contour_index], radius[largest_contour_index] + 10, INTER_LINEAR + WARP_FILL_OUTLIERS);
	imshow("testimg", testimg);

	Point2f cen;
	cen = centers.at(largest_contour_index);
	// Call Function make polar
	NoY = Polar_Function(imageOriginal, cen, radius[largest_contour_index]);

	largest_area = 0;
	return NoY;

}

int Recheck(Mat imageOriginal) {
	Mat imgG, imgCanny, imgRz;
	int status = 0;
	imgRz = imageOriginal.clone();
	cvtColor(imgRz, imgG, COLOR_BGR2GRAY);
	blur(imgG, imgG, Size(3, 3));
	//threshold(imgG, imgTh, 120, 255, THRESH_BINARY); //Threshold the gray.
	Canny(imgG, imgCanny, 300, 550);
	status = Center_Circle(imgCanny, imageOriginal);

	if (status == 0)
	{
		rotate(imageOriginal, imageOriginal, ROTATE_90_CLOCKWISE);
		cvtColor(imageOriginal, imgG, COLOR_BGR2GRAY);
		blur(imgG, imgG, Size(3, 3));
		//threshold(imgG, imgTh, 120, 255, THRESH_BINARY); //Threshold the gray.
		Canny(imgG, imgCanny, 300, 550);
		status = Center_Circle(imgCanny, imageOriginal);
		if (status == 0)
		{
			rotate(imgRz, imgRz, ROTATE_180);
			cvtColor(imgRz, imgG, COLOR_BGR2GRAY);
			blur(imgG, imgG, Size(3, 3));
			//threshold(imgG, imgTh, 120, 255, THRESH_BINARY); //Threshold the gray.
			Canny(imgG, imgCanny, 300, 550);
			status = Center_Circle(imgCanny, imageOriginal);

		}
	}


	return status;
}

int main(int argc, const char* argv[]) {

	Mat imgOri;
	Mat imgRz, imgG, imgCn, imgMr, imgPoLog, imgPoLin, imgRePoLin, imgRePoLog;
	int recheck = 0;

	string folder("img/*.jpg");
	vector<String> fn;
	glob(folder, fn, false);
	vector<Mat> images;
	size_t count = fn.size(); //number of png files in images folder.\

	//Check number of images.
	cout << "image in folder  " << count << endl;

	//Main LooB.
	for (size_t i = 0; i < count; i++)
	{
		//Preprocessing
		imgOri = imread(fn[i]);
		resize(imgOri, imgRz, Size(), 0.5, 0.5); //Half Resize 1280*1040 to 640*520 pixcel.

		recheck = Recheck(imgRz);
		//imshow("Resize", imgRz);
		cout << " Last status" << recheck << endl;

		waitKey(0);
	}
	waitKey(0);
	return 0;

}
