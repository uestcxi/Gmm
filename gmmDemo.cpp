#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat img, fgmask, fgimg;
bool update_bg_model = true;
bool pause = false;
Ptr<BackgroundSubtractorMOG2> bg_model = createBackgroundSubtractorMOG2(20, 16, true);
void refineSegments(const Mat&img, Mat& mask, Mat& dst)
{
	int niters = 3;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat temp;

	dilate(mask, temp, Mat(), Point(-1, -1), niters);
	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);
	dilate(temp, temp, Mat(), Point(-1, -1), niters);

	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	dst = Mat::zeros(img.size(), CV_8UC3);
	if (contours.size() == 0)
		return;
	//iterate through all the top-level contours
	//draw each connected component with its own randow color
	int idx = 0, largestComp = 0;
	double maxArea = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		if (area > maxArea)
		{
			maxArea = area;
			largestComp = idx;
		}
	}
	Scalar color(0, 255, 255);
	drawContours(dst, contours, largestComp, color, CV_FILLED, 8, hierarchy);
}

//process
int main()
{
	VideoCapture capture("H:\\video\\videotest.mp4");
	if (!capture.isOpened())
		return 0;
	int nFs = capture.get(CV_CAP_PROP_FRAME_COUNT);
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;
	while (nFs--)
	{
		capture.read(img);
		resize(img, img, Size(), 0.5, 0.5, INTER_LINEAR);
		bg_model->apply(img, fgmask, update_bg_model ? 0.005 : 0);
		refineSegments(img, fgmask, fgimg);
		imshow("foreground image", fgimg);
		imshow("image", img);
		waitKey(delay);
	}
	return 0;
}
