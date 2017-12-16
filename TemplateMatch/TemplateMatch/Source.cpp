#include <iostream>
#include <algorithm>
#include <Windows.h>
#include <tchar.h>
#include <sstream>
#include <string>
#include <opencv2\opencv.hpp>
#include <queue>
#include <opencv2\features2d.hpp>
#include <iterator>
#include <future>

using namespace std;
using namespace cv;

namespace
{
	int g_hueBin = 57;
	int g_sueBin = 25;
	int g_vueBin = 11;

	Mat imgSrc;
	Mat imgSrcTemplate;

	struct QNode
	{
		QNode(float tVal, cv::Point tP)
			:val(tVal), p(tP)
		{ }

		float val;
		cv::Point p;

		bool operator<(const QNode &oth)
		{
			return val < oth.val;
		}

		bool operator>(const QNode &oth)
		{
			return !(*this < oth) && !(*this == oth);
		}

		bool operator==(const QNode &oth)
		{
			return val == oth.val;
		}
	};
}

void TemplateMatch(const Mat &imgSrc, const Mat &imgSrcTemplate)
{
	int nResultRow = imgSrc.rows - imgSrcTemplate.rows;
	int nReusltCol = imgSrc.cols - imgSrcTemplate.cols;

	Mat imgResult;
	imgResult.create(nResultRow, nReusltCol, CV_32FC1);

	matchTemplate(imgSrc, imgSrcTemplate, imgResult, TM_SQDIFF);
	normalize(imgResult, imgResult, 0, 1, NORM_MINMAX);


	std::priority_queue<QNode,std::vector<QNode>, std::greater<>> que;
	Mat imgOut(::imgSrc.clone());

	for (int i = 0; i < imgResult.rows; i++)
	{
		for (int j = 0; j < imgResult.cols; j++)
		{
			float val = imgResult.at<float>(i, j);
			if (val < 0.6)
			{
				que.push(QNode(val, Point(j, i)));
			}
		}
	}

	for (int i = 0; i < 20000 && i < que.size(); i++ )
	{
		auto node = que.top();
		que.pop();
		cv::rectangle(imgOut, node.p,
			Point(node.p.x + imgSrcTemplate.cols, node.p.y + imgSrcTemplate.rows),
			Scalar(0, 255, 255),
			2, 8);
	}

	imshow("result", imgOut);
}

void CalcBackProjectMatch(const Mat &imgSrc,const Mat &imgSrcTemplate, int index)
{
	if (g_hueBin == 0 || g_sueBin == 0 || g_vueBin == 0)
		return;

	Mat imgHSVTemplate;
	cvtColor(imgSrcTemplate, imgHSVTemplate, CV_RGB2HSV);

	int histSize[]{ g_hueBin,g_sueBin, g_vueBin };
	float hueRange[]{ 0,180 };
	float sueRange[]{ 0,256 };
	float vueRange[]{ 0,256 };
	const float *ranges[]{ hueRange ,sueRange, vueRange };
	int channels[]{ 0 ,1, 2 };

	Mat matHist;
	calcHist(&imgHSVTemplate, 1, channels, Mat(), matHist, 3, histSize, ranges);
	normalize(matHist, matHist, 0, 255, NORM_MINMAX);

	Mat matBackProj;

	Mat imgHSVSrc;
	cvtColor(imgSrc, imgHSVSrc, CV_RGB2HSV);
	calcBackProject(&imgHSVSrc, 1, channels, matHist, matBackProj, ranges);
	//equalizeHist(matBackProj, matBackProj);

	//DrawMatchRectangle(imgSrc, imgSrcTemplate, matBackProj);

	Mat imgWhite;
	imgWhite.create(cv::Size(imgSrcTemplate.cols, imgSrcTemplate.rows), CV_8UC1);
	cv::rectangle(imgWhite, Point(0,0),
		Point(imgSrcTemplate.cols, imgSrcTemplate.rows),
		Scalar(255, 255, 255), cv::FILLED);

	TemplateMatch(matBackProj, imgWhite);
	imshow("CalcBackProject", matBackProj);
}

void OnBinChange(int ,void *)
{
	CalcBackProjectMatch(imgSrc, imgSrcTemplate, 0);
}

int _tmain()
{
	imgSrc = cv::imread("E:\\Users\\Administrator\\pictures\\Test\\matchtest\\origin1.jpg");
	imgSrcTemplate = cv::imread("E:\\Users\\Administrator\\pictures\\Test\\matchtest\\template11.jpg");

	namedWindow("CalcBackProject");
	namedWindow("result");

	createTrackbar("Hue", "CalcBackProject", &g_hueBin, 180, OnBinChange);
	createTrackbar("Sue", "CalcBackProject", &g_sueBin, 256, OnBinChange);
	createTrackbar("Vue", "CalcBackProject", &g_vueBin, 256, OnBinChange);

	CalcBackProjectMatch(imgSrc, imgSrcTemplate, 0);

	waitKey(0);
	std::shared_ptr<int> p;

	return 0;
}