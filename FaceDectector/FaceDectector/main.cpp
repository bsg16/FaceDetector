#include"detect.h"
#include <opencv2/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"

Mat *spcm;     //所有样本
Mat *intg;     //积分图
int posnum;    //正样本数
int negnum;    //负样本数
int num;       //总样本数
float *spcmweight; //每个样本的权重

vector<haarclassifier> haarcfs;               //单个haar特征的分类器               
vector<haarclassifier>weakcfs;                //挑选出的最优弱分类器，与haar分类器是同一类
vector<strongclassifier>strongcfs;            //级联分类器，由一组强分类器组成，一个强分类器由一组弱分类器线性组合

int weakcfnum =300;                           //挑选的弱分类器个数
int **weakoutput;                             //弱分类器对样本的预测值
int **haarcfoutput;


int main() {
	GetLocalTime(&sys);
	printf("%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "训练开始弱分类器！！！\n";
	readpic(spcm, posnum, num);
	negnum = num - posnum;

	intg = new Mat[num];
	
	for (int i = 0; i < num; i++)
		integral(spcm[i], intg[i], CV_32SC1);
	

	/*getallclassifier(haarcfs,num,posnum);


	trainweakclassifier(haarcfs, weakcfs, num, posnum, weakcfnum,spcmweight,intg);
	cout << "最优弱分类器个数：" << weakcfs.size() << endl;

	float sumweight = 0;
	for (int i = 0; i < num; i++)
		sumweight += spcmweight[i];
	cout << "样本权重之和" << sumweight;

	获取弱分类器的输出
	getweakclassifieroutput(weakcfs, num, intg, weakoutput);

	写回分类器
	writeweakclissfier(weakcfs, haarcfs, weakoutput, num, spcmweight);

	GetLocalTime(&sys);
	printf("\n%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "弱分类器训练结束！！！\n";*/

	
	weakcfs.clear();
	//读取分类器
	readweakclissfier(weakcfs, weakcfnum, weakoutput, num);
	cout << "最优弱分类器个数：" << weakcfs.size() << endl;

	getweakclassifieroutput(weakcfs, num, intg, weakoutput);

	float tpr = 1.0;
	float fpr = 1.0;
	float neededfpr = 0.0000001;
	float neededtpr = 0.5;
	trainstrongclassifier(weakcfs, strongcfs, num, posnum,tpr,fpr,neededtpr,neededfpr, spcmweight, weakoutput);
	cout << "\n理论总正确率：" << tpr<<endl;
	cout << "理论总误报率：" << fpr << endl;

	GetLocalTime(&sys);
	printf("\n%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "训练结束！！！\n";

	long long head1, tail1, freq;        // timers
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	//
	QueryPerformanceCounter((LARGE_INTEGER *)&head1);	// start time
	
	

	//Mat src = imread("faces_test/image_0042.jpg");
	Mat src = imread("dong.jpg");
	//src = imread("faces_test/image_0011.jpg");
	//src = imread("dong.jpg");
	vector<Rect> recs;
	vector<float> winweight;   //窗口权重

	Mat gray;
	Mat in;
	cvtColor(src, gray, CV_BGR2GRAY);
	integral(gray, in, CV_32SC1);
	detect(weakcfs, strongcfs, in, recs, winweight);
	//合并重叠的窗口
	vector<Rect>win;
	emergewindow(recs, win, winweight);
	for (int i = 0; i < recs.size(); i++)
		rectangle(src, recs[i], Scalar(0, 255, 0), 2*winweight[i], 1, 0);
	
	/*for (int i = 0; i < win.size(); i++) {
		rectangle(src, win[i], Scalar(0, 255, 0), 1, 1, 0);
	}*/

	
	/*for (int i = 0; i < recs.size(); i++)
		cout << recs[i].x<<" "<< recs[i].y<<" "<< recs[i].width<<" "<< winweight[i] << endl;
	cout <<"窗口数量："<< recs.size();*/

	QueryPerformanceCounter((LARGE_INTEGER *)&tail1);	// end time
	double t = (tail1 - head1) * 1000.0 / freq;
	cout << "耗时：" << t << "ms" << endl;
	imshow("detectedfaces", src);
	Mat rst = src;
	mask(rst, win);
	imshow("mosaic", src);

	cv::waitKey(0);
	return 0;
}