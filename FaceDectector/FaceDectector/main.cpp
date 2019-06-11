#include"detect.h"
#include <opencv2/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"

Mat *spcm;     //��������
Mat *intg;     //����ͼ
int posnum;    //��������
int negnum;    //��������
int num;       //��������
float *spcmweight; //ÿ��������Ȩ��

vector<haarclassifier> haarcfs;               //����haar�����ķ�����               
vector<haarclassifier>weakcfs;                //��ѡ��������������������haar��������ͬһ��
vector<strongclassifier>strongcfs;            //��������������һ��ǿ��������ɣ�һ��ǿ��������һ�����������������

int weakcfnum =300;                           //��ѡ��������������
int **weakoutput;                             //����������������Ԥ��ֵ
int **haarcfoutput;


int main() {
	GetLocalTime(&sys);
	printf("%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "ѵ����ʼ��������������\n";
	readpic(spcm, posnum, num);
	negnum = num - posnum;

	intg = new Mat[num];
	
	for (int i = 0; i < num; i++)
		integral(spcm[i], intg[i], CV_32SC1);
	

	/*getallclassifier(haarcfs,num,posnum);


	trainweakclassifier(haarcfs, weakcfs, num, posnum, weakcfnum,spcmweight,intg);
	cout << "������������������" << weakcfs.size() << endl;

	float sumweight = 0;
	for (int i = 0; i < num; i++)
		sumweight += spcmweight[i];
	cout << "����Ȩ��֮��" << sumweight;

	��ȡ�������������
	getweakclassifieroutput(weakcfs, num, intg, weakoutput);

	д�ط�����
	writeweakclissfier(weakcfs, haarcfs, weakoutput, num, spcmweight);

	GetLocalTime(&sys);
	printf("\n%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "��������ѵ������������\n";*/

	
	weakcfs.clear();
	//��ȡ������
	readweakclissfier(weakcfs, weakcfnum, weakoutput, num);
	cout << "������������������" << weakcfs.size() << endl;

	getweakclassifieroutput(weakcfs, num, intg, weakoutput);

	float tpr = 1.0;
	float fpr = 1.0;
	float neededfpr = 0.0000001;
	float neededtpr = 0.5;
	trainstrongclassifier(weakcfs, strongcfs, num, posnum,tpr,fpr,neededtpr,neededfpr, spcmweight, weakoutput);
	cout << "\n��������ȷ�ʣ�" << tpr<<endl;
	cout << "���������ʣ�" << fpr << endl;

	GetLocalTime(&sys);
	printf("\n%02d:%02d:%02d \n", sys.wHour, sys.wMinute, sys.wSecond);
	cout << "ѵ������������\n";

	long long head1, tail1, freq;        // timers
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	//
	QueryPerformanceCounter((LARGE_INTEGER *)&head1);	// start time
	
	

	//Mat src = imread("faces_test/image_0042.jpg");
	Mat src = imread("dong.jpg");
	//src = imread("faces_test/image_0011.jpg");
	//src = imread("dong.jpg");
	vector<Rect> recs;
	vector<float> winweight;   //����Ȩ��

	Mat gray;
	Mat in;
	cvtColor(src, gray, CV_BGR2GRAY);
	integral(gray, in, CV_32SC1);
	detect(weakcfs, strongcfs, in, recs, winweight);
	//�ϲ��ص��Ĵ���
	vector<Rect>win;
	emergewindow(recs, win, winweight);
	for (int i = 0; i < recs.size(); i++)
		rectangle(src, recs[i], Scalar(0, 255, 0), 2*winweight[i], 1, 0);
	
	/*for (int i = 0; i < win.size(); i++) {
		rectangle(src, win[i], Scalar(0, 255, 0), 1, 1, 0);
	}*/

	
	/*for (int i = 0; i < recs.size(); i++)
		cout << recs[i].x<<" "<< recs[i].y<<" "<< recs[i].width<<" "<< winweight[i] << endl;
	cout <<"����������"<< recs.size();*/

	QueryPerformanceCounter((LARGE_INTEGER *)&tail1);	// end time
	double t = (tail1 - head1) * 1000.0 / freq;
	cout << "��ʱ��" << t << "ms" << endl;
	imshow("detectedfaces", src);
	Mat rst = src;
	mask(rst, win);
	imshow("mosaic", src);

	cv::waitKey(0);
	return 0;
}