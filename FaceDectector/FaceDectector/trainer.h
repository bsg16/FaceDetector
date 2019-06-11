#include<Cmlib.h>
#include<omp.h>
SYSTEMTIME sys;

struct haarclassifier{
	float theta;                   //阈值
	int x1, y1;
	int x2, y2;                    //特征矩形的坐标，竖直方向记x，水平方向记y
	int model;                     //haar模板类型，1*2，1*3，2*1，3*1,2*2
	int p;                         //不等号方向
	float err;                     //误差
	float alpha;                   //权重系数
};

struct strongclassifier {
	int idx;                        //在弱分类器的起始位置
	int size;						//包含弱分类器的个数
};
void readpic(Mat * &spcm,int &posnum, int &num) {
	string pattern_face = "R:\\程序\\vs项目\\FaceDectector\\FaceDectector\\faces\\*.bmp";
	string pattern_nonface = "R:\\程序\\vs项目\\FaceDectector\\FaceDectector\\nonfaces\\*.bmp";
	vector<cv::String> image_files_face;
	vector<cv::String> image_files_nonface;
	
	glob(pattern_face, image_files_face);
	glob(pattern_nonface, image_files_nonface);
	posnum = image_files_face.size();
	int negnum= image_files_nonface.size();
	num = posnum + negnum;

	spcm = new Mat[num];

	for (int i = 0; i < posnum; i++) 
		spcm[i] = imread(image_files_face[i]);
	for (int i =0; i < negnum; i++)
		spcm[i+posnum] = imread(image_files_nonface[i]);
}

int getrectingt(Mat &itg, int x1, int y1, int x2, int y2) {
	int i= itg.at<int>(x2, y2) + itg.at<int>(x1, y1) - itg.at<int>(x1, y2) - itg.at<int>(x2, y1);
	return i;
}

float gethaar(Mat &intg, int x1, int y1, int x2, int y2, int model) {
	int black;
	int white;
	switch (model) {
	case 1:                          //1*2
		white = getrectingt(intg, x1, y1, x2, (y2 + y1) / 2);
		black = getrectingt(intg, x1, (y2 + y1) / 2, x2, y2 );
		break;
	case 2:                           //1*3
		white = getrectingt(intg, x1, y1, x2, (y2 + 2*y1) / 3);
		white += getrectingt(intg, x1,(2*y2 + y1) / 3, x2,  y2);
		black=2* getrectingt(intg, x1, (y2 + 2 * y1) / 3, x2, (2 * y2 + y1) / 3);
		break;
	case 3:                           //2*1
		white = getrectingt(intg, x1, y1,(x2+x1)/2, y2);
		black = getrectingt(intg,(x2+ x1) / 2, y1, x2, y2);
		break;
	case 4:                           //3*1
		white = getrectingt(intg, x1, y1, (x2 + 2*x1) / 3, y2);
		white+= getrectingt(intg, (2 * x2 + x1) / 3, y1, x2, y2);
		black=2* getrectingt(intg, (x2 + 2 * x1) / 3, y1, (2*x2 + x1) / 3, y2);
		break;
	case 5:                           //2*2
		white= getrectingt(intg, x1, y1, (x1+x2)/2,(y2 + y1) / 2);
		white+= getrectingt(intg, (x1 + x2) / 2, (y2 + y1) / 2, x2,y2);
		black= getrectingt(intg, x1, (y2 + y1) / 2, (x1 + x2) / 2, y2);
		black += getrectingt(intg, (x1 + x2) / 2, y1, x2, (y2 + y1) / 2);
		break;
	}
	return white - black;
}



//计算所有的haar特征对应的分类器
void getallclassifier(vector<haarclassifier> &allhcf,  int num, int posnum) { 
	int m;  
	for (int x1 = 0; x1 < 20;x1++)
		for (int y1 =0; y1< 20; y1++) {
			m = 1;
			for (int x2 =x1+1; x2 <= 20; x2 += 1)
				for (int y2 = y1 + 2; y2 <= 20; y2 += 2)
				{
					haarclassifier tem;
					tem.x1 = x1;
					tem.x2 = x2;
					tem.y1 = y1;
					tem.y2 = y2;
					tem.model = m;
					allhcf.push_back(tem);
				}
			m = 2;
			for (int x2 = x1 + 1; x2 <=20; x2 += 1)
				for (int y2 = y1 + 3; y2 <= 20; y2 +=3)
				{
					haarclassifier tem;
					tem.x1 = x1;
					tem.x2 = x2;
					tem.y1 = y1;
					tem.y2 = y2;
					tem.model = m;
					allhcf.push_back(tem);
				}
			m = 3;
			for (int x2 = x1 + 2; x2 <= 20; x2 += 2)
				for (int y2 = y1 + 1; y2 <= 20; y2 +=1)
				{
					haarclassifier tem;
					tem.x1 = x1;
					tem.x2 = x2;
					tem.y1 = y1;
					tem.y2 = y2;
					tem.model = m;
					allhcf.push_back(tem);
				}
			m = 4;
			for (int x2 = x1 + 3; x2 <= 20; x2 += 3)
				for (int y2 = y1 + 1; y2 <= 20; y2 += 1)
				{
					haarclassifier tem;
					tem.x1 = x1;
					tem.x2 = x2;
					tem.y1 = y1;
					tem.y2 = y2;
					tem.model = m;
					allhcf.push_back(tem);
				}
			m = 5;
			for (int x2 = x1 + 2; x2 <= 20; x2 += 2)
				for (int y2 = y1 + 2; y2 <= 20; y2 += 2)
				{
					haarclassifier tem;
					tem.x1 = x1;
					tem.x2 = x2;
					tem.y1 = y1;
					tem.y2 = y2;
					tem.model = m;
					allhcf.push_back(tem);
				}
		}
}

//比较函数
bool largerthan(pair<int, float> &p1, pair<int, float> &p2) {
	return p1.second > p2.second;
};

void trainhaarclassifier(haarclassifier &hcf, int num, int posnum, Mat *intg, float *spcmweight) {
	vector<pair<int, float>> idx_haar;
	for (int i = 0; i < num; i++) {
		float h = gethaar(intg[i], hcf.x1, hcf.y1, hcf.x2, hcf.y2, hcf.model);
		pair<int, float>tem(i, h);
		idx_haar.push_back(tem);
	}
	sort(idx_haar.begin(), idx_haar.end(), largerthan);

	float t1 = 0.0;		 //全部人脸样本的权重的和t1
	float t0 = 0.0;		 //全部非人脸样本的权重的和t0
	for (int i = 0; i < posnum; i++)
		t1 += spcmweight[i];
	for (int i = posnum; i < num; i++)
		t0 += spcmweight[i];
	float s0 = 0.0;						     //在此元素之前的负样本的s0
	float s1 = 0.0;				             //在此元素之前的正样本数量s1
	float minerr = 1;                          //最小误差
	float theta = 0;
	int p = 0;
	float border = 1000000000;                //分界线
	for (int i = 0; i < num; i++) {
		float err = 1;
		int ptem;
		if (border != idx_haar[i].second) {
			if (s0 + (t1 - s1) < s1 + (t0 - s0)) {
				err = (s0 + (t1 - s1));
				ptem = -1;
			}
			else {
				err = (s1 + (t0 - s0));
				ptem = 1;
			}
			if (err <= minerr) {
				minerr = err;
				theta = idx_haar[i].second;
				p = ptem;
			}
			border = idx_haar[i].second;
		}
		if (idx_haar[i].first < posnum)
			s1 += spcmweight[idx_haar[i].first];
		else
			s0 += spcmweight[idx_haar[i].first];
	}
	hcf.p = p;
	hcf.theta = theta - 0.5;                     //避免特征值与theta相等的值过多，难以判断
	hcf.err = minerr;
}
//更新样本权重
void updatespcmweight(haarclassifier &hcf, float e,int num, int posnum, float* spcmweight,Mat *intg) {
	float beta = e / (1 - e);
	for (int i = 0; i < num; i++) {
		float ha = gethaar(intg[i], hcf.x1, hcf.y1, hcf.x2, hcf.y2, hcf.model);
		if (i < posnum) {
			if (hcf.p * ha < hcf.p * hcf.theta)
				spcmweight[i] *= beta;                 //若该正样本带入在某分类器的输出正确，降低该样本的权值
		}
		else {
			if (hcf.p * ha >= hcf.p * hcf.theta)
				spcmweight[i] *= beta;
		}
	}
	float sumweight = 0;
	for (int i = 0; i < num; i++)
		sumweight += spcmweight[i];
	for (int i = 0; i < num; i++)
		spcmweight[i] /= sumweight;                            //样本权重归一化*/
}


//adaboost迭代训练weakcfnum个最优弱分类器
void trainweakclassifier(vector<haarclassifier> &haarcfs,vector<haarclassifier> &weakcfs,int num,int posnum,int weakcfnum,float * &spcmweight,Mat *intg) {

	spcmweight = new float[num];                                 //初始化样本权重，均匀分布
	for (int i = 0; i < num; i++)
		spcmweight[i] = 1.0 / num;

	for (int i = 0; i < weakcfnum; i++) { 
		float minerr = 1.0;
		int minidx;
		#pragma omp parallel for
		for (int j = 0; j < haarcfs.size(); j++) {
			trainhaarclassifier(haarcfs[j], num, posnum, intg, spcmweight);
			if (minerr >= haarcfs[j].err) {
				minerr = haarcfs[j].err;
				minidx = j;
			}
		}
		weakcfs.push_back(haarcfs[minidx]);
		//haarcfs.erase(haarcfs.begin() + minidx);
		weakcfs[i].alpha = log((1 - weakcfs[i].err)/weakcfs[i].err);
		updatespcmweight(weakcfs[i] ,weakcfs[i].err, num, posnum, spcmweight,intg);
	}
}

//获取弱分类器对于全部样本的输出矩阵
void getweakclassifieroutput(vector<haarclassifier> &wcfs, int num, Mat * ingt, int ** &wcfopt) {
	wcfopt = new int*[wcfs.size()];
	for (int i = 0; i < wcfs.size(); i++)
		wcfopt[i] = new int[num];
	for (int i = 0; i<wcfs.size(); i++)
		for (int j = 0; j < num; j++) {
			float ha = gethaar(ingt[j], wcfs[i].x1, wcfs[i].y1, wcfs[i].x2, wcfs[i].y2, wcfs[i].model);
			if (wcfs[i].p*ha < wcfs[i].p*wcfs[i].theta)
				wcfopt[i][j] = 1;
			else
				wcfopt[i][j] = -1;
		}
}
//将中间结果写回磁盘
void writeweakclissfier(vector<haarclassifier>&weakcfs, vector<haarclassifier>&hcfs, int ** weakoutput, int num,float *spcmweight) {
	ofstream o1("weakcf1.txt");
	for (int i = 0; i < weakcfs.size(); i++) {
		o1 << weakcfs[i].p << " " << weakcfs[i].theta << " " << weakcfs[i].model << " " << weakcfs[i].x1 << " ";
		o1 << weakcfs[i].y1 << " " << weakcfs[i].x2 << " " << weakcfs[i].y2 << " " << weakcfs[i].err <<" "<< weakcfs[i].alpha<< endl;
	}
	o1.close();

	ofstream o3("haarclassifier1.txt");
	for (int i = 0; i < hcfs.size(); i++) {
		o3 << hcfs[i].p << " " << hcfs[i].theta << " " << hcfs[i].model << " " << hcfs[i].x1 << " ";
		o3 << hcfs[i].y1 << " " << hcfs[i].x2 << " " << hcfs[i].y2 << " " <<hcfs[i].err << endl;
	};
	o3.close();
	ofstream o4("spcmweight1.txt");
	for (int i = 0; i < num; i++) {
		o4 << spcmweight[i] << endl;
	}
	o4.close();
}

//从磁盘读训练好的最优弱分类器
void readweakclissfier(vector<haarclassifier>&weakcfs,int weaknum, int ** &weakoutput, int num) {
	weakoutput = new int*[weaknum];
	for (int i = 0; i < weaknum; i++)
		weakoutput[i] = new int[num];
	ifstream in1("weakcf1.txt");
	ifstream in2("weakoutput1.txt");
	int i = 0;
	for(int i = 0;i<weaknum;i++) {
		haarclassifier tem;
		in1 >> tem.p >> tem.theta >> tem.model >> tem.x1 >> tem.y1 >> tem.x2 >> tem.y2>>tem.err>>tem.alpha;
		for (int j = 0; j < num; j++) {
			in2 >> weakoutput[i][j];
		}
		weakcfs.push_back(tem);
		in1.get();
	}
	in1.close();
	in2.close();
}

//计算强分类器的检测率和误检率
void assessstrongclassifier(strongclassifier &scf, vector<haarclassifier>&weakcfs,float pass,float &tpr,float &fpr, int num,int posnum, int **weakoutput) {
	float *fea = new float[num];
	for (int j = 0; j < num; j++)
		for (int i = 0; i < scf.size; i++)
			fea[j] += weakcfs[scf.idx+i].alpha * weakoutput[scf.idx + i][j];    //该层强分类器对某样本的预测值
	int tp = 0;           //正确检测到的人脸数
	int fp = 0;           //被误检为人脸的非人脸数
	for (int i = 0; i < posnum; i++)
		if (fea[i] >0)
			tp++;
	for (int i = posnum; i < num; i++)
		if (fea[i]>0)
			fp++;
	tpr = tp / (float)posnum;
	fpr = fp / (float)(num - posnum);
}



//强分类器由一组弱分类器线性组合
//级联分类器由一组强分类器组成的容器
//样本总数，正样本数，每层检测率，每层误检率，总检测率，总误检率,弱分类器权重，样本权重，弱分类器输出
void trainstrongclassifier(vector<haarclassifier>&weakcfs,vector<strongclassifier>& strongcfs,int num, int posnum,float &realtpr,float &realfpr,float neededtpr,float neededfpr, float *spcmweight,int **weakoutput) {
	float currtpr = 1.0;               //当前总检测率
	float currfpr = 1.0;              //当前总误检率
	int weakidx = 0;                //弱分类器容器中元素迭代器
	int strongidx = 0;             //强分类器元素迭代器
	while (currfpr > neededfpr&& weakidx<weakcfs.size()&& currtpr>neededtpr) {                    //误检率合格或弱分类器不够用时终止
		strongclassifier s;
		s.idx = weakidx;
		s.size = 0;
		strongcfs.push_back(s);
		float currdivtpr = 0.0;          //该层jiancelv
		float currdivfpr = 1.0;          //该层误检率
		float pass = 0.0;
		//while (currdivtpr<=0.99999999|| currdivfpr>0.5) {
		while(strongcfs[strongidx].size<75){
			strongcfs[strongidx].size++;              //向强分类器中加一个弱分类器
			pass +=0.5* weakcfs[weakidx].alpha;
			assessstrongclassifier(strongcfs[strongidx],weakcfs,pass,currdivtpr, currdivfpr, num, posnum,weakoutput);
			weakidx++;
			if (weakidx == weakcfs.size())
				break;
		}
		currtpr *= currdivtpr;
		currfpr *= currdivfpr;
		cout << "第" << strongidx << "层起始强分类器参数:  起始位置"<< strongcfs[strongidx].idx<<" ";
		cout<<"弱分类器个数"<< strongcfs[strongidx].size << " 该层通过率：" << currdivtpr << " " << "该层误报率：" <<currdivfpr << endl;
		strongidx++;
	}
	realtpr = currtpr;
	realfpr = currfpr;
}