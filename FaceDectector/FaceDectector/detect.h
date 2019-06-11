#include"trainer.h" 
#include <opencv2\imgproc\types_c.h>


void test(vector<haarclassifier>&weakcfs, vector<strongclassifier>& strongcfs) {
	Mat *te;
	Mat *intg;
	string pattern_test = "R:\\程序\\vs项目\\FaceDectector\\FaceDectector\\faces_test\\*.jpg";
	vector<cv::String> image_files_test;
	glob(pattern_test, image_files_test);
	int picnum = image_files_test.size();
	te = new Mat[picnum];
	intg = new Mat[picnum];
	for (int i = 0; i < picnum; i++) {
		te[i] = imread(image_files_test[i]);
	}
	for (int i = 0; i < picnum; i++) {
		vector<Rect> recs;
		Mat gr;
		cvtColor(te[i], gr, CV_BGR2GRAY);
		integral(gr, intg[i], CV_32SC1);
	}
}

void detect(vector<haarclassifier>&weakcfs, vector<strongclassifier>& strongcfs, Mat &src, vector<Rect> &pos,vector<float> &winweight) {
	int len = src.rows;
	int wid = src.cols;
	int offset =20;             //每次滑动的距离
	int scale = len /20 < wid / 20 ? len / 20 : wid / 20;  //缩放规模
	//找出所有备选窗口
	for (int t = 1; t <= scale; t++) {
		for (int x = 0; x + 20 * t < len; x += offset) {               //竖直方向滑动窗口
			for (int y = 0; y + 20 * t < wid; y += offset)            //水平方向滑动窗口
			{
				int i = 0;
				while (true)
				{
					int idx = strongcfs[i].idx;
					int size = strongcfs[i].size;
					float judge = 0.0;
					float wwt = 1.0;
					int x1, y1, x2, y2;
					for (int j = idx; j < idx + size; j++)               //遍历一个强分类器
					{
						x1 = x + weakcfs[j].x1*t;
						y1 = y + weakcfs[j].y1*t;
						x2 = x1 + (weakcfs[j].x2 - weakcfs[j].x1)*t;
						y2 = y1 + (weakcfs[j].y2 - weakcfs[j].y1)*t;
						float ha = gethaar(src, x1, y1, x2, y2, weakcfs[j].model);
						if (weakcfs[j].p*ha < weakcfs[j].p*weakcfs[j].theta*t*t)
							judge = judge + weakcfs[j].alpha;
						else
							judge = judge - weakcfs[j].alpha;
					}
					if (judge < 0)
						break;
					else {
						wwt *= judge;
						i++;
						if (i == strongcfs.size()) {
							Rect tem;
							tem.x = y;
							tem.y = x;
							tem.width = 20*t;
							tem.height = 20*t;
							pos.push_back(tem);
							winweight.push_back(wwt);
							break;
						}

					}
				}
			}
		}
	}

}

//合并窗口
void emergewindow(vector<Rect> pos, vector<Rect>&recs,vector<float>winweight) {
	if (pos.size() == 0)
		return;
	//把得到的人脸窗口分类,用tags数组区分
	//若两个窗口重叠区域的面积大于等于任意一个窗口面积的一半, 则判断两个窗口在同一个区域
	int *tag = new int[pos.size()];
	for (int i = 0; i < pos.size(); i++)
		tag[i] = -1;

	int areano = 0;
	for (int i = pos.size() - 1; i >= 0; i--) {
		if (tag[i] != -1)                      //第i个窗口已分类，跳过
			continue;
		for (int j = pos.size() - 1; j >= 0; j--) {
			if (tag[j] == -1)                  //在已分过类的窗口中判断
				continue;
			int maxx1 = pos[i].x > pos[j].x ? pos[i].x : pos[j].x;
			int maxy1 = pos[i].y > pos[j].y ? pos[i].y : pos[j].y;
			int minx2 = pos[i].x + pos[i].width < pos[j].x + pos[j].width ? pos[i].x + pos[i].width : pos[j].x + pos[j].width;
			int miny2 = pos[i].y + pos[i].height < pos[j].y + pos[j].height ? pos[i].y + pos[i].height : pos[j].y + pos[j].height;
			if ((maxx1 < minx2) && (maxy1 < miny2)) {
				float ovlap = (minx2 - maxx1)*(miny2 - maxy1);
				if (ovlap >= 0.5* pos[i].area() || ovlap >= 0.5*pos[j].area()) {
					tag[i] = tag[j];
					break;
				}
			}
		}
		if (tag[i] == -1) {                      //第i个窗口未找分类，为其分配一个
			tag[i] = areano;
			areano++;
		}
	}
	//对同一区域的窗口求均值;
	float *areacontain = new float[areano];
	for (int i = 0; i < areano; i++)
		areacontain[i] = 0;
	Rect tem{ 0,0,0,0 };
	for (int i = 0; i < areano; i++)
		recs.push_back(tem);
	for (int i = 0; i < pos.size(); i++) {
		areacontain[tag[i]]+=winweight[i];
		recs[tag[i]].x +=  winweight[i] * pos[i].x;
		recs[tag[i]].y +=  winweight[i] * pos[i].y;
		recs[tag[i]].width +=  winweight[i] * pos[i].width;
		recs[tag[i]].height +=  winweight[i] * pos[i].height;
	}
	for (int i = 0; i < areano; i++) {
		recs[i].x /= areacontain[i];
		recs[i].y /= areacontain[i];
		recs[i].width /= areacontain[i];
		recs[i].height /= areacontain[i];
	}
}

void mask(Mat &src, vector<Rect>win) {

	for (vector<Rect>::const_iterator r = win.begin(); r != win.end(); r++) {
		cv::Rect facerect = *r;
		Mat roi = src(cv::Rect(facerect.x, facerect.y, facerect.width, facerect.height));
		int W = 16;
		int H = 16;
		for (int i = 0; i<roi.cols; i += W) {
			for (int j =0; j<roi.rows; j += H) {
				uchar s = roi.at<uchar>(j + H / 2, (i + W / 2) * 3);
				uchar s1 = roi.at<uchar>(j + H / 2, (i+ W / 2) * 3 + 1);
				uchar s2 = roi.at<uchar>(j + H / 2, (i+ W / 2) * 3 + 2);
				for (int ii = i ; ii < i+W; ii++) {
					for (int jj = j; jj < j+H; jj++) {
						roi.at<uchar>(jj, ii * 3 + 0) = s;
						roi.at<uchar>(jj, ii * 3 + 1) = s1;
						roi.at<uchar>(jj, ii * 3 + 2) = s2;
					}
				}
			}
		}
	}

}