#include"iostream"
#include<stdlib.h>
#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/video/background_segm.hpp"
using namespace std;
using namespace cv;

double alpha_R = 0.5;
double alpha_G = 0.065;
double alpha_B = 0.075;
//Pixel�࣬������ص���ͨ����Сֵ������
class Pixel
{
public:
	int p_row;
	int p_col;
	double p_color;
	Pixel(int row, int col, double color);
};
Pixel::Pixel(int row, int col, double color)
{
	p_row = row;
	p_col = col;
	p_color = color;
}

//������ɫֵ��С�������У������ҳ���Сֵ��
bool color_cmp(double c1, double c2)
{
	if (c1 < c2)
		return true;
	return false;
}

//������ɫֵ�Ӵ�С���У������ҳ�ǰ0.1%���ȵ����ص�)
bool color_cmp2(const Pixel& p1, const Pixel& p2)
{
	if (p1.p_color > p2.p_color)
		return true;
	return false;
}

//���ɸ�˹����
double generateGaussianNoise(double mu, double sigma)
{
	//����Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�ٹ����˹�������X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flagΪ�湹���˹�������
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

//Ϊͼ����Ӹ�˹����
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//�ж�ͼ���������
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//��Ӹ�˹����
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 10;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}

int t_cal(double t_element[3],double A_element[3])
{

	Mat OriginalImg;
	OriginalImg = imread("example2.jpg", IMREAD_COLOR);//��ȡԭʼ��ɫͼ��
	if (OriginalImg.empty())  //�ж�ͼ��Է��ȡ�ɹ�
	{
		cout << "����!��ȡͼ��ʧ��\n";
		return -1;
	}
	//imshow("ԭͼ", OriginalImg); //��ʾԭʼͼ��
	//cout << "Width:" << OriginalImg.rows << "\tHeight:" << OriginalImg.cols << endl;//��ӡͼ�񳤿�

	//�޸�ͼƬ��С
	Mat ResizeImg;
	if (OriginalImg.cols > 640)
		resize(OriginalImg, ResizeImg, Size(640, 640 * OriginalImg.rows / OriginalImg.cols));
	else
		ResizeImg = OriginalImg.clone();

	//��ȡԭͼ��ͨ��
	Mat MinImg = Mat::zeros(ResizeImg.size(), CV_8UC1);
	double color_element[3] = {};
	for (int i = 0; i < MinImg.rows; i++)
	{
		for (int j = 0; j < MinImg.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				color_element[k] = (double)ResizeImg.at<Vec3b>(i, j)[k]; //��ȡ�����ص���ͨ������Сֵ
			}
			std::sort(color_element, color_element + 3, color_cmp);
			MinImg.at<uchar>(i, j) = color_element[0];
		}
	}

	Mat ExpandImg;
	int window_element = 5; //�������ڰ뾶
	copyMakeBorder(MinImg, ExpandImg, window_element, window_element,
		window_element, window_element, BORDER_REPLICATE); //�����Ե
	//imshow("��չ�߽�", ExpandImg);
	Mat DarkImg = Mat::zeros(ResizeImg.size(), CV_8UC1);
	vector<Pixel> pixel_element;
	double min_element = 0;
	for (int i = 0; i < ExpandImg.rows - (window_element * 2 + 1); i++)
	{
		for (int j = 0; j < ExpandImg.cols - (window_element * 2 + 1); j++)
		{
			Mat WindowImg = ExpandImg(Range(i + 1, i + 2 * window_element + 2), Range(j + 1, j + 2 * window_element + 2));
			minMaxLoc(WindowImg, &min_element); //Ѱ����������Сֵ
			DarkImg.at<uchar>(i, j) = min_element;
			pixel_element.push_back(Pixel(i, j, min_element));
		}
	}

	//��������ֵAȷ��
	std::sort(pixel_element.begin(), pixel_element.end(), color_cmp2); //�����ذ��Ҷ�ֵ�ɴ�С����
	int N_element = ceil(DarkImg.rows*DarkImg.cols / 1000);
	//int B_element = 0, G_element = 0, R_element = 0;
	//int A_element[3] = { B_element ,G_element ,R_element };
	for (int i = 0; i < N_element; i++) //ѡȡ���е�0.1%�����ص�����
	{
		for (int j = 0; j < 3; j++)
		{
			A_element[j] += (double) ResizeImg.at<Vec3b>(pixel_element[i].p_row, pixel_element[i].p_col)[j];
		}
	}
	for (int k = 0; k < 3; k++)
	{
		A_element[k] = A_element[k] / N_element;
	}

	//�����ͨ��͸���� https://wenku.baidu.com/view/fe8225a259fafab069dc5022aaea998fcd224039.html
	vector<Mat> channels_element;
	split(ResizeImg, channels_element);
	//Mat RtImg1 = Mat::zeros(ResizeImg.size(),CV_32FC1);
	//subtract(channels_element.at(2), A_element[2], RtImg1);
	//divide(RtImg1, -A_element[2], RtImg1);
	//min_element = 0;
	//minMaxLoc(RtImg1, &min_element, NULL, NULL, NULL);

	//Mat RtImg2 = Mat::zeros(ResizeImg.size(), CV_32FC1);
	//subtract(channels_element.at(2), A_element[2], RtImg2);
	//divide(RtImg2, (255 - A_element[2]), RtImg2);
	double max_element = 0;
	minMaxLoc(channels_element.at(2), NULL, &max_element, NULL, NULL);

	t_element[2] = max(max_element / (-A_element[2]) + 1, (max_element - A_element[2]) / (255 - A_element[2]));
	t_element[1] = pow(t_element[2], alpha_G / alpha_R);
	t_element[0] = pow(t_element[2], alpha_B / alpha_R);

}

int main()
{

	double t_element[3], A_element[3] = {0,0,0};
	t_cal(t_element,A_element);
	char num[1];
	for (int m = 10; m < 16; m++)
	{
		// itoa(m, num, 10);
		ostringstream oss1;
		oss1 << "F:/�ҵ���Դ/data/batch_" << m << "/";
		cv::String path = oss1.str();
		ostringstream oss2;
		oss2 << "F:/�ҵ���Դ/data/batch_" << m << "_new/";
		cv::String dest = oss2.str();
		cv::String savedfilename;
		std::vector<cv::String> filenames;
		cv::Mat srcImg, dstImg;
		cv::glob(path, filenames); //opencv����������ȡָ��·�����ļ�����һ���ܺ��õĺ���

		for (int n = 0; n < filenames.size(); n++) {
			Mat OriginalImg;
			OriginalImg = imread(filenames[n], IMREAD_COLOR);//��ȡԭʼ��ɫͼ��
			if (OriginalImg.empty())  //�ж�ͼ��Է��ȡ�ɹ�
			{
				cout << "����!��ȡͼ��ʧ��\n";
				return -1;
			}

			Mat MistImg = OriginalImg.clone();
			for (int i = 0; i < MistImg.rows; i++)
			{
				for (int j = 0; j < MistImg.cols; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						MistImg.at<Vec3b>(i, j)[k] = (double)MistImg.at<Vec3b>(i, j)[k] * t_element[k]
							+ ((1 - t_element[k]) * A_element[k]);
					}
				}
			}

			GaussianBlur(MistImg, MistImg, Size(5, 5), 11, 11);
			MistImg = addGaussianNoise(MistImg);
			savedfilename = dest + filenames[n].substr(path.length());
			imwrite(savedfilename, MistImg);
		}
	}
	
	//while (1)
	//{
	//	if (waitKey(0) == 27) break;
	//}
}