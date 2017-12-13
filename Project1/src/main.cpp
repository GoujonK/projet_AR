#include "RobustMatcher.h"
#include "dirent.h"

using namespace std;
using namespace cv;

int main() {

	VideoCapture cam;
	cam.open(0);
	cam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (!cam.isOpened()) {
		cout << "Erreur ouverture de la caméra" << endl;
		return 0;
	}

	string textDisp;

	Mat testImage, trainImage = imread("data/000083.jpg", CV_LOAD_IMAGE_UNCHANGED);
	if (trainImage.empty())
	{
		cout << "Error loading the image..." << endl;
		return 0;
	}

	RobustMatcher RM;
	vector<KeyPoint> trainKP, testKP;
	Mat trainDesc, testDesc;
	//Mat descriptor_model;
	vector<cv::DMatch> good_matches;

	RM.computeKeyPoints(trainImage, trainKP);
	RM.computeDescriptors(trainImage, trainKP, trainDesc);

	while (true)
	{
		double tm = (double)cvGetTickCount();

		cam >> testImage;

		//RM.fastRobustMatch(testImage, good_matches, testKP, trainDesc);

	
	
		RM.robustMatch(testImage, good_matches, testKP, trainDesc);

		sprintf((char *)textDisp.c_str(), "#Good Matches %3d", good_matches.size());
		putText(testImage, (char *)textDisp.c_str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.7,
			CV_RGB(50, 255, 50), 1, -1, false);

		tm = (double)cvGetTickCount() - tm;

		sprintf((char *)textDisp.c_str(), "FPS%5.1f", 1000. / (tm / (cvGetTickFrequency()*1000.)));
		putText(testImage, (char *)textDisp.c_str(), Point(testImage.cols - 110, 20), FONT_HERSHEY_SIMPLEX, 0.7,
			CV_RGB(255, 0, 0), 1, -1, false);

		imshow("Image Recognition", testImage);
		if (waitKey(1) == 27) break;
	}

	return 0;
}

/*#include "RobustMatcher.h"

using namespace std;
using namespace cv;

int main() {

	VideoCapture cam;
	cam.open(0);
	cam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (!cam.isOpened()) {
		cout << "Erreur ouverture de la caméra" << endl;
		return 0;
	}

	string textDisp;

	/*Mat testImage[3], trainImage[3];
	testImage[0], trainImage[0] = imread("data/bc.jpg", CV_LOAD_IMAGE_UNCHANGED);
	testImage[1], trainImage[1] = imread("data/wotlk.jpg", CV_LOAD_IMAGE_UNCHANGED);
	testImage[2], trainImage[2] = imread("data/cata.jpg", CV_LOAD_IMAGE_UNCHANGED);
	testImage[3], trainImage[3] = imread("data/mop.jpg", CV_LOAD_IMAGE_UNCHANGED);
	
	Mat testImage, trainImage = imread("data/000083.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//for (int n = 0; n == 3; n++) {
		if (trainImage.empty())
		{
			cout << "Error loading the image..." << endl;
			return 0;
		}
	//}

	RobustMatcher RM;
	vector<KeyPoint> trainKP, testKP;
	Mat trainDesc, testDesc;
	//Mat descriptor_model;
	vector<cv::DMatch> good_matches;

	RM.computeKeyPoints(trainImage, trainKP);
	RM.computeDescriptors(trainImage, trainKP, trainDesc);

	while (true)
	{
		double tm = (double)cvGetTickCount();

		cam >> testImage;

		//RM.fastRobustMatch(testImage, good_matches, testKP, trainDesc);

		RM.robustMatch(testImage, good_matches, testKP, trainDesc);

		sprintf((char *)textDisp.c_str(), "#Good Matches %3d", good_matches.size());
		putText(testImage, (char *)textDisp.c_str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.7,
				CV_RGB(50, 255, 50), 1, -1, false);

		tm = (double)cvGetTickCount() - tm;

		sprintf((char *)textDisp.c_str(), "FPS%5.1f", 1000. / (tm / (cvGetTickFrequency()*1000.)));
		putText(testImage, (char *)textDisp.c_str(), Point(testImage.cols-110, 20), FONT_HERSHEY_SIMPLEX, 0.7,
				CV_RGB(255, 0, 0), 1, -1, false);

		imshow("Image Recognition", testImage);
		if (waitKey(1) == 27) break;
	}

	return 0;
}
*/