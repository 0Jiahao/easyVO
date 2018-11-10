#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <bits/stdc++.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

class VO_estimator
{
    public:
    // members
    bool isempty;                               // flag of start
    Mat rot_l;                                  // transform matrix of left camera
    Mat rot_r;                                  // ...              of right camera
    Mat K_l;                                    // camera matrix of left camera
    Mat K_r;                                    // ...           of right camera
    Mat dist_l;                                 // distortion coefficients of left camera 
    Mat dist_r;                                 // ...                     of right camera
    Mat projMat_l;                              // project matrix of left camera
    Mat projMat_r;                              // ...            of right camera
    Mat F;                                      // fundamental matrix between left and right camera
    Mat img_l;                                  // left image
    Mat img_l_prev;                             // left image (previous)
    Mat img_r;                                  // right image
    Mat rvec_pnp;                               // extracted rotational vector
    Mat tvec_pnp;                               // extracted traslation vector
    int n_landmarks_l;                          // number of landmarks desired(lower bound)
    int n_landmarks_u;                          // number of landmarks (upper bound)
    vector<Point3d> landmarks;                              // 3d points for tracking 
    vector<Point2f> landmarks_on_img_l;         // 2d points for tracking (on left image)
    vector<Point2d> new_matches_on_img_l;       // new added matches on left image
    vector<Point2d> new_matches_on_img_r;       // ...               on right image

    // constructor
    VO_estimator();
    // read stereo camera parameters
    void read_stereo_camera_param(Mat rot_l, Mat K_l, Mat dist_l, Mat rot_r, Mat K_r, Mat dist_r);
    // read new frame
    void read_new_frame(Mat img0, Mat img1);
    // estimator process
    void process();
};

VO_estimator::VO_estimator()
{
    this->isempty = true;
    this->n_landmarks_l = 200;
    this->n_landmarks_u = 350;
}

void VO_estimator::read_stereo_camera_param(Mat rot_l, Mat K_l, Mat dist_l, Mat rot_r, Mat K_r, Mat dist_r)
{
    this->rot_l = rot_l;
    this->rot_r = rot_r;
    this->K_l = K_l;
    this->K_r = K_r;
    this->dist_l = dist_l;
    this->dist_r = dist_r;
    // compute projection matries
    Mat identity34 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    this->projMat_l = this->K_l * identity34 * rot_l;
    this->projMat_r = this->K_r * identity34 * rot_r;
    // compute fundamental matrix from left to right camera
    Mat Tlr = rot_l(Rect(0, 0, 3, 3)).t() * (rot_r(Rect(3, 0, 1, 3)) - rot_l(Rect(3, 0, 1, 3)));
    double tx = Tlr.at<double>(0,0);
    double ty = Tlr.at<double>(0,1);
    double tz = Tlr.at<double>(0,2);
    Mat Tx = (Mat_<double>(3, 3) << 0, -tz, ty, tz, 0, -tx, -ty, tx, 0);
    this->F = K_r.inv().t() * rot_r(Rect(0, 0, 3, 3)) * rot_l(Rect(0, 0, 3, 3)).t() * Tx * K_l.inv();
}

void VO_estimator::read_new_frame(Mat img0, Mat img1)
{
    this->img_l_prev = this->img_l;
    this->img_l = img0;
    this->img_r = img1;
    this->isempty = ((this->img_l.empty()) || (this->img_r.empty()));
}

void VO_estimator::process()
{
    cout << "--------------------" << endl;
    this->new_matches_on_img_l.clear();
    this->new_matches_on_img_r.clear();
    // track the previous feature (standard only)
    if((this->landmarks.size() > 0) && ~(this->img_l_prev.empty()))
    {
        cout << "standard" << endl;
        vector<uchar> status;
	    vector<float> err;
        vector<Point2f> tracked;
        calcOpticalFlowPyrLK(this->img_l_prev, this->img_l, this->landmarks_on_img_l, tracked, status, err, Size(11, 11), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
        // only restore the well-tracked points
        for(int p = status.size()-1; p > -1; p--)
        {
            if((int)(status[p]) != 1)
            {
                tracked.erase(tracked.begin() + p);
                this->landmarks.erase(this->landmarks.begin() + p);
            }
        }
        this->landmarks_on_img_l = tracked;
        cout << "[DEBUG]\t" << "Successfully track: " << this->landmarks.size() << endl;
        // RANSAC filter out outliners
        vector<int> inliers;
        solvePnPRansac(this->landmarks, this->landmarks_on_img_l, this->K_l, this->dist_l, this->rvec_pnp, this->tvec_pnp, true, 100, 100.0, 0.99, inliers, CV_P3P);
        vector<Point3d> landmarks_inliers;
        vector<Point2f> landmarks_on_img_l_inliers;
        for(int i = 0; i < inliers.size(); i++)
        {
            landmarks_inliers.push_back(this->landmarks[inliers[i]]);
            landmarks_on_img_l_inliers.push_back(this->landmarks_on_img_l[inliers[i]]);
        }
        this->landmarks = landmarks_inliers;
        this->landmarks_on_img_l = landmarks_on_img_l_inliers;
    }
    // the case new landmarks need to be added 
    if(this->landmarks.size() < this->n_landmarks_l)
    {
        cout << "keyframe" << endl;
        // extract feature points
        vector<KeyPoint> pts_l, pts_r; // grid adapted detector is more preferable
        // number of feature going to extract
        int n_feature_need = 1.3 * (this->n_landmarks_u - this->landmarks.size());
        Ptr<ORB> orb = ORB::create(n_feature_need, 1, 1, 31, 0, 2, ORB::FAST_SCORE, 31, 10);
        orb->detect(this->img_l, pts_l);
        orb->detect(this->img_r, pts_r);
        // compute descriptors
        Mat des_l, des_r;
        orb->compute(this->img_l, pts_l, des_l);
        orb->compute(this->img_r, pts_r, des_r);
        // match them
        vector<DMatch> matches; // this can be faster, because we know the F
        BFMatcher bfMatcher(NORM_HAMMING);
        if(pts_l.size() > 0 && pts_r.size() > 0)
        {
            bfMatcher.match(des_l, des_r, matches);
            // compute score for each correspondences
            vector<float> scores;
            int n_good_matches = 0;
            for(int m = 0; m < matches.size(); m++)
            {
                Mat score;
                Mat q = (Mat_<double>(1, 3) << pts_r[matches[m].trainIdx].pt.x, pts_r[matches[m].trainIdx].pt.y, 1);
                Mat p = (Mat_<double>(3, 1) << pts_l[matches[m].queryIdx].pt.x, pts_l[matches[m].queryIdx].pt.y, 1);
                score = q * this->F * p;
                if(fabs(score.at<double>(0,0)) < 0.003)
                {
                    n_good_matches++;
                }
                scores.push_back(fabs(score.at<double>(0,0)));
            }
            // make an vector of index
            vector<int> idx(scores.size());
            size_t n(0);
            generate(begin(idx), end(idx), [&]{return n++;});
            // sort the score and return corresponding index
            sort(begin(idx), end(idx), [&](int x, int y){return scores[x] < scores[y];});
            for(int m = 0; m < ((n_good_matches > this->n_landmarks_u - landmarks.size())?this->n_landmarks_u - landmarks.size():n_good_matches) ; m++)
            {
                this->new_matches_on_img_l.push_back(pts_l[matches[idx[m]].queryIdx].pt);
                this->new_matches_on_img_r.push_back(pts_r[matches[idx[m]].trainIdx].pt);
            }
            Mat new_landmarks;
            triangulatePoints(this->projMat_l, this->projMat_r, new_matches_on_img_l, new_matches_on_img_r, new_landmarks);
            // rescale the 3d point
            for(int p = 0; p < new_landmarks.cols; p++)
            {
                new_landmarks.at<double>(0, p) = new_landmarks.at<double>(0, p) / new_landmarks.at<double>(3, p);
                new_landmarks.at<double>(1, p) = new_landmarks.at<double>(1, p) / new_landmarks.at<double>(3, p);
                new_landmarks.at<double>(2, p) = new_landmarks.at<double>(2, p) / new_landmarks.at<double>(3, p);
                // add result (3d) to the estimator
                this->landmarks.push_back(Point3d(new_landmarks.at<double>(0, p), new_landmarks.at<double>(1, p), new_landmarks.at<double>(2, p)));
            }
            // add result (2d) to the estimator
            this->landmarks_on_img_l.insert(this->landmarks_on_img_l.end(), new_matches_on_img_l.begin(), new_matches_on_img_l.end());
        }
    }
    else
    {

    }
}

vector<unsigned long long> extract_timestamps(const char *path)
{
   struct dirent *entry;
   DIR *dir = opendir(path);
   
   vector<unsigned long long> timestamps;
   int idx = 0;
   while ((entry = readdir(dir)) != NULL)
   {
        string filename;
        filename = entry->d_name;
        if(filename.length() == 23)
        {
            unsigned long long data_time = std::stoull(filename.substr(0,19),0); 
            timestamps.push_back(data_time);
            idx++;  
        }
   }
   closedir(dir);
   sort(timestamps.begin(), timestamps.end());
   return timestamps;
}

void visualization(VO_estimator estimator)
{
    if(~estimator.isempty)
    {
        Mat show;
        hconcat(estimator.img_l, estimator.img_r, show);
        cvtColor(show, show, CV_GRAY2RGB);
        Point2d bais(estimator.img_l.cols, 0);
        for(int f = 0; f < estimator.landmarks_on_img_l.size(); f++)
        {
            circle(show, estimator.landmarks_on_img_l[f], 1, Scalar(0, 255, 0), 3);
        }
        for(int m = 0; m < estimator.new_matches_on_img_l.size(); m++)
        {
            line(show, estimator.new_matches_on_img_l[m], estimator.new_matches_on_img_r[m] + bais, Scalar(0, 0, 255), 1);
            circle(show, estimator.new_matches_on_img_l[m], 1, Scalar(0, 0, 255), 3);
            circle(show, estimator.new_matches_on_img_r[m] + bais, 1, Scalar(0, 0, 255), 3);
        }
        imshow("Image",show);
        waitKey(50);
    }
}

int main() {
    // extract the timestamps
    vector<unsigned long long> timestamps;
    timestamps = extract_timestamps("../mav0/cam0/data");
    // data initialization
    Mat img_l, img_r;
    // estimator initialization
    VO_estimator my_estimator;
    // read camera parameters
    FileStorage fs_l("../sensors/camera_left.yml", FileStorage::READ);
	Mat rot_l, K_l, dist_l;
	fs_l["T_BS"] >> rot_l;
    fs_l["cameraMatrix"] >> K_l;
    fs_l["distCoeffs"] >> dist_l;
    fs_l.release();
    FileStorage fs_r("../sensors/camera_right.yml", FileStorage::READ);
	Mat rot_r, K_r, dist_r;
	fs_r["T_BS"] >> rot_r;
    fs_r["cameraMatrix"] >> K_r;
    fs_r["distCoeffs"] >> dist_r;
    fs_r.release();
    my_estimator.read_stereo_camera_param(rot_l, K_l, dist_l, rot_r, K_r, dist_r);
    // loop start
    for(int i = 0; i < timestamps.size(); i++)
    {
        // read raw data
        img_l = imread("../mav0/cam0/data/" + to_string(timestamps[i]) + ".png", 0);
        img_r = imread("../mav0/cam1/data/" + to_string(timestamps[i]) + ".png", 0);
        // start timer
        clock_t startTime = clock();
        // estimator functions
        my_estimator.read_new_frame(img_l, img_r);
        my_estimator.process();
        // stop timer
        clock_t endTime = clock();
        clock_t clockTicksTaken = endTime - startTime;
        double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
        cout << "time: " << timeInSeconds << endl;
        // visualization
        visualization(my_estimator);
    }
}