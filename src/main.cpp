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
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

class VO_estimator
{
    public:
    // members
    bool isempty;                               // flag of start
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
    Mat rmat_w;                                 // rotation world frame
    Mat tvec_w;                                 // translation world frame
    int n_landmarks_l;                          // number of landmarks desired(lower bound)
    int n_landmarks_u;                          // number of landmarks (upper bound)
    vector<Point3d> landmarks;                  // 3d points for tracking 
    vector<Point2f> landmarks_on_img_l;         // 2d points for tracking (on left image)
    int vstep;                                  // vertical step (image segamentation)
    int hstep;                                  // horizontal step
    vector<Point2d> new_matches_on_img_l;       // new added matches on left image
    vector<Point2d> new_matches_on_img_r;       // ...               on right image

    // constructor
    VO_estimator();
    // read stereo camera parameters
    void read_stereo_camera_param(Mat rot_l, Mat K_l, Mat dist_l, Mat rot_r, Mat K_r, Mat dist_r);
    // read new frame
    void read_new_frame(Mat img0, Mat img1);
    // track the existed landmarks
    void track_landmarks();
    // extract correspondences
    void extract_correspondences(int d); // d-> minimal distance between features
    // triangulate new landmarks
    void triangulate_correspondences();
    // update the projection matrix
    void update_projMat();
    // estimator process
    void process();
};

VO_estimator::VO_estimator()
{
    this->isempty = true;
    this->n_landmarks_l = 150;
    this->n_landmarks_u = 225;
    this->rmat_w = Mat::eye(3, 3, CV_64F);
    this->tvec_w = (Mat_<double>(3, 1) << 0,0,0);
    this->vstep = 6;
    this->hstep = 8;
}

void VO_estimator::read_stereo_camera_param(Mat rot_l, Mat K_l, Mat dist_l, Mat rot_r, Mat K_r, Mat dist_r)
{
    this->K_l = K_l;
    this->K_r = K_r;
    this->dist_l = dist_l;
    this->dist_r = dist_r;
    // compute projection matries
    Mat identity34 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Mat scale = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    // compute fundamental matrix from left to right camera
    Mat Tlr = rot_l(Rect(0, 0, 3, 3)) * (rot_r(Rect(3, 0, 1, 3)) - rot_l(Rect(3, 0, 1, 3)));
    Mat transform_mat;
    hconcat(rot_r(Rect(0, 0, 3, 3)) * rot_l(Rect(0, 0, 3, 3)).t(), Tlr, transform_mat);
    vconcat(transform_mat, scale, transform_mat);
    this->projMat_l = K_l * identity34;
    this->projMat_r = K_r * identity34 * transform_mat;
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

void VO_estimator::track_landmarks()
{
    vector<uchar> status;
    vector<float> err;
    vector<Point2f> tracked;
    calcOpticalFlowPyrLK(this->img_l_prev, this->img_l, this->landmarks_on_img_l, tracked, status, err, Size(11, 11), 6, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
    // remove untracked landmarks
    for(int p = status.size()-1; p > -1; p--)
    {
        if((int)(status[p]) != 1)
        {
            tracked.erase(tracked.begin() + p);
            this->landmarks.erase(this->landmarks.begin() + p);
        }
    }
    this->landmarks_on_img_l = tracked;
    // estimate rotation and translation
    vector<int> inliers;
    Mat rvec_pnp, tvec_pnp, rmat_pnp;
    solvePnPRansac(this->landmarks, this->landmarks_on_img_l, this->K_l, this->dist_l, rvec_pnp, tvec_pnp, true, 100, 15.0, 0.99, inliers, CV_P3P);
    Rodrigues(rvec_pnp, rmat_pnp);
    // inverse the rotation and translation
    this->rmat_w = rmat_pnp.inv();
    this->tvec_w = -rmat_pnp.inv() * tvec_pnp;
    // RANSAC filter out outliners
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

void VO_estimator::extract_correspondences(int d)
{
    // extract feature points
    vector<KeyPoint> kps_l, kps_r; // grid adapted detector is more preferable
    FAST(this->img_l, kps_l, 50, true, FastFeatureDetector::TYPE_9_16);
    FAST(this->img_r, kps_r, 50, true, FastFeatureDetector::TYPE_9_16);
    // sort kps_l by response
    sort(kps_l.begin(), kps_l.end(), [](KeyPoint& a, KeyPoint& b){return a.response > b.response;});
    // filter out features base on distance
    bool too_close;
    for(int i = 0; i < kps_l.size(); i++)
    {
        too_close = false;
        // compare with tracked landmarks
        for(int l = 0; l < this->landmarks_on_img_l.size(); l++)
        {
            if(norm(this->landmarks_on_img_l[l] - kps_l[i].pt) < d)
            {
                too_close = true;
                kps_l.erase(kps_l.begin() + i);
                i--;
                break;
            }
        }
        if((too_close == false))
        {
            // compare with stronger new features
            for(int j = 0; j < i; j++)
            {
                if(norm(kps_l[i].pt - kps_l[j].pt) < d)
                {
                    kps_l.erase(kps_l.begin() + i);
                    i--;
                    break;
                }
            }
        }
    }
    // compute descriptors
    Mat des_l, des_r;
    Ptr<DescriptorExtractor> freak = xfeatures2d::FREAK::create();
    Ptr<DescriptorExtractor> orb = ORB::create();
    orb->compute(this->img_l, kps_l, des_l);
    orb->compute(this->img_r, kps_r, des_r);
    // match them
    vector<DMatch> matches; 
    BFMatcher bfMatcher(NORM_HAMMING);
    if(kps_l.size() > 0 && kps_r.size() > 0)
    {
        bfMatcher.match(des_l, des_r, matches);
        vector<Point2d> pts_l, pts_r;
        for(int m = 0; m < matches.size(); m++)
        {
            this->new_matches_on_img_l.push_back(kps_l[matches[m].queryIdx].pt);
            this->new_matches_on_img_r.push_back(kps_r[matches[m].trainIdx].pt);
        }
    }
}

void VO_estimator::triangulate_correspondences()
{
    // undistortion
    vector<Point2d> undistort_l, undistort_r;
    undistortPoints(this->new_matches_on_img_l, undistort_l, this->K_l, this->dist_l, Mat::eye(3, 3, CV_64F), this->projMat_l);
    undistortPoints(this->new_matches_on_img_r, undistort_r, this->K_r, this->dist_r, Mat::eye(3, 3, CV_64F), this->projMat_r);
    for(int m = this->new_matches_on_img_l.size() - 1; m > -1; m--)
    {
        Mat score;
        Mat q = (Mat_<double>(1, 3) << undistort_r[m].x, undistort_r[m].y, 1);
        Mat p = (Mat_<double>(3, 1) << undistort_l[m].x, undistort_l[m].y, 1);
        score = q * this->F * p;
        // filter 1: geometry based
        if(fabs(score.at<double>(0,0)) < 0.003)
        {
            Mat p3d_;
            double x_, y_, z_, w_;
            // triangulation
            triangulatePoints(this->projMat_l, this->projMat_r, Mat(undistort_l[m]), Mat(undistort_r[m]), p3d_);
            w_ = p3d_.at<double>(3, 0);
            z_ = p3d_.at<double>(2, 0) / w_;
            // filter 2: positive depth
            if((z_ > 0.5) && (z_ < 30))
            {
                x_ = p3d_.at<double>(0, 0) / w_;
                y_ = p3d_.at<double>(1, 0) / w_;
                // transform the landmark to world frame
                Mat p_ = (Mat_<double>(3, 1) << x_, y_, z_);
                p_ = this->rmat_w * p_ + this->tvec_w;
                this->landmarks.push_back(Point3d(p_.at<double>(0,0), p_.at<double>(1,0), p_.at<double>(2,0)));
                this->landmarks_on_img_l.push_back(Point2f(this->new_matches_on_img_l[m]));
            }
            else
            {
                this->new_matches_on_img_l.erase(this->new_matches_on_img_l.begin() + m);
                this->new_matches_on_img_r.erase(this->new_matches_on_img_r.begin() + m);
            }
        }
        else
        {
            this->new_matches_on_img_l.erase(this->new_matches_on_img_l.begin() + m);
            this->new_matches_on_img_r.erase(this->new_matches_on_img_r.begin() + m);
        }
    }
}

void VO_estimator::process()
{
    cout << "--------------------" << endl;
    this->new_matches_on_img_l.clear();
    this->new_matches_on_img_r.clear();
    // track the previous feature (standard only)
    if((this->landmarks.size() > 0) && ~(this->img_l_prev.empty()))
    {
        track_landmarks();
    }
    // the case new landmarks need to be added 
    if(this->landmarks.size() < this->n_landmarks_l)
    {
        extract_correspondences(30);
        if(this->new_matches_on_img_l.size() > 0)
        {
            triangulate_correspondences();
        }
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

void visualization(VO_estimator estimator, float timeInSeconds)
{
    static vector<Point2d> trajectory;
    if(~estimator.isempty)
    {
        float factor = 15;
        Mat show;
        cvtColor(estimator.img_l, show, CV_GRAY2RGB);
        // draw new matches
        for(int f = 0; f < estimator.landmarks_on_img_l.size(); f++)
        {
            circle(show, estimator.landmarks_on_img_l[f], 1, Scalar(0, 255, 0), 3);
        }
        for(int m = 0; m < estimator.new_matches_on_img_l.size(); m++)
        {
            circle(show, estimator.new_matches_on_img_l[m], 1, Scalar(0, 0, 255), 3);
        }
        // draw map
        Mat map_2d = Mat::ones(480,480,CV_8UC3);
        // body point
        Point2d body(estimator.tvec_w.at<double>(0,0) * factor + map_2d.cols / 2, -estimator.tvec_w.at<double>(2,0) * factor + map_2d.rows / 2);
        // update trajectory
        trajectory.push_back(body);
        // draw trajectory
        if(trajectory.size() > 2)
        {
            for(int t = 1; t < trajectory.size(); t++)
            {
                line(map_2d, trajectory[t-1], trajectory[t], Scalar(175, 175, 175), 2);
            }
        }
        // draw landmarks
        Point2d l_;
        for(int l = 0; l < estimator.landmarks.size(); l++)
        {
            l_.x = estimator.landmarks[l].x * factor + map_2d.cols / 2;
            l_.y = -estimator.landmarks[l].z * factor + map_2d.rows / 2;
            circle(map_2d, l_, 1, Scalar(0, 255, 0), 1.3);
        }
        // draw direction
        Mat axis = (Mat_<double>(3, 1) << 0, 0, 25);
        axis = estimator.rmat_w * axis;
        Point2d arrow(axis.at<double>(0, 0), -axis.at<double>(2, 0));
        line(map_2d, body, body + arrow, Scalar(0, 0, 255), 2);
        // draw body
        circle(map_2d, body, 1, Scalar(0, 0, 255), 5);
        // write informations
        string text;
        text = to_string(timeInSeconds);
        text = text.substr(0,5);
        putText(map_2d, "t: " + text, Point2f(5, map_2d.rows - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 1, false);
        text = to_string(estimator.tvec_w.at<double>(2, 0));
        text = text.substr(0,5);
        putText(map_2d, "z: " + text, Point2f(5, map_2d.rows - 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 1, false);
        text = to_string(estimator.tvec_w.at<double>(1, 0));
        text = text.substr(0,5);
        putText(map_2d, "y: " + text, Point2f(5, map_2d.rows - 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 1, false);
        text = to_string(estimator.tvec_w.at<double>(0, 0));
        text = text.substr(0,5);
        putText(map_2d, "x: " + text, Point2f(5, map_2d.rows - 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 1, false);
        hconcat(show, map_2d, show);
        // show
        imshow("Visualization",show);
        waitKey(1);
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
        visualization(my_estimator, timeInSeconds);
    }
}