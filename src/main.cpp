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
#include <cvsba/cvsba.h>
#include <sstream>
#include <fstream>

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
    vector<Point3d> landmarks;                  // 3d points for tracking 
    vector<Point2f> landmarks_on_img_l;         // 2d points for tracking (on left image)
    vector<Point3d> new_landmarks;              // new 3d points via triangulation
    vector<Point2d> new_matches_on_img_l;       // new added matches on left image
    vector<Point2d> new_matches_on_img_r;       // ...               on right image
    bool is_keyframe;                           // flag of keyfrome judgment
    int n_ba_size;
    cvsba::Sba sba;                             // sparse bundle adjustment
    vector<int> index_tracking;                 // the index of landmarks that are tracking
    vector<Point3d> sba_point_3d;
    vector<vector<Point2d>> sba_point_2d;
    vector<vector<int>> sba_visibility;
    vector<Mat> sba_R, sba_T, sba_Kv, sba_distv;
    int NPOINTS;                                // number of landmarks from latest keyframe
    int NCAMS;                                  // the frame index from latest keyframe
    Mat sba_dist;
    Mat tvec_pnp;
    Mat rvec_pnp;

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
    // construct sba
    void construct_sba();
    // update sba
    void update_sba();
};

VO_estimator::VO_estimator()
{
    this->isempty = true;
    this->rmat_w = Mat::eye(3, 3, CV_64F);
    this->tvec_w = (Mat_<double>(3, 1) << 0, 0, 0);
    this->is_keyframe = true;
    // initialization of sba
    this->NCAMS = 1;
    cvsba::Sba::Params params;
    params.type = cvsba::Sba::MOTION;
    params.iterations = 75;
    params.minError = 1e-10;
    params.fixedIntrinsics = 5;
    params.fixedDistortion = 5;
    params.verbose = false;
    this->sba.setParams(params);
    this->rvec_pnp = (Mat_<double>(3, 1) << 0, 0, 0);
    this->tvec_pnp = (Mat_<double>(3, 1) << 0, 0, 0);
    this->n_ba_size = 20;
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
    // camera matrix for sba (s = 1, k3 = 0)
    double k1, k2, p1, p2;
    k1 = this->dist_l.at<double>(0,0);
    k2 = this->dist_l.at<double>(0,1);
    p1 = this->dist_l.at<double>(0,2);
    p2 = this->dist_l.at<double>(0,3);
    this->sba_dist = (Mat_<double>(1, 5) << k1, k2, p1, p2, 0);
}

void VO_estimator::read_new_frame(Mat img0, Mat img1)
{
    this->img_l_prev = this->img_l;
    this->img_l = img0;
    this->img_r = img1;
    this->isempty = ((this->img_l.empty()) || (this->img_r.empty()));
}

void VO_estimator::update_sba()
{
    // new data for curent frame
    vector<int> new_visibility;
    vector<Point2d> new_point_2d; 
    int prev_idx = 0;
    int idx_i = 0;
    for(int np = 0; np < this->sba_point_3d.size(); np++)
    {
        if(this->sba_visibility[this->sba_visibility.size() - 1][np] == 1)
        {
            if(prev_idx == this->index_tracking[idx_i])
            {
                new_visibility.push_back(1);
                new_point_2d.push_back(Point2d(this->landmarks_on_img_l[idx_i]));
                idx_i++;
            }
            else
            {
                new_visibility.push_back(0);
                new_point_2d.push_back(Point2d(0, 0));
            }
            prev_idx++;
        }
        else
        {
            new_visibility.push_back(0);
            new_point_2d.push_back(Point2d(0, 0));
        }
    }
    // add new data to matrix
    this->sba_visibility.push_back(new_visibility);
    this->sba_point_2d.push_back(new_point_2d);
    this->sba_R.push_back(this->rvec_pnp);
    this->sba_T.push_back(this->tvec_pnp);
    this->sba_Kv.push_back(this->K_l);
    this->sba_distv.push_back(this->sba_dist);
    this->sba.run(this->sba_point_3d, this->sba_point_2d, this->sba_visibility, this->sba_Kv, this->sba_R, this->sba_T, this->sba_distv);
    if(this->sba.getFinalReprjError() < 200)
    {
        // inverse the rotation and translation
        Mat rmat;
        Rodrigues(this->sba_R[this->sba_R.size() - 1], rmat);
        this->rmat_w = rmat.inv();
        this->tvec_w = -rmat.inv() * this->sba_T[this->sba_T.size() - 1];
    }
    // remove current data
    this->sba_visibility.pop_back();
    this->sba_point_2d.pop_back();
    this->sba_R.pop_back();
    this->sba_T.pop_back();
    this->sba_Kv.pop_back();
    this->sba_distv.pop_back();
}

void VO_estimator::construct_sba()
{
    cout << "/********/" << endl;
    if(this->sba_visibility.size() == this->n_ba_size)
    {
        // erase old data
        this->sba_visibility.erase(this->sba_visibility.begin());
        this->sba_point_2d.erase(this->sba_point_2d.begin());
        this->sba_R.erase(this->sba_R.begin());
        this->sba_T.erase(this->sba_T.begin());
        this->sba_Kv.erase(this->sba_Kv.begin());
        this->sba_distv.erase(this->sba_distv.begin());
    }
    // remove invisible points
    vector<int> visibilities;
    for(int np = 0; np < this->sba_point_3d.size(); np++)
    {
        int visibility = 0;
        for(int nc = 0; nc < this->sba_visibility.size(); nc++)
        {
            visibility = visibility + this->sba_visibility[nc][np];
        }
        visibilities.push_back(visibility);
    }
    for(int np = this->sba_point_3d.size() - 1; np > -1; np--)
    {
        if(visibilities[np] == 0)
        {
            this->sba_point_3d.erase(this->sba_point_3d.begin() + np);
            for(int nc = 0; nc < this->sba_visibility.size(); nc++)
            {
                this->sba_point_2d[nc].erase(this->sba_point_2d[nc].begin() + np);
                this->sba_visibility[nc].erase(this->sba_visibility[nc].begin() + np);
            }
        }
    }
    // add new points
    this->sba_point_3d.insert(this->sba_point_3d.end(), this->new_landmarks.begin(), this->new_landmarks.end());
    for(int nc = 0; nc < this->sba_visibility.size(); nc++)
    {
        for (int i = 0; i < this->new_landmarks.size(); i++)
        {
            this->sba_visibility[nc].push_back(0);
            this->sba_point_2d[nc].push_back(Point2d(0, 0));
        }
    }
    // adding new row
    vector<int> new_visibility;
    vector<Point2d> new_point_2d; 
    // from old to new visibility (visibility)
    if(this->sba_visibility.size() > 0)
    {
        int prev_idx = 0;
        int idx_i = 0;
        for(int np = 0; np < this->sba_point_3d.size() - this->new_landmarks.size(); np++)
        {
            if(this->sba_visibility[this->sba_visibility.size() - 1][np] == 1)
            {
                if(prev_idx == this->index_tracking[idx_i])
                {
                    new_visibility.push_back(1);
                    new_point_2d.push_back(Point2d(this->landmarks_on_img_l[idx_i]));
                    idx_i++;
                }
                else
                {
                    new_visibility.push_back(0);
                    new_point_2d.push_back(Point2d(0, 0));
                }
                prev_idx++;
            }
            else
            {
                new_visibility.push_back(0);
                new_point_2d.push_back(Point2d(0, 0));
            }
        }
    }
    // from new matches
    cout << "size new 2d:" << new_point_2d.size() << endl;
    for(int np = this->new_landmarks.size() - 1; np > -1; np--)
    {
        new_visibility.push_back(1);
        new_point_2d.push_back(this->new_matches_on_img_l[np]);
    }
    // add new data to matrix
    this->sba_visibility.push_back(new_visibility);
    this->sba_point_2d.push_back(new_point_2d);
    // add new data (R, T, K, dist)
    this->sba_R.push_back(this->rvec_pnp);
    this->sba_T.push_back(this->tvec_pnp);
    this->sba_Kv.push_back(this->K_l);
    this->sba_distv.push_back(this->sba_dist);
}

void VO_estimator::track_landmarks()
{
    vector<uchar> status;
    vector<float> err;
    vector<Point2f> tracked;
    calcOpticalFlowPyrLK(this->img_l_prev, this->img_l, this->landmarks_on_img_l, tracked, status, err, Size(15, 15), 6, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 1e-4);
    // remove untracked landmarks
    vector<Point3d> tracked_3d;
    vector<Point2f> tracked_2d;
    vector<int> tracked_index;
    for(int i = 0; i < status.size(); i++)
    {
        if(((bool)(status[i] == 1)))
        {
            tracked_3d.push_back(this->landmarks[i]);
            tracked_2d.push_back(tracked[i]);
            tracked_index.push_back(this->index_tracking[i]);
        }
    }
    // estimate rotation and translation
    vector<int> inliers;
    solvePnPRansac(tracked_3d, tracked_2d, this->K_l, this->dist_l, this->rvec_pnp, this->tvec_pnp, true, 100, 5.0, 0.99, inliers, CV_P3P);
    Mat rmat;
    Rodrigues(this->rvec_pnp, rmat);
    this->rmat_w = rmat.inv();
    this->tvec_w = -rmat.inv() * this->tvec_pnp;
    // RANSAC filter out outliners
    vector<Point3d> inliers_3d;
    vector<Point2f> inliers_2d;
    vector<int> inliers_index;
    for(int i = 0; i < inliers.size(); i++)
    {
        inliers_3d.push_back(tracked_3d[inliers[i]]);
        inliers_2d.push_back(tracked_2d[inliers[i]]);
        inliers_index.push_back(tracked_index[inliers[i]]);
    }
    this->landmarks = inliers_3d;
    this->landmarks_on_img_l = inliers_2d;
    this->index_tracking = inliers_index;
    update_sba();
    // keyframe judegment
    this->is_keyframe = (this->landmarks.size() / (float)this->NPOINTS > 0.8)?false:true;
    // NCAMS too big also need to create a new key frame
    if(this->is_keyframe)
    {
        this->NCAMS = 1;
    }
    else
    {
        this->NCAMS++;
    }
}

void VO_estimator::extract_correspondences(int d)
{
    // extract feature points
    vector<KeyPoint> kps_l, kps_r; // grid adapted detector is more preferable
    FAST(this->img_l, kps_l, 30, true, FastFeatureDetector::TYPE_9_16);
    FAST(this->img_r, kps_r, 30, true, FastFeatureDetector::TYPE_9_16);
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
    this->new_landmarks.clear();
    for(int m = this->new_matches_on_img_l.size() - 1; m > -1; m--)
    {
        Mat score;
        Mat q = (Mat_<double>(1, 3) << undistort_r[m].x, undistort_r[m].y, 1);
        Mat p = (Mat_<double>(3, 1) << undistort_l[m].x, undistort_l[m].y, 1);
        score = q * this->F * p;
        // filter 1: geometry based
        if(fabs(score.at<double>(0,0)) < 0.005)
        {
            Mat p3d_;
            double x_, y_, z_, w_;
            // triangulation
            triangulatePoints(this->projMat_l, this->projMat_r, Mat(undistort_l[m]), Mat(undistort_r[m]), p3d_);
            w_ = p3d_.at<double>(3, 0);
            z_ = p3d_.at<double>(2, 0) / w_;
            // filter 2: positive depth
            if((z_ > 0.0) && (z_ < 20))
            {
                x_ = p3d_.at<double>(0, 0) / w_;
                y_ = p3d_.at<double>(1, 0) / w_;
                // transform the landmark to world frame
                Mat p_ = (Mat_<double>(3, 1) << x_, y_, z_);
                p_ = this->rmat_w * p_ + this->tvec_w;
                this->landmarks.push_back(Point3d(p_.at<double>(0,0), p_.at<double>(1,0), p_.at<double>(2,0)));
                this->new_landmarks.push_back(Point3d(p_.at<double>(0,0), p_.at<double>(1,0), p_.at<double>(2,0)));
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
    construct_sba();
    // update the number of landmarks from the latest keyframe
    this->NPOINTS = this->landmarks.size();
    // initialize the index_tracking
    this->index_tracking.clear();
    for(int p = 0; p < this->NPOINTS; p++)
    {
        index_tracking.push_back(p);
    }
}

void VO_estimator::process()
{
    this->new_matches_on_img_l.clear();
    this->new_matches_on_img_r.clear();
    // track the previous feature (standard only)
    if((this->landmarks.size() > 0) && ~(this->img_l_prev.empty()))
    {
        track_landmarks();
    }
    // the case new landmarks need to be added 
    if(this->is_keyframe)
    {
        extract_correspondences(15);
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
    static vector<Point2d> keyframes;
    if(~estimator.isempty)
    {
        float factor = 20;
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
        // // draw sba trace
        // int n = 0;
        // int nc = estimator.sba_visibility.size() - 1;
        // if(estimator.sba_visibility.size() > 1)
        // {
        //     for(int i = 0; i < estimator.sba_point_3d.size(); i++)
        //     {
        //         if((estimator.sba_visibility[0][i] == 1) && (estimator.sba_visibility[1][i] == 1))
        //         {
        //             n++;
        //             line(show, estimator.sba_point_2d[0][i], estimator.sba_point_2d[1][i], Scalar(175, 175, 175), 2);
        //         }
        //     }
        //     cout << "draw " << n << "traces" << endl;
        // }
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
        // draw keyframe sign
        if(estimator.is_keyframe)
        {
            // update keyframes
            keyframes.push_back(body);
            rectangle(show, Point2f(0, 0), Point2f(show.cols, show.rows), Scalar(0, 0, 255), 10, 8, 0);
        }
        // draw keyframes
        for(int t = 0; t < keyframes.size(); t++)
        {
            circle(map_2d, keyframes[t], 1, Scalar(0, 0, 255), 2);  
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
        line(map_2d, body, body + arrow, Scalar(255, 0, 0), 2);
        // draw body
        circle(map_2d, body, 1, Scalar(255, 0, 0), 5);
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
    // write file
    ofstream outFile;
    outFile.open("../est_traj.txt", ios::out);
    outFile << "# timestamp tx ty tz qx qy qz qw" << endl;
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
        cout << "time:\t" << timeInSeconds << endl;
        // visualization
        visualization(my_estimator, timeInSeconds);
        outFile << timestamps[i] / 1000000000 << '.' << timestamps[i] % 1000000000 << ' ' << my_estimator.tvec_w.at<double>(0,0) << ' ' << my_estimator.tvec_w.at<double>(0,1) << ' ' << my_estimator.tvec_w.at<double>(0,2) << ' ' << 0 << ' ' << 0 << ' ' << 0 << ' ' << 1 << endl;
    }
    outFile.close();
}