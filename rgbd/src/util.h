#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include <string>
#include <vector>
#include <utility>

typedef pcl::PointCloud<pcl::PointXYZRGB> Cloud;
using namespace std;

struct Calibration
{
  Calibration( float fx, float fy, float cx, float cy ) : fx(fx), fy(fy), cx(cx), cy(cy) {}
  Calibration() : fx(0), fy(0), cx(0), cy(0) {}
  float fx;
  float fy;
  float cx;
  float cy;
};

map<string, Calibration> makeCalibrations();

static const map<string, Calibration> CAMERA_CALIBRATIONS = makeCalibrations();

static const int LEN = 6;

static const string DATASETS[6] = {
  "long/rgbd_dataset_freiburg1_room/", 
  "long/rgbd_dataset_freiburg2_large_with_loop/", 
  "long/rgbd_dataset_freiburg2_pioneer_slam2/", 
  //"long/rgbd_dataset_freiburg3_long_office_household/",
  "long/rgbd_dataset_freiburg2_large_no_loop/", 
  "long/rgbd_dataset_freiburg2_pioneer_slam/", 
  "long/rgbd_dataset_freiburg2_pioneer_slam3/"
};

static const string DATA_FILES[6] = {
  "rgbd_dataset_freiburg1_room", 
  "rgbd_dataset_freiburg2_large_with_loop", 
  "rgbd_dataset_freiburg2_pioneer_slam2", 
  //"rgbd_dataset_freiburg3_long_office_household",
  "rgbd_dataset_freiburg2_large_no_loop", 
  "rgbd_dataset_freiburg2_pioneer_slam", 
  "rgbd_dataset_freiburg2_pioneer_slam3"
};

int
depthToColorCloud(const cv::Mat & depth,
                  const cv::Mat & color,
                  const float fx,
                  const float fy,
                  const float cx,
                  const float cy,
                  const float factor,
                  Cloud::Ptr * out);

int
loadColorCloud(const string & depth_path,
               const string & color_path,
               const string & calibration_id,
               const float factor,
               Cloud::Ptr * out);

pair<vector<int>, vector<int> >
matchTimestamps(vector<int> & timestamps1, 
                vector<int> & timestamps2,
                int maxDifference);
void
displayPairs(const vector<int> & l1,
             const vector<int> & l2,
             const string & base_path,
             const vector<string> & color_ind,
             const int dataset_num);

void save(const vector<int> & l1,
          const vector<int> & l2,
          const int & dataset_num,
          const vector<string> & color_ind,
          const string & set_name,
          const string & set_file_name,
          const string & tag,
          const string & base_path,
          const string & train_path,
          const string & val_path,
          const string & test_path);
