#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <cstdio>

#include "util.h"

using namespace std;

map<string, Calibration>
makeCalibrations()
{
  map<string, Calibration> camera_calibrations;
  camera_calibrations["def"] = Calibration(525.0, 525.0, 319.5, 239.5);
  camera_calibrations["fr1"] = Calibration(517.3, 516.5, 318.6, 255.3);
  camera_calibrations["fr2"] = Calibration(520.9, 521.0, 325.1, 249.7);
  camera_calibrations["fr3"] = Calibration(535.4, 539.2, 320.1, 247.6);
  camera_calibrations["asus"] = Calibration(544.47329, 544.47329, 320, 240);
  return camera_calibrations;
}

int
depthToColorCloud(const cv::Mat & depth,
                  const cv::Mat & color,
                  const float fx,
                  const float fy,
                  const float cx,
                  const float cy,
                  const float factor,
                  Cloud::Ptr * out)
{
  Cloud::Ptr cloud (new Cloud(depth.cols, depth.rows));
  int missing = 0;
  for(int h = 0; h < depth.rows; h++)
  {
    for(int w = 0; w < depth.cols; w++)
    {
      if (depth.at<ushort>(h, w) > 0.0f)
      {
        pcl::PointXYZRGB pt;
        pt.z = depth.at<ushort>(h, w) / factor;
        pt.x = (w - cx) * pt.z / fx;
        pt.y = (h - cy) * pt.z / fy;
        cv::Vec3b col = color.at<cv::Vec3b>(h, w);
        pt.r = col[0];
        pt.g = col[1];
        pt.b = col[2];
        pt.a = 255;
        cloud->at(w, h) = pt;
      } else {
        missing++;
      }
    }
  }
  *out = cloud;
  return missing;
}

int
loadColorCloud(const string & depth_path,
               const string & color_path,
               const string & calibration_id,
               const float factor,
               Cloud::Ptr * out)
{
  cv::Mat color = cv::imread(color_path, 1);
  cv::Mat depth = cv::imread(depth_path, -1);
  const Calibration *cal = &CAMERA_CALIBRATIONS.at(calibration_id);
  int missing = depthToColorCloud(depth, color,
                                  cal->fx, cal->fy, cal->cx, cal->cy,
                                  factor, out);
  return missing;
}

//  matches for the first array
pair<vector<int>, vector<int> >
matchTimestamps(vector<int> & timestamps1, 
                vector<int> & timestamps2,
                int maxDifference)
{
  vector<int> index1;
  vector<int> index2;
  int fails = 0;
  for(uint i=uint(0); i<timestamps1.size(); i++)
  {
    int min_diff = 9999999;
    int match = -1;
    for(uint j=uint(0); j<timestamps2.size(); j++)
    {
      int diff = abs(timestamps1.at(i) - timestamps2.at(j));
      if (min_diff > diff && diff < maxDifference)
      {
        match = timestamps2.at(j);
        min_diff = diff;
      }
    }
    if (match != -1)
    {
      index1.push_back(timestamps1.at(i));
      index2.push_back(match);
    } else {
      fails++;
    }
  }
  cout << "Failed to find matches for " << fails << " pairs" << endl;
  pair<vector<int>, vector<int> > output (index1, index2);
  return output;
}

void
displayPairs(const vector<int> & l1,
             const vector<int> & l2,
             const string & base_path,
             const vector<string> & color_ind,
             const int dataset_num)
{
  const string color_ind_path = base_path + DATASETS[dataset_num] + "rgb.txt";
  //const vector<string> color_ind = loadImageList(color_ind_path);
  for(uint i=0; i<l1.size(); i++)
  {
    const string p1 = color_ind.at(l1.at(i));
    const string p2 = color_ind.at(l2.at(i));
    cout << p1 << endl;
    cout << p2 << endl;
    const cv::Mat im1 = cv::imread(p1, 1);
    const cv::Mat im2 = cv::imread(p2, 1);
    cv::Mat dst;
    cv::hconcat(im1, im2, dst);
    cv::imshow("Pair", dst);
    cv::waitKey(0);
  }
}

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
          const string & test_path)
{
  //const string color_ind_path = base_path + DATASETS[dataset_num] + "rgb.txt";
  //const vector<string> color_ind = loadImageList(color_ind_path);
  int test_percent = 10;
  int val_percent = 5;

  ofstream train_file;
  const string train_path_f = (train_path + set_file_name + "_" + tag + ".txt");
  train_file.open(train_path_f.c_str(), ios::out | ios::trunc);
  ofstream val_file;
  const string val_path_f = (val_path + set_file_name + "_" + tag + ".txt");
  val_file.open(val_path_f.c_str(), ios::out | ios::trunc);
  ofstream test_file;
  const string test_path_f = (test_path + set_file_name + "_" + tag + ".txt");
  test_file.open(test_path_f.c_str(), ios::out | ios::trunc);
  int train_count = 0;
  int val_count = 0;
  int test_count = 0;

  for(uint i=0; i<l1.size(); i++)
  {
    const string p1 = color_ind.at(l1.at(i));
    const string p2 = color_ind.at(l2.at(i));
    int r = rand() % 100 + 1;
    if (r > test_percent + val_percent)
    {
      train_file << p1 << " " << p2 << endl;
      train_file.flush();
      train_count++;
    } else if (r > test_percent) {
      val_file << p1 << " " << p2 << endl;
      val_file.flush();
      val_count++;
    } else {
      test_file << p1 << " " << p2 << endl;
      test_file.flush();
      test_count++;
    }
  }
  cout << train_path_f << " " << train_count << endl;
  cout << val_path_f << " " << val_count << endl;
  cout << test_path_f << " " << test_count << endl;
  train_file.close();
  val_file.close();
  test_file.close();
}
