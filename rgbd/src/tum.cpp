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
#include "tum.h"

using namespace std;

map<int, Eigen::Matrix4f> 
tum::loadGroundtruth(const string & path)
{
  //vector<Eigen::Matrix4f> vector;
  map<int, Eigen::Matrix4f> gt_map;
  ifstream file;
  file.open(path.c_str());
  string line;
  while (getline(file, line))
  {
    stringstream stream(line);
    string element;
    vector<float> gt;
    int index = 0;
    int timestamp;
    while (stream >> element)
    {
      if (strstr(element.c_str(), "#") != NULL)
        break;
      if (index == 0)
        timestamp = static_cast<int>(fmod(atof(element.c_str()) * 100000, 1000000000));
      if (index > 0)
        gt.push_back(atof(element.c_str()));
      index++;
    }
    if (gt.size() >= 7)
    {
      Eigen::Quaternion<float> qt = Eigen::Quaternion<float>(
          gt.at(6), -gt.at(3), -gt.at(4), -gt.at(5));
      Eigen::Vector3f trans = Eigen::Vector3f(gt.at(0), gt.at(1), gt.at(2));
      trans *= 1;
      Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
      translation.block<3, 1>(0, 3) = trans;
      //cout << "Trans" << endl;
      //cout << translation << endl;
      Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
      rotation.block<3, 3>(0, 0) = qt.toRotationMatrix().inverse();
      //cout << "Rot" << endl;
      //cout << rotation << endl;

      Eigen::Matrix4f transform = translation*rotation;
      //Eigen::Matrix4f transform = rotation*translation;
      //cout << "Transform" << endl;
      //cout << transform << endl;
      //vector.push_back(transform);
      //cout << gt.at(0) << endl;
      gt_map[timestamp] = transform;

      //vector.push_back(translation);
      //vector.push_back(Eigen::Matrix4f::Identity());
    }
  }
  file.close();
  return gt_map;
}

map<int, string>
tum::loadImageMap(const string & path)
{
  map<int, string> path_map;
  ifstream file;
  file.open(path.c_str());
  string line;
  while (getline(file, line))
  {
    stringstream stream(line);
    string element;
    int index = 0;
    //while (getline(stream, element, ' '))
    int timestamp = 0;
    string path = "";
    while (stream >> element)
    {
      if (strstr(element.c_str(), "#") != NULL)
        break;
      if (index == 1)
        path = element;
      if (index == 0)
      {
        timestamp = static_cast<int>(fmod(atof(element.c_str()) * 100000, 1000000000));
      }
      index++;
    }
    if (timestamp != 0 && path != "")
      path_map[timestamp] = path;
  }
  file.close();
  return path_map;
}

vector<string>
tum::loadImageList(const string & path,
                   const string & base_dir)
{
  vector<string> paths;
  ifstream file;
  file.open(path.c_str());
  string line;
  while (getline(file, line))
  {
    stringstream stream(line);
    string element;
    int index = 0;
    while (stream >> element)
    {
      if (strstr(element.c_str(), "#") != NULL)
        break;
      if (index == 1)
        paths.push_back(base_dir + element);
      index++;
    }
  }
  file.close();
  return paths;
}

vector<int>
tum::loadDataset(
            const int dataset_num,
            const string & base_path,
            const string & calib_id,
            const uint skip_num,
            const int max_missing_points,
            vector<Cloud::Ptr> * clouds,
            Cloud::Ptr * main_cloud)
{
  // Declare paths
  const string gt_path = base_path + DATASETS[dataset_num] + "groundtruth.txt";
  const string color_ind_path = base_path + DATASETS[dataset_num] + "rgb.txt";
  const string depth_ind_path = base_path + DATASETS[dataset_num] + "depth.txt";
  const string base_dir = base_path + DATASETS[dataset_num];

  // Load shit
  map<int, string> color_ind = tum::loadImageMap(color_ind_path);
  map<int, string> depth_ind = tum::loadImageMap(depth_ind_path);
  map<int, Eigen::Matrix4f> gt = tum::loadGroundtruth(gt_path);

  // Get the timestamps for the indexes
  vector<int> color_time;
  for(map<int, string>::iterator it = color_ind.begin(); it != color_ind.end(); it++)
  {
    color_time.push_back(it->first);
  }
  vector<int> depth_time;
  for(map<int, string>::iterator it = depth_ind.begin(); it != depth_ind.end(); it++)
  {
    depth_time.push_back(it->first);
  }
  vector<int> gt_time;
  for(map<int, Eigen::Matrix4f>::iterator it = gt.begin(); it != gt.end(); it++)
  {
    gt_time.push_back(it->first);
  }

  int maxDifference = 2000;

  vector<int> longer_list;
  vector<int> shorter_list;
  // Sync color and depth
  if(color_time.size() > depth_time.size())
  {
    const pair<vector<int>, vector<int> > first_compare = 
      matchTimestamps(depth_time, color_time, maxDifference);
    longer_list = first_compare.first;
    shorter_list = first_compare.second;
  } else {
    const pair<vector<int>, vector<int> > first_compare = 
      matchTimestamps(color_time, depth_time, maxDifference);
    longer_list = first_compare.first;
    shorter_list = first_compare.second;
  }

  // Sync shorter of the two with gt
  const pair<vector<int>, vector<int> > balance_gt = 
    matchTimestamps(longer_list, gt_time, maxDifference);
  longer_list = balance_gt.first;
  vector<int> gt_list = balance_gt.second;

  // Rebalance the shorter with the longer one
  vector<int> color_list;
  vector<int> depth_list;
  const pair<vector<int>, vector<int> > balance_last = 
    matchTimestamps(longer_list, shorter_list, maxDifference);
  if(color_time.size() > depth_time.size())
  {
    depth_list = balance_last.first;
    color_list = balance_last.second;
  } else {
    depth_list = balance_last.first;
    color_list = balance_last.second;
  }

  // Load and transform clouds
  int removed = 0;
  vector<int> indices;
  for(uint i=0; i<depth_list.size()-skip_num+uint(1); i+=skip_num)
  {
    string color_p = base_dir + color_ind[color_list.at(i)];
    string depth_p = base_dir + depth_ind[depth_list.at(i)];
    Eigen::Matrix4f transform = gt[gt_list.at(i)];
    Cloud::Ptr cloud (new Cloud);
    int missing = loadColorCloud(depth_p, color_p, calib_id, 5000, &cloud);
    if (missing > max_missing_points)
    {
      removed++;
      continue;
    }
    indices.push_back(i);
    Cloud::Ptr transformed (new Cloud);
    pcl::transformPointCloud(*cloud, *transformed, transform);
    **main_cloud += *transformed;

    clouds->push_back(transformed);
  }
  cout << "Removed " << removed << endl;
  return indices;
}

