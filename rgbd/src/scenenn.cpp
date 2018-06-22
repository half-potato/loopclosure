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
#include <iomanip>

#include "util.h"
#include "scenenn.h"
#include "scenenn_dataset.h"

using namespace std;

/*
 * File format
 * image/image000000.png
 * depth/depth000000.png
 * timestamp.txt
 * frame - color timestamp - depth timestamp
 *
 * trajectory.txt
 * frame id - col frame id - dep frame id
 */

vector<Eigen::Matrix4f> 
scenenn::loadGroundtruth(const string & path,
                         vector<vector<int> > * frame_ids)
{
  vector<Eigen::Matrix4f> transforms;
  ifstream file;
  file.open(path.c_str());
  string line;
  int j = 0; // matrix row counter
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  while (getline(file, line))
  {
    stringstream stream(line);
    string element;
    vector<float> arr;
    vector<int> ids;
    while (stream >> element)
    {
      arr.push_back(atof(element.c_str()));
    }
    if (arr.size() == 4)
    {
      Eigen::Vector4f v2(arr.data());
      transform.block<1, 4>(j, 0) = v2;
      j++;
      if (j >= 4) 
      {
        j = 0;
        transforms.push_back(transform);
        transform = Eigen::Matrix4f::Identity();
      }
    } else {
      vector<int> ids (arr.begin(), arr.end());
      frame_ids->push_back(ids);
    }
  }
  return transforms;
}

void
scenenn::makeImageList(const string & path,
                       const vector<vector<int> > & frame_ids,
                       vector<string> * color_paths,
                       vector<string> * depth_paths)
{
  ostringstream convert;
  cout << path << endl;
  for(uint i=0; i<frame_ids.size(); i++)
  {
    // Color image
    convert << setfill('0') << setw(5) << frame_ids.at(i).at(1)+1;
    string col_p = path + "image/image" + convert.str() + ".png";
    convert.str("");
    convert.clear();
    color_paths->push_back(col_p);
    // Depth image
    convert << setfill('0') << setw(5) << frame_ids.at(i).at(2)+1;
    string dep_p = path + "depth/depth" + convert.str() + ".png";
    convert.str("");
    convert.clear();
    depth_paths->push_back(dep_p);
  }
}

vector<int>
scenenn::loadDataset(const int dataset_num,
                     const string & base_path,
                     const string & image_path,
                     const string & calib_id, // should always be asus
                     const uint skip_num,
                     const int max_missing_points,
                     vector<string> * color_paths,
                     vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> * clouds,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr * main_cloud)
{
  // Load lists of images and transforms
  vector<vector<int> > frame_ids;
  string data_path = base_path + image_path + SCENENN_DATASETS[dataset_num];
  //scenenn::loadGroundtruth("../../raw/SceneNN/data/005/trajectory.log", &frame_ids);
  string gt_path = data_path + "trajectory.log";
  vector<Eigen::Matrix4f> gt = scenenn::loadGroundtruth(gt_path, &frame_ids);
  //vector<string> color_paths;
  vector<string> depth_paths;
  scenenn::makeImageList(image_path + SCENENN_DATASETS[dataset_num], frame_ids,
      color_paths, &depth_paths);

  int removed = 0;
  vector<int> indices;
  for(uint i=0; i<depth_paths.size()-skip_num+uint(1); i+=skip_num)
  {
    Cloud::Ptr cloud (new Cloud);
    int missing = loadColorCloud(base_path+depth_paths.at(i), base_path+color_paths->at(i),
                                 calib_id, 1000, &cloud);
    if (missing > max_missing_points)
    {
      removed++;
      continue;
    }
    indices.push_back(i);
    Cloud::Ptr transformed (new Cloud);
    Eigen::Matrix4f transform = gt.at(i);//.inverse();
    pcl::transformPointCloud(*cloud, *transformed, transform);
    **main_cloud += *transformed;

    clouds->push_back(transformed);
  }
  cout << "Removed " << removed << endl;
  return indices;
}
