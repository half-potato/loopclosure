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

namespace scenenn {

  vector<Eigen::Matrix4f> 
  loadGroundtruth(const string & path,
                  vector<vector<int> > * frame_ids);

  void
  makeImageList(const string & path,
                const vector<vector<int> > & frame_ids,
                vector<string> * color_paths,
                vector<string> * depth_paths);

  vector<int>
  loadDataset(const int dataset_num,
              const string & base_path,
              const string & image_path,
              const string & calib_id,
              const uint skip_num,
              const int max_missing_points,
              vector<string> * color_paths,
              vector<Cloud::Ptr> * clouds,
              Cloud::Ptr * main_cloud);
}

