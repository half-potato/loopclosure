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

namespace tum {

  map<int, Eigen::Matrix4f> 
  loadGroundtruth(const string & path);

  map<int, string>
  loadImageMap(const string & path);

  vector<string>
  loadImageList(const string & path,
                const string & base_dir);

  vector<int>
  loadDataset(const int dataset_num,
              const string & base_path,
              const string & calib_id,
              const uint skip_num,
              const int max_missing_points,
              vector<Cloud::Ptr> * clouds,
              Cloud::Ptr * main_cloud);
}
