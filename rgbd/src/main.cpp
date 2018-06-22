#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <utility>
#include <ctype.h>

#include "util.h"
#include "tum.h"
#include "scenenn.h"
#include "scenenn_dataset.h"
#include "processing.h"

using namespace std;

void loadAndSave(bool scenenn,
                 int dataset_num,
                 int skip_num,
                 float threshold,
                 float negative_threshold)
{
  string base_path;
  vector<Cloud::Ptr> clouds;
  Cloud::Ptr main_cloud (new Cloud); 
  vector<string> color_ind;
  vector<int> indices;
  string set_name;
  string set_file_name;
  cout << "Loading" << endl;
  if (scenenn)
  {
    set_name = SCENENN_DATASETS[dataset_num];
    set_file_name = SCENENN_FILES[dataset_num];
    base_path = "raw/SceneNN/data/";
    cout << dataset_num << endl;
    indices = scenenn::loadDataset(dataset_num, "../../", base_path, "asus", skip_num, 
                                           100000, &color_ind, &clouds, &main_cloud);
  } else {
    set_name = DATASETS[dataset_num];
    set_file_name = DATA_FILES[dataset_num];
    base_path = "raw/RGBD/data/";
    indices = tum::loadDataset(dataset_num, "../../"+base_path, "def", skip_num, 
                                           100000, &clouds, &main_cloud);
  }
  cout << clouds.size() << endl;
  if (!scenenn)
    color_ind = tum::loadImageList(base_path + DATASETS[dataset_num] + "rgb.txt",
        base_path + set_file_name);
  
  /*
  pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  viewer.showCloud (main_cloud);
  while (!viewer.wasStopped ())
  {
  }
  */

  vector<int> pos1;
  vector<int> pos2;
  vector<int> neg1;
  vector<int> neg2;
  cout << "Comparing" << endl;
  compareAllClouds(clouds, threshold, negative_threshold, skip_num, indices,
                   &pos1, &pos2, &neg1, &neg2);
  cout << "Found " << pos1.size() << " positive pairs" << endl;
  cout << "Found " << neg1.size() << " negative pairs" << endl;
  //displayPairs(pos1, pos2, "../../"+base_path, color_ind, dataset_num);
  save(pos1, pos2, dataset_num, color_ind,
       set_name, set_file_name,
       "pos",
       base_path,
       "../../indexes/train/",
       "../../indexes/val/",
       "../../indexes/test/");
  save(neg1, neg2, dataset_num, color_ind,
       set_name, set_file_name,
       "neg",
       base_path,
       "../../indexes/train/",
       "../../indexes/val/",
       "../../indexes/test/");
}

int main(int argc, char** args) {
  if (argc != 5)
  {
    cout << "Use: ./rgbd_dataset DATASET_NUM THRESHOLD NEGATIVE_THRESHOLD SKIP_NUM" << endl;
    return 0;
  }
  if (!isdigit(args[1][0]))
  {
    const char *message = "Databases are:\n"
      "0: long/rgbd_dataset_freiburg1_room/\n"
      "1: long/rgbd_dataset_freiburg2_large_with_loop/\n"
      "2: long/rgbd_dataset_freiburg2_pioneer_slam2/\n"
      "3: long/rgbd_dataset_freiburg2_large_no_loop/\n"
      "4: long/rgbd_dataset_freiburg2_pioneer_slam/\n"
      "5: long/rgbd_dataset_freiburg2_pioneer_slam3/\n";
    cout << message << endl;
    return 0;
  }
  if (!isdigit(args[2][0]))
  {
    cout << "Second argument must be the threshold as a float. Recommended value of 0.5-0.6" << endl;
    return 0;
  }

  if (!isdigit(args[3][0]))
  {
    cout << "Third argument must be the negative threshold as a float. Recommended value of 0.05" << endl;
    return 0;
  }

  if (!isdigit(args[4][0]))
  {
    cout << "Fourth argument must be the number of pointclouds skipped every iteration as an int. Recommended value of 60 for TUMs" << endl;
    return 0;
  }

  int dataset_num = atoi(args[1]);
  float threshold = atof(args[2]);
  float negative_threshold = atof(args[3]);
  int skip_num = atoi(args[4]);
  bool scenenn = true;
  int len;
  if (scenenn)
    len = S_LEN;
  else
    len = LEN;
  if (dataset_num >= len)
  {
    for (int i=0; i<len; i++)
    {
      loadAndSave(scenenn, i, skip_num, threshold, negative_threshold);
    }
    return 0;
  } else {
    loadAndSave(scenenn, dataset_num, skip_num, threshold, negative_threshold);
    return 0;
  }
}
