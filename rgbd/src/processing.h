#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

#include <vector>
#include <cstdlib>
#include <utility>
#include <algorithm>

#include "util.h"

using namespace std;

pair<vector<vector<pcl::Vertices> >, vector<Cloud::Ptr> >
preprocess(const vector<Cloud::Ptr> clouds);

float
cloudOverlap(const Cloud::Ptr & hull1, 
             const Cloud::Ptr & hull2,
             const vector<pcl::Vertices> & verts1,
             const Cloud::Ptr & cloud2);

void
compareAllClouds(const vector<Cloud::Ptr> clouds,
                 const float overlapThres,
                 const float negativeThres,
                 const uint skip_num,
                 const vector<int> indices,
                 vector<int> * pos_list1,
                 vector<int> * pos_list2,
                 vector<int> * neg_list1,
                 vector<int> * neg_list2);
