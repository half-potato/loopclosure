#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

#include <vector>
#include <cstdlib>
#include <utility>
#include <algorithm>

#include "processing.h"
#include "util.h"

using namespace std;

pair<vector<vector<pcl::Vertices> >,
          vector<Cloud::Ptr> >
preprocess(const vector<Cloud::Ptr> clouds)
{
  vector<vector<pcl::Vertices> > verts;
  vector<Cloud::Ptr> vert_clouds;
  pcl::ConvexHull<pcl::PointXYZRGB> hull;
  for(uint i=0; i<clouds.size(); i++)
  {
    vector<pcl::Vertices> polys;
    Cloud::Ptr surfacePoints (new Cloud);
    hull.setInputCloud(clouds.at(i));
    hull.reconstruct(*surfacePoints, polys);
    verts.push_back(polys);
    vert_clouds.push_back(surfacePoints);
  }
  pair<vector<vector<pcl::Vertices> >,
            vector<Cloud::Ptr> > output (verts, vert_clouds);
  return output;
}

float
cloudOverlap(const Cloud::Ptr & hull1, 
             const Cloud::Ptr & hull2,
             const vector<pcl::Vertices> & verts1,
             const Cloud::Ptr & cloud2)
{
  pcl::CropHull<pcl::PointXYZRGB> cropper;
  vector<int> indices;
  cropper.setInputCloud(hull2);
  cropper.setHullCloud(hull1);
  cropper.setHullIndices(verts1);
  cropper.filter(indices);
  return float(indices.size()) / hull2->size();
}

void
compareAllClouds(const vector<Cloud::Ptr> clouds,
                 const float overlapThres,
                 const float negativeThres,
                 const uint skip_num,
                 const vector<int> indices,
                 vector<int> * pos_list1,
                 vector<int> * pos_list2,
                 vector<int> * neg_list1,
                 vector<int> * neg_list2)
{
  pair<vector<vector<pcl::Vertices> >,
            vector<Cloud::Ptr> > hulls = preprocess(clouds);

  for(uint i=0; i<indices.size(); i++)
  {
    for(uint j=0; j<indices.size()-1; j++)
    {
      if (i==j)
        j++;
      float val = cloudOverlap(hulls.second.at(i), hulls.second.at(j),
                               hulls.first.at(i), clouds.at(j));
      //cout << val << endl;
      if (val > overlapThres)
      {
        pos_list1->push_back(indices.at(i));
        pos_list2->push_back(indices.at(j));
      }
      if (val <= negativeThres)
      {
        neg_list1->push_back(indices.at(i));
        neg_list2->push_back(indices.at(j));
      }
    }
  }
}
