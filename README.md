# loopclosure-scripts

# Setup
To train on your computer:

install tensorflow

sudo pip3 install utm python-opencv opencv opencv-python scipy h5py matplotlib python3-tk scipy

sudo apt install python3-tk python-scipy libpcl1.7 build-essential cmake libopencv-dev libpcl-dev libproj-dev

### To add rgbd examples
cd rgbd/build

cmake ..

./rgbd\_dataset 8 0.60 0.05 60

cd ../..

# Downloading dataset

bash download/download\_all.sh

python scripts/gen.py

# Training

Check which setting number to use in src/constants.py

src/train.py SETTING\_NUMBER NUMBER\_OF\_STEPS

Train all using 

src/train.py all NUMBER\_OF\_STEPS

# Testing

The dataset number is 0 for Lip6Indoor and 1 for Lip6Outdoor
Again, check which setting number to use in src/constants.py

python scripts/test\_lipdataset.py SETTING\_NUMBER DATASET\_NUMBER

To test all models, run the command:

python scripts/test\_lipdataset.py all DATASET\_NUMBER

# TODO

Process SceneNN

Add way to download Streetview

Train all networks and test them

Staging to speed up training

Fix image stat calculations

# Notes

Use ESC to exit the viewer

Varying the ratio may make it train more quickly. Investigate more.
 it train more quickly. Investigate more.
