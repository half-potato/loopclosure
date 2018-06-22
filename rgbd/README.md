
sudo apt install libopencv-dev libpcl-dev libvtk6-dev
If you are getting the error "cannot find -lvtkproj4", run:
sudo ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so
