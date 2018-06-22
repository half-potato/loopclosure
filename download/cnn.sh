mkdir -p raw/CNNImgRetrieval/tars
mkdir -p raw/CNNImgRetrieval/images
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz -P raw/CNNImgRetrieval/tars/
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-30k.mat -P raw/CNNImgRetrieval
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/retrieval-SfM-30k-imagenames-clusterids.mat -P raw/CNNImgRetrieval
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k.mat -P raw/CNNImgRetrieval
wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/retrieval-SfM-120k-imagenames-clusterids.mat -P raw/CNNImgRetrieval

cd raw/CNNImgRetrieval/images
tar -xvzf ../tars/ims.tar.gz
