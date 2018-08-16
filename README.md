# accessmath-textboxes

This project is meant as a helper for our [main work](https://github.com/bhargavaurala/accessmath-icfhr2018) for detection of handwritten whiteboard content in lecture videos. It is a fork of [TextBoxes](https://github.com/MhLiao/TextBoxes).

# Setting up Textboxes

1. Clone this repository. We will call the clone directory `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/bhargavaurala/accessmath-textboxes.git
  
  cd accessmath-textboxes
  ```
2. Edit the makefile configuration file `Makefile.config` according to your system needs. Refer Caffe installation [instructions](http://caffe.berkeleyvision.org/installation.html) for details about dependencies. Make sure that the python wrapper dependencies are installed since we need that for this project. This code has been tested on Ubuntu 14.04.
  ```
  cp Makefile.config.example Makefile.config
  
  mkdir build
  ```
3. Build caffe, caffe-python and test if build went correctly.
  ```
  make -j8
  
  make py
  
  export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python
  
  python -c "import caffe"
  ``` 
 
 ### Download
1. Models trained on ICDAR 2013: [Dropbox link](https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1qY73XHq)
2. Fully convolutional reduced (atrous) VGGNet: [Dropbox link](https://www.dropbox.com/s/qxc64az0a21vodt/VGG_ILSVRC_16_layers_fc_reduced.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1slQyMiL)
3. Compiled mex file for evaluation(for multi-scale test evaluation: evaluation_nms.m): [Dropbox link](https://www.dropbox.com/s/xtjuwvphxnz1nl8/polygon_intersect.mexa64?dl=0) [BaiduYun link](http://pan.baidu.com/s/1jIe9UWA)
4. Frame version of the AccessMath dataset from [here](https://buffalo.box.com/s/6gklgrotfd5drbxvdw2xtrt9i2ldmt01). Download the 3-part zip archive and extract into a folder called AccessMathVOC and place in [AccessMath-ICFHR18](https://github.com/bhargavaurala/accessmath-icfhr2018) project root.
```
export AM_DATA_DIR=/path/to/AccessMathVOC
```

### Generate training and validation LMDBs.

1. `cd $CAFFE_ROOT/data/AccessMath`
2. `./create_data.sh`
3. This will create train, validation and test LMDBs in `$AM_DATA_DIR/AccessMath/lmdb`

### Train
1. In `models/VGGNet/text/longer_conv_300x300/` Modify `data_param` in the first layer (`data`) in `train.prototxt` and `test.prototxt` as shown below
```
  data_param {
    source: "/path/to/AccessMathVOC/AccessMath/lmdb/AccessMath_train_lmdb"
    batch_size: 32
    backend: LMDB
  }
 ```
2. Use `cd $CAFFE_ROOT/build/tools ./caffe train_net -iterations 10000 -solver models/VGGNet/text/longer_conv_300x300/solver.prototxt -weights /path/to/model_trained_on_icdar2013`
3. You should see around 77.5% as the final validation performance.
4. Transfer the model to `models/text_detection` in the [AccessMath](https://github.com/bhargavaurala/accessmath-icfhr2018) root folder
