# Image-Dehazing
 Image Haze Removal Using Dark Channel Prior And Color Attenuation Prior
程序图形界面主要是利用Qt进行搭建的，利用Qt Creator进行界面设计，其中有很多组件方便实现相关功能，用c++和opencv完成图像去雾相应函数的编写以及相应回调函数的编写。
基本函数实现均在mainwindow.cpp中完成。导向滤波函数部分在guidedfilter.cpp中实现。
函数主要包括显示图像函数，打开图像，保存图像函数以及去雾相关的函数。
暗通道去雾的主要实现函数是HazeRemoval (Mat &source, Mat &output, double w, int minr, int maxA, int guider, double guideeps, int L)；
颜色衰减先验去雾的主要实现函数是dehazing_CAP(const Mat &Img, Mat &J, double beta, int r)。具体参数含义见源代码相应注释。
