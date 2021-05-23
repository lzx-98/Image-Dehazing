#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include<QPixmap>
#include <QFileDialog>
#include <QTimer>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QImage Mat2Qimage(Mat& mat);
    void darkChannel(Mat& source, Mat& output, int r);
    void GuideFilter1(Mat &source, Mat &guided_image, Mat &output, int radius, double epsilon);
    void makeDepth32f(Mat &source, Mat &output);
    void mynorm(Mat &source, Mat &output);
    void MinFilter(Mat& source, Mat& output, int r);
    void HazeRemoval(Mat &source, Mat &output, double w = 0.95, int minr = 7, int maxA = 220, int guider = 30, double guideeps = 0.001, int L = 0);
    void calVSMap(const Mat &I, int r, Mat &dR, Mat &dP);
    vector<double> estA(const Mat &img, Mat &Jdark);
    void dehazing_CAP(const Mat &Img, Mat &J, double beta = 1.0, int r = 15);
    Scalar getMSSIM(Mat reimage, Mat dehazeimage);
    double getMSM(const Mat &reimage, const Mat &dehazeimage);

private:
    Ui::MainWindow *ui;

public:
    Mat src, dst, J1, J2, Re_t, re_I;
    QImage dstImg;
private slots:

    void on_actionopen_triggered();
    void on_actionsave_triggered();
    void on_actionrecover_triggered();
    void on_actiontest_triggered();
    void on_actionCAP_triggered();
    void on_pushButtonw_clicked();
    void on_Button_Beta_clicked();
    void on_actionsave_darkC_triggered();
    void on_actionsave_CAP_triggered();
    void on_actionSSIM_triggered();
    void on_actionopen_reimage_triggered();
    void on_actionMSE_triggered();
};

#endif // MAINWINDOW_H
