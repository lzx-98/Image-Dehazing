#include "mainwindow.h"
#include "ui_mainwindow.h"
#include"guidedfilter.h"
#include<QDebug>
#include<QTime>
#include<QGraphicsView>
#include<QGraphicsScene>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QFont font("Arial", 12, 50);                //设置字体大小和效果
    //ui->label_w->setFont(font);
//    Mat image=imread("E:\\edc.JPEG",1);
//    namedWindow( "Display window", WINDOW_AUTOSIZE );
//    imshow( "Display window", image );
}

MainWindow::~MainWindow()
{
    delete ui;
}
QImage MainWindow::Mat2Qimage(Mat& mat)
{
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i ,i));
        }
        uchar *psrc = mat.data;
        for(int row = 0; row < mat.rows; row++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, psrc, mat.cols);
            psrc += mat.step;
        }
        return image;
    }
    else if(mat.type() == CV_8UC3)
    {
        const uchar *psrc = (const uchar*)mat.data;
        QImage image(psrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        const uchar* psrc = (const uchar*)mat.data;
        QImage image(psrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();


//    cvtColor(mat, mat, COLOR_BGR2RGB);
//    QImage qim((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step,
//        QImage::Format_RGB888);
//    return qim;

}
void MainWindow::on_actionopen_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open", ".", "Image files(*.bmp *.jpg *.png *.jpeg)");
    src = imread(fileName.toStdString());
    dst = src.clone();
    dstImg = Mat2Qimage(dst);
    QGraphicsScene *scene = new QGraphicsScene;
    scene ->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView -> setScene(scene);
    //ui->graphicsView->resize(dstImg.width() + 10, dstImg.height() + 10);
    //ui->graphicsView -> setScaleContents(true);
    ui->graphicsView -> show();
    ui->label->setPixmap(QPixmap::fromImage(dstImg));
    ui->label->setAlignment(Qt::AlignCenter);               //居中显示
    //ui->label->setScaledContents(true);
}

void MainWindow::on_actionsave_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save", ".", "Image files(*.bmp *.jpg *.png)");
    imwrite(fileName.toStdString(), dst);
}

void MainWindow::on_actionrecover_triggered()
{
    dstImg = Mat2Qimage(src);
    QGraphicsScene *scene = new QGraphicsScene;
    scene->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView -> setScene(scene);
    ui->graphicsView -> show();
}

void MainWindow::on_actiontest_triggered()
{
    //暗通道去雾图像显示
    dst = src.clone();
    Mat J;
    HazeRemoval(dst,J);
    J1 = J;
    dstImg = Mat2Qimage(J);
    QGraphicsScene *scene = new QGraphicsScene;
    scene ->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView_2 -> setScene(scene);
    ui->graphicsView_2 -> show();
    dstImg = Mat2Qimage(Re_t);
    ui->label->setPixmap(QPixmap::fromImage(dstImg));
    //ui->label->setScaledContents(true);
}

void MainWindow::MinFilter(Mat &source, Mat &output, int r)
{
    Mat input;
    source.copyTo(input);

    output.create(source.rows, source.cols, CV_8U);
    for (int i = 0; i <= (input.rows - 1) / r; i++)
    {
        for (int j = 0; j <= (input.cols - 1) / r; j++)
        {
            int w = r;
            int h = r;
            if (i * r + h > input.rows)
            {
                h = input.rows - i * r;
            }
            if (j * r + w > input.cols)
            {
                w = input.cols - j * r;
            }

            Mat ROI = input(Rect(j * r, i * r, w, h));

            double mmin;
            minMaxLoc(ROI, &mmin, 0);

            Mat desROI = output(Rect(j * r, i * r, w, h));
            desROI.setTo(uchar(mmin));
        }
    }
}

void MainWindow::darkChannel(Mat& source, Mat& output, int r)
{
    //彩色图像先取每个像素RGB三个通道内的最小值，再进行最小值滤波
    Mat input;
    input.create(source.rows, source.cols, CV_8U);

    for (int i = 0; i < source.rows; i++)
    {
        uchar *sourcedata = source.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        for (int j = 0; j < source.cols * source.channels(); j += 3)
        {
            uchar mmin;
            mmin = min(sourcedata[j], sourcedata[j + 1]);
            mmin = min(mmin, sourcedata[j + 2]);

            indata[j / 3] = mmin;
        }
    }

    MinFilter(input, output, r);
}

void MainWindow::makeDepth32f(Mat &source, Mat &output)
{
    if (source.depth() != CV_32F)
        source.convertTo(output, CV_32F);
    else
        output = source;
}

void MainWindow::mynorm(Mat &source, Mat &output)
{
    for (int i = 0; i < source.rows; i++)
    {
        float *indata = source.ptr<float>(i);
        float *outdata = output.ptr<float>(i);
        for (int j = 0; j < source.cols * source.channels(); j++)
        {
            outdata[j] = indata[j] / 255.0;
        }
    }
}

void MainWindow::GuideFilter1(Mat &source, Mat &guided_image, Mat &output, int radius, double epsilon)      //epsilon为截断值
{
    //CV_Assert ()中值为false,返回一个错误信息
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    //I为导向图(单通道)
    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    //将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    mynorm(source_32f, source_32f);
    makeDepth32f(guided, guided_32f);
    mynorm(guided_32f, guided_32f);

    //计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiply(guided_32f, source_32f, mat_Ip);
    multiply(guided_32f, guided_32f, mat_I2);

    //计算各种均值:p,I,Ip,I2
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    //计算Ip的协方差cov_Ip和I的方差var_I
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);               //a=cov_Ip/(var_I+epsilon)
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);             //积分图算法实现的boxfilter
    boxFilter(b, mean_b, CV_32F, win_size);

    //计算输出 (depth == CV_32F)   q=aI+b
    Mat tempoutput = mean_a.mul(guided_32f) + mean_b;

    output.create(source.rows, source.cols, CV_8U);

    for (int i = 0; i < source.rows; i++)
    {
        float *data = tempoutput.ptr<float>(i);
        uchar *outdata = output.ptr<uchar>(i);
        for (int j = 0; j < source.cols; j++)
        {
            outdata[j] = saturate_cast<uchar>(data[j] * 255);
        }
    }
}

void MainWindow::HazeRemoval(Mat &source, Mat &output, double w, int minr, int maxA, int guider, double guideeps, int L)
{
    QTime time;
    time.start();
    Mat input;
    source.copyTo(input);

    //获得图像的暗通道
    Mat dark;
    darkChannel(input, dark, minr * 2 + 1);         //窗口大小为15*15

    //计算大气光值Ac
    int hash[256];
    memset(hash, 0, sizeof(hash));
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = dark.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            hash[data[j]]++;
        }
    }
    int num = dark.rows * dark.cols / 1000.0;
    int count = 0;
    uchar thres;
    for (int i = 0; i < 256; i++)
    {
        count += hash[255 - i];
        if (count >= num)
        {
            thres = 255 - i;
            break;
        }
    }
    num = count;
    double b_max = 0, B;
    double g_max = 0, G;
    double r_max = 0, R;
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = dark.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            if (data[j] >= thres)
            {
                B = indata[3 * j];
                G = indata[3 * j + 1];
                R = indata[3 * j + 2];
                b_max += B;
                g_max += G;
                r_max += R;
            }
        }
    }
    b_max /= num;
    g_max /= num;
    r_max /= num;
    //限定了大气光值的上限
    uchar MMAX = maxA;
    if (b_max > MMAX) b_max = MMAX;
    if (g_max > MMAX) g_max = MMAX;
    if (r_max > MMAX) r_max = MMAX;

    //计算得到粗透射率
    Mat img_t;          //保存粗透射率图
    img_t.create(dark.rows, dark.cols, CV_8U);
    Mat temp;
    temp.create(dark.rows, dark.cols, CV_8UC3);
    double b_temp = b_max / 255;
    double g_temp = g_max / 255;
    double r_temp = r_max / 255;
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *data = input.ptr<uchar>(i);
        uchar *tdata = temp.ptr<uchar>(i);
        for (int j = 0; j < dark.cols * 3; j += 3)
        {
            tdata[j] = saturate_cast<uchar>(data[j] / b_temp);          //saturate_cast溢出保护
            tdata[j + 1] = saturate_cast<uchar>(data[j] / g_temp);
            tdata[j + 2] = saturate_cast<uchar>(data[j] / r_temp);
        }
    }
    Mat gray;
    cvtColor(temp, gray, CV_BGR2GRAY);
    darkChannel(temp, temp, minr * 2 + 1);
    for (int i = 0; i < dark.rows; i++)
    {
        uchar *darkdata = temp.ptr<uchar>(i);
        uchar *tdata = img_t.ptr<uchar>(i);
        for (int j = 0; j < dark.cols; j++)
        {
            tdata[j] = 255 - w * darkdata[j];                   //w一般取0.95，保留一定量的雾
        }
    }
    //导向滤波细化透射率得到细透射率
    GuideFilter1(img_t, gray, img_t, guider, guideeps);         //gray作为导向图,细化透射率图仍保存在img_t
    Re_t = img_t;

    //还原图像
    output.create(input.rows, input.cols, CV_8UC3);     //去雾图像
    for (int i = 0; i < input.rows; i++)
    {
        uchar *tdata = img_t.ptr<uchar>(i);
        uchar *indata = input.ptr<uchar>(i);
        uchar *outdata = output.ptr<uchar>(i);      //指向第i+1行第一个元素的指针
        for (int j = 0; j < input.cols; j++)
        {
            uchar b = indata[3 * j];
            uchar g = indata[3 * j + 1];
            uchar r = indata[3 * j + 2];
            double t = tdata[j];
            t /= 255;
            if (t < 0.1) t = 0.1;                   //保证透射率t的下限值0.1，在浓雾区保留一定的雾

            outdata[3 * j] = saturate_cast<uchar>((b - b_max) / t + b_max + L);
            outdata[3 * j + 1] = saturate_cast<uchar>((g - g_max) / t + g_max + L);
            outdata[3 * j + 2] = saturate_cast<uchar>((r - r_max) / t + r_max + L);

            ui->timelabel->setText("Time: " + QString::number(time.elapsed()) + " ms");
            ui->timelabel->setAlignment(Qt::AlignCenter);                       //居中显示
        }
    }
}

void MainWindow::on_actionCAP_triggered()
{

    dst = src.clone();
    Mat J;
    dehazing_CAP(dst, J);
    J2 = J;
    //namedWindow( "Display window", WINDOW_AUTOSIZE );
    //imshow( "Display window", J );
    qDebug() << J.type();               //CV_32FC3
    dstImg = Mat2Qimage(J);
    QGraphicsScene *scene = new QGraphicsScene;
    scene ->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView_3 -> setScene(scene);
    ui->graphicsView_3 -> show();
}

void MainWindow::calVSMap(const Mat &I, int r, Mat &dR, Mat &dP)
{
    Mat hsvI, fI;
    I.convertTo(fI, CV_32FC3);              //32位浮点型
    cvtColor(fI/255.0, hsvI, COLOR_BGR2HSV);

    std::vector<Mat> hsv_vec;
    split(hsvI, hsv_vec);
    Mat output;
    addWeighted(hsv_vec[1], -0.780245, hsv_vec[2], 0.959710, 0.121779, output);     //进行点预算，图像融合
//    cv::addWeighted(hsv_vec[1], -1.2966, hsv_vec[2], 1.0267, 0.1893, output);
    dP = output;
    Mat se = getStructuringElement(MORPH_RECT, Size(r, r));
    Mat outputRegion;
    erode(output, outputRegion, se, Point(-1,-1), 1, BORDER_REFLECT);               //Point(x,y) x表示列，y表示行
    dR = outputRegion;
}

std::vector<double> MainWindow::estA(const Mat &img, Mat &Jdark){
    Mat img_norm = img/255.0;
    double n_bright = ceil(img_norm.rows * img_norm.cols * 0.001);		//向上舍入为最接近的整数
    Mat Jdark_val = Jdark.reshape(1, 1);
    Mat Loc;
    sortIdx(Jdark_val, Loc, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);		//返回排序后对应原矩阵的索引

    Mat Ics = img_norm.reshape(3, 1);
    Mat Acand(1, n_bright, CV_32FC3);
    Mat Amag(1, n_bright, CV_32F);
    for(int i = 0; i < n_bright; i++)
    {
        float b = Ics.at<Vec3f>(0, Loc.at<int>(0, i))[0];
        float g = Ics.at<Vec3f>(0, Loc.at<int>(0, i))[1];
        float r = Ics.at<Vec3f>(0, Loc.at<int>(0, i))[2];

        Acand.at<Vec3f>(0, i)[0] = b;
        Acand.at<Vec3f>(0, i)[1] = g;
        Acand.at<Vec3f>(0, i)[2] = r;
        Amag.at<float>(0, i) = b*b + g*g + r*r;
    }

    Mat Loc2;
    sortIdx(Amag, Loc2, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
    Mat A_arr(1, std::min(20.0, n_bright), CV_32FC3);
    for(int i = 0; i < std::min(20.0, n_bright); i++){
        A_arr.at<Vec3f>(0, i)[0] = Acand.at<Vec3f>(0, Loc2.at<int>(0, i))[0];
        A_arr.at<Vec3f>(0, i)[1] = Acand.at<Vec3f>(0, Loc2.at<int>(0, i))[1];
        A_arr.at<Vec3f>(0, i)[2] = Acand.at<Vec3f>(0, Loc2.at<int>(0, i))[2];
    }

    std::vector<Mat> A_vec;
    split(A_arr, A_vec);
    double max1, max2, max3;
    minMaxLoc(A_vec[0], NULL, &max1);
    minMaxLoc(A_vec[1], NULL, &max2);
    minMaxLoc(A_vec[2], NULL, &max3);

    std::vector<double> A(3);
    A[0] = max1;
    A[1] = max2;
    A[2] = max3;

    return A;
}

void MainWindow::dehazing_CAP(const Mat &Img, Mat &J, double beta, int r)
{
    QTime time;
    time.start();
    Mat I = Img;
    I.convertTo(I, CV_32FC3);
    Mat dR(I.rows, I.cols, CV_32FC3);
    Mat dP;
    //int r = 15;
    //double beta = 1.0;

    calVSMap(I, r, dR, dP);

    Mat p = dP.clone();     //TODO
    double eps = 0.001;
    I.convertTo(I, CV_32FC3);
    Mat refineDR = guidedFilter(I/255.0, p, r, eps);		//导向滤波

    Mat tR1, tR;
    cv::exp(-beta * refineDR, tR1);
    double t0 = 0.05;
    double t1 = 1;
    Mat t = tR1.clone();
    for(int h = 0; h < t.rows; h++)
    {
        for(int w = 0; w < t.cols; w++)
        {
            if(t.at<float>(h, w) < t0)
                t.at<float>(h, w) = t0;
            else if(t.at<float>(h, w) > t1)
                t.at<float>(h, w) = t1;
        }
    }
    std::vector<Mat> tR_vec;
    tR_vec.push_back(t);
    tR_vec.push_back(t);
    tR_vec.push_back(t);
    merge(tR_vec, tR);

    std::vector<double> a;
    a = estA(I, dR);
    Mat A(I.rows, I.cols, CV_32FC3, Scalar(a[0], a[1], a[2]));

    Mat J1, J2;
    I = I/255.0;
    scaleAdd(A, -1, I, J1);
    divide(J1, tR, J2);
    add(J2, A, J);
    J.convertTo(J, CV_8UC3, 255);

    ui->timelabel->setText("Time: " + QString::number(time.elapsed()) + " ms");
    ui->timelabel->setAlignment(Qt::AlignCenter);                       //居中显示
    }

Scalar MainWindow::getMSSIM(Mat  hazeimage, Mat dehazeimage)
{
    Mat i1 = hazeimage;
    Mat i2 = dehazeimage;
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);
    return mssim;
}

double MainWindow::getMSM(const Mat &reimage, const Mat &dehazeimage)
{
    Mat s1;
    absdiff(reimage, dehazeimage, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double mse = (s.val[0] + s.val[1] + s.val[2])/(3 * dehazeimage.cols * dehazeimage.rows);
    mse = sqrt(mse);
    return mse;
}

//暗通道改变参数w的去雾效果对比(w默认值为0.95)
void MainWindow::on_pushButtonw_clicked()
{
    dst = src.clone();
    Mat J;
    QString str = ui->wEdit->text (); //读取w的值
    double w = str.toDouble();
    HazeRemoval(dst,J, w);
    J1 = J;
    //namedWindow( "Display window", WINDOW_AUTOSIZE );
    //imshow( "Display window", J );
    //qDebug() << J.type();               //CV_32FC3
    dstImg = Mat2Qimage(J);
    QGraphicsScene *scene = new QGraphicsScene;
    scene ->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView_2 -> setScene(scene);
    ui->graphicsView_2 -> show();
}

//颜色衰减先验改变参数β的值(β默认值为1.0)
void MainWindow::on_Button_Beta_clicked()
{
    dst = src.clone();
    Mat J;
    QString str = ui->BetaEdit->text (); //读取beta的值
    double beta = str.toDouble();
    dehazing_CAP(dst,J, beta);
    J2 = J;
    dstImg = Mat2Qimage(J);
    QGraphicsScene *scene = new QGraphicsScene;
    scene ->addPixmap(QPixmap::fromImage(dstImg));
    ui->graphicsView_3 -> setScene(scene);
    ui->graphicsView_3 -> show();
}

//保存暗通道去雾后的图像
void MainWindow::on_actionsave_darkC_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save", ".", "Image files(*.bmp *.jpg *.png)");
    imwrite(fileName.toStdString(), J1);
}

//保存颜色衰减先验去雾图像
void MainWindow::on_actionsave_CAP_triggered()
{
    QString fileName = QFileDialog::getSaveFileName(this,"Save", ".", "Image files(*.bmp *.jpg *.png)");
    imwrite(fileName.toStdString(), J2);
}

void MainWindow::on_actionSSIM_triggered()
{
    //J1是暗通道去雾图，J2是CAP去雾图
    Scalar SSIM_DCP = getMSSIM(re_I, J1);
    Scalar SSIM_CAP = getMSSIM(re_I, J2);
    double SSIM_dcp = (SSIM_DCP.val[0] + SSIM_DCP.val[1] + SSIM_DCP.val[2]) / 3;
    double SSIM_cap = (SSIM_CAP.val[0] + SSIM_CAP.val[1] + SSIM_CAP.val[2]) / 3;
    ui->SSIM_label->setText("SSIM_dcp: " + QString::number(SSIM_dcp) + "\n" + "SSIM_cap: " + QString::number(SSIM_cap));
    //ui->SSIM_label->setText("SSIM_cap: " + QString::number(SSIM_cap));

}

void MainWindow::on_actionopen_reimage_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Open", ".", "Image files(*.bmp *.jpg *.png *.jpeg)");
    re_I = imread(fileName.toStdString());
}

void MainWindow::on_actionMSE_triggered()
{
    double MSE_dcp = getMSM(re_I, J1);
    double MSE_cap = getMSM(re_I, J2);
    ui->MSE_label->setText("MSE_dcp: " + QString::number(MSE_dcp) + "\n" + "MSE_cap: " + QString::number(MSE_cap));
}
