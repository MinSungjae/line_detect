#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <vector>
#include <geometry_msgs/Pose2D.h>

using namespace std;
using namespace cv;

//White ROI
int roi_x = 2250;
int roi_y = 400;
int roi_width = 1000;
int roi_height = 350;

//White Color Setting
int yellow_hue_low = 13;
int yellow_hue_high = 30;
int yellow_sat_low = 55;
int yellow_sat_high = 200;
int yellow_val_low = 40;
int yellow_val_high = 125;

int main(int argc, char** argv)
{
  // Node Name : yellow_detect
  ros::init(argc, argv, "yellow_detect");

  ros::NodeHandle nh;

  /*  fitLine_msg
    Type : geometry_msgs/Pose2D

    msg.x = b.x   (x value of point on line)
    msg.y = b.y   (y value of point on line)
    msg.theta = m (Radian degree of line)
  */
  geometry_msgs::Pose2D fitLine_msg;
  ros::Publisher pub_fitLine = nh.advertise<geometry_msgs::Pose2D>("/yellow_detect/yellow_line_pos", 1000);

  // Set Publishers & Sublscribers
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("/yellow_detect/yellow_detect_img", 1);
  image_transport::Publisher pub2 = it.advertise("/yellow_detect/yellow_img", 1);
  image_transport::Subscriber sub = it.subscribe("/mindvision1/image_rect_color", 1,
  [&](const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge Error ! : %s", e.what());
      return;
    }

    // Recognize Slope angle tolerance
    int slope_tor = 75;
    // Recognize Slope angle treshold (-45 deg ~ 45deg)
    double slope_treshold = (90 - slope_tor) * CV_PI / 180.0;

    Mat img_hsv, yellow_mask, img_yellow, img_edge;
    Mat frame = cv_ptr->image;
    Mat grayImg, blurImg, edgeImg, copyImg;

    Point pt1, pt2;
    vector<Vec4i> lines, selected_lines;
    vector<double> slopes;
    vector<Point> pts;
    Vec4d fit_line;

    Rect bounds(0, 0, frame.cols, frame.rows);
    Rect roi(roi_x, roi_y, roi_width, roi_height);
    frame = frame(bounds & roi);

    // Color Filtering

    cvtColor(frame, img_hsv, COLOR_BGR2HSV);
    inRange(img_hsv, Scalar(yellow_hue_low, yellow_sat_low, yellow_val_low) , Scalar(yellow_hue_high, yellow_sat_high, yellow_val_high), yellow_mask);
    bitwise_and(frame, frame, img_yellow, yellow_mask);
    medianBlur(img_yellow, img_yellow, 5);
    img_yellow.copyTo(copyImg);

    // Canny Edge Detection
    cvtColor(img_yellow, img_yellow, COLOR_BGR2GRAY);
    // Canny(img_yellow, img_edge, 30, 60);
    // Sobel(img_yellow, img_edge, img_yellow.type(), 1, 0, 3);
    // Scharr(img_yellow, img_edge, img_yellow.type(), 1, 0, 3);
    Mat img_edgeXp;
    Mat prewittP = (Mat_<int>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    filter2D(img_yellow, img_edgeXp, img_yellow.type(), prewittP);

    Mat img_edgeXm;
    Mat prewittM = (Mat_<int>(3,3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    filter2D(img_yellow, img_edgeXm, img_yellow.type(), prewittM);

    img_edge = img_edgeXp + img_edgeXm;
    // medianBlur(edges, img_edge, 9);



    // Line Dtection
    HoughLinesP(img_edge, lines, 1, CV_PI / 180 , 50, 50, 25);

    //cout << "slope treshol : " << slope_treshold << endl;

    for(size_t i = 0; i < lines.size(); i++)
    {
      Vec4i line = lines[i];
      pt1 = Point(line[0] , line[1]);
      pt2 = Point(line[2], line[3]);

      double slope = (static_cast<double>(pt1.y) - static_cast<double>(pt2.y)) / (static_cast<double>(pt1.x) - static_cast<double>(pt2.x) );
      //cout << slope << endl;

      cv::line(frame, Point(pt1.x, pt1.y) , Point(pt2.x , pt2.y) , Scalar(0,255,0) , 2 , 8);
      if(abs(slope) >= slope_treshold)
      {
        selected_lines.push_back(line);
        pts.push_back(pt1);
        pts.push_back(pt2);
      }
    }

    if(pts.size() > 0)
    {
      fitLine(pts, fit_line, DIST_L2, 0, 0.01, 0.01);

      double m = fit_line[1] / fit_line[0];
      Point b = Point(fit_line[2], fit_line[3]);

      int pt1_y = frame.rows;
      int pt2_y = 0;

      double pt1_x = ((pt1_y - b.y) / m) + b.x;
      double pt2_x = ((pt2_y - b.y) / m) + b.x;

      //cout << "slope : " << (static_cast<double>(pt1_y) - static_cast<double>(pt2_y)) / (static_cast<double>(pt1_x) - static_cast<double>(pt2_x)) << endl;

      line(frame, Point(pt1_x, pt1_y) , Point(pt2_x , pt2_y) , Scalar(0,0,255) , 2 , 8);

      fitLine_msg.x = b.x;
      fitLine_msg.y = b.y;
      fitLine_msg.theta = m;
    }


    for(size_t i = 0; i < selected_lines.size(); i++)
    {
      //cout << "i : " << i << endl;
      Vec4i I = selected_lines[i];
      line(frame, Point(I[0], I[1]), Point(I[2], I[3]) , Scalar(255,0,0) , 2 , 8);
    }


    //ROS_INFO("cols : %d , rows : %d" , frame.cols, frame.rows);
    sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    sensor_msgs::ImagePtr pub_msg2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", copyImg).toImageMsg();
    pub.publish(pub_msg);
    pub2.publish(pub_msg2);
    pub_fitLine.publish(fitLine_msg);

  });



  ros::spin();

  return 0;

}
