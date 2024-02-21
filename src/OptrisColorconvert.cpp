#include "OptrisColorconvert.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> // For putText
#include <opencv2/highgui.hpp> // For imshow, waitKey


#define scaled_image_width 300
#define scaled_image_height 300

namespace optris_drivers2
{

  OptrisColorconvert* _this = nullptr;

  // TODO: these static functions were needed to make image_transport usable. In later versions image_transport might callback member functions.
  void _onThermalDataReceive(const sensor_msgs::msg::Image::ConstSharedPtr & image)
  {
    _this->onThermalDataReceive(image);
  }

  void _onVisibleDataReceive(const sensor_msgs::msg::Image::ConstSharedPtr & image)
  {
    _this->onVisibleDataReceive(image);
  }

  OptrisColorconvert::OptrisColorconvert() : Node("optris_colorconvert"), _camera_info_manager(this)
  {
    _this          = this;
    _bufferThermal = nullptr;
    _resizedBufferThermal = nullptr;
    _bufferVisible = nullptr;
    _frame         = 0;
    int palette    = 6;
    double tMin    = 20.0;
    double tMax    = 40.0;

    evo::EnumOptrisPaletteScalingMethod scalingMethod = evo::eMinMax;

    _iBuilder.setPaletteScalingMethod(scalingMethod);
    _iBuilder.setPalette((evo::EnumOptrisColoringPalette)palette);
    _iBuilder.setManualTemperatureRange((float)tMin, (float)tMax);

    rmw_qos_profile_t profile = rmw_qos_profile_default;
      
    _subThermal = image_transport::create_subscription(this,
                                                       "thermal_image",
                                                       _onThermalDataReceive,
                                                       "raw",
                                                       profile);

    _subVisible  = image_transport::create_subscription(this,
                                                       "visible_image",
                                                       _onVisibleDataReceive,
                                                       "raw",
                                                       profile);

    _pubThermal = image_transport::create_camera_publisher(this, "thermal_image_view", profile);
    _pubVisible = image_transport::create_camera_publisher(this, "visible_image_view", profile);

    _sPalette = this->create_service<optris_drivers2::srv::Palette>("palette", std::bind(&OptrisColorconvert::onPalette, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    std::string camera_name;
    std::string camera_info_url;
    this->get_parameter("camera_name", camera_name);
    this->get_parameter("camera_info_url", camera_info_url);

    if (!_camera_info_manager.setCameraName(camera_name))
    {
      // GUID is 16 hex digits, which should be valid.
      // If not, use it for log messages anyway.
      RCLCPP_WARN(this->get_logger(), "[%s] name not valid for camera_info_manger", camera_name);
    }

    if (_camera_info_manager.validateURL(camera_info_url))
    {
      if ( !_camera_info_manager.loadCameraInfo(camera_info_url) )
      {
        RCLCPP_WARN(this->get_logger(), "camera_info_url does not contain calibration data." );
      } 
      else if ( !_camera_info_manager.isCalibrated() )
      {
        RCLCPP_WARN(this->get_logger(), "Camera is not calibrated. Using default values." );
      } 
    } 
    else
    {
      RCLCPP_ERROR_ONCE(this->get_logger(), "Calibration URL syntax is not supported by CameraInfoManager." );
    }
  }

  OptrisColorconvert::~OptrisColorconvert()
  {
    if(_bufferThermal)	delete [] _bufferThermal;
    if(_resizedBufferThermal)  delete [] _resizedBufferThermal;
    if(_bufferVisible)  delete [] _bufferVisible;
  }

  void OptrisColorconvert::onThermalDataReceive(const sensor_msgs::msg::Image::ConstSharedPtr & image)
  {
    //RCLCPP_INFO(get_logger(), "Received new thermal image");

    // check for any subscribers to save computation time
    //if(_pubThermal.getNumSubscribers() == 0)
    //   return;

    unsigned short* data = (unsigned short*)&image->data[0];
    _iBuilder.setData(image->width, image->height, data);

    if(_bufferThermal==NULL)
      _bufferThermal = new unsigned char[image->width * image->height * 3];

    _iBuilder.convertTemperatureToPaletteImage(_bufferThermal, true);

    //get the max and min temperature
    int radius = 3;
    if(image->width <9 || image->height<9) radius = 2;
    evo::ExtremalRegion minRegion;
    evo::ExtremalRegion maxRegion;
    _iBuilder.getMinMaxRegion(radius, &minRegion, &maxRegion);
    //RCLCPP_INFO(get_logger(), "Max temp: %f degree, Min temp: %f degree", maxRegion.t, minRegion.t);
    
    // 假設 _bufferThermal 是您已經有的包含圖像數據的buffer
    // 並且您已經有了 maxRegion.t 和 minRegion.t 這兩個溫度值

    // 首先，將_bufferThermal轉換成OpenCV的Mat格式以便操作
    cv::Mat thermalImage = cv::Mat(image->height, image->width, CV_8UC3, _bufferThermal);
    cv::Mat resizedImage; 

    cv::resize(thermalImage, resizedImage, cv::Size(scaled_image_width, scaled_image_height));

    float scaleX = scaled_image_width / static_cast<float>(thermalImage.cols);
    float scaleY = scaled_image_height / static_cast<float>(thermalImage.rows);

    // 准备进行k-means聚类的数据
    cv::Mat data_kmeans;
    thermalImage.convertTo(data_kmeans, CV_32F);
    data_kmeans = data_kmeans.reshape(1, data_kmeans.total());

    // 执行k-means聚类
    int K = 2; // 假设我们想要将图像分成两个聚类
    cv::Mat labels, centers;
    cv::kmeans(data_kmeans, K, labels, 
               cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<cv::Mat> masks(K);
    for (int k = 0; k < K; k++) {
        masks[k] = cv::Mat::zeros(thermalImage.size(), CV_8UC1); // 使用原始影像大小創建遮罩
    }

    for (int i = 0; i < labels.rows; i++) {
        int clusterIdx = labels.at<int>(i);
        int x = i % thermalImage.cols;
        int y = i / thermalImage.cols;
        masks[clusterIdx].at<uchar>(y, x) = 255; // 在對應的遮罩上標記聚類點
    }

     // I want to declare a center for each cluster
     std::vector<cv::Point2f> scaledCenters(K);
     std::vector<float> temperatures(K);

    for (int k = 0; k < K; k++) {
        int minX = thermalImage.cols, minY = thermalImage.rows;
        int maxX = 0, maxY = 0;

        for (int i = 0; i < labels.rows; i++) {
            int clusterIdx = labels.at<int>(i);
            if (clusterIdx == k) {
                int x = i % thermalImage.cols;
                int y = i / thermalImage.cols;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        // 計算矩形中心座標
        int centerX = (minX + maxX) / 2;
        int centerY = (minY + maxY) / 2;
        temperatures[k] = _iBuilder.getTemperatureAt(centerX, centerY); // get temperature at the center
        // get temperature at the center

        // 轉換中心座標到縮放後的影像尺寸
        scaledCenters[k] = cv::Point2f(centerX * scaleX, centerY * scaleY);
    }

    // 假設您有一個函數getTemperatureAtCoordinate
    
    

    for (int k = 0; k < K; k++) {
          cv::Moments combinedMoments; // 用於存儲所有輪廓矩的組合
          std::vector<std::vector<cv::Point>> contours;
          cv::findContours(masks[k], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

          
          
          // 繪製每個輪廓到縮放後的影像上
          for (const auto& contour : contours) {
              std::vector<cv::Point> resizedContour;
              for (size_t i = 0; i < contour.size(); i++) {
                  resizedContour.push_back(cv::Point(static_cast<int>(contour[i].x * scaleX), static_cast<int>(contour[i].y * scaleY)));
              }
              cv::polylines(resizedImage, std::vector<std::vector<cv::Point>>{resizedContour}, true, cv::Scalar(0, 255, 0), 2);
          }
          // 在縮放後的影像上標示總重心
          cv::circle(resizedImage, scaledCenters[k], 5, cv::Scalar(255, 0, 0), -1); // 使用紅色標示重心
          std::string tempText = "Temp: " + std::to_string(temperatures[k]) + " C";
          cv::putText(resizedImage, tempText, cv::Point(scaledCenters[k].x, scaledCenters[k].y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    // 將修改後的圖像 data 回寫到_bufferThermal
    if(_resizedBufferThermal==NULL)
      _resizedBufferThermal = new unsigned char[scaled_image_width * scaled_image_height * 3];

    memcpy(_resizedBufferThermal, resizedImage.data, scaled_image_width * scaled_image_height*3*sizeof(unsigned char));

    sensor_msgs::msg::Image img;
    img.header.frame_id = "thermal_image_view";
    img.height 	        = scaled_image_height;
    img.width 	        = scaled_image_width;
    img.encoding        = "rgb8";
    img.step            = scaled_image_width*3;
    img.header.stamp    = this->now();

    // copy the image buffer
    img.data.resize(img.height*img.step);
    memcpy(&img.data[0], &_resizedBufferThermal[0], img.height * img.step * sizeof(*_resizedBufferThermal));
    
    sensor_msgs::msg::CameraInfo camera_info = _camera_info_manager.getCameraInfo();
    camera_info.header = img.header;
    _pubThermal.publish(img, camera_info);
    
  }

  void OptrisColorconvert::onVisibleDataReceive(const sensor_msgs::msg::Image::ConstSharedPtr & image)
  {
    // check for any subscribers to save computation time
    //if(_pubVisible.getNumSubscribers() == 0)
    //   return;

    if(_bufferVisible==NULL)
      _bufferVisible = new unsigned char[image->width * image->height * 3];

    const unsigned char* data = &image->data[0];
    _iBuilder.yuv422torgb24(data, _bufferVisible, image->width, image->height);

    sensor_msgs::msg::Image img;
    img.header.frame_id = "visible_image_view";
    img.height          = image->height;
    img.width           = image->width;
    img.encoding        = "rgb8";
    img.step            = image->width*3;
    img.data.resize(img.height*img.step);

    img.header.stamp    = this->now();

    for(unsigned int i=0; i<image->width*image->height*3; i++) {
      img.data[i] = _bufferVisible[i];
    }

    sensor_msgs::msg::CameraInfo camera_info = _camera_info_manager.getCameraInfo();
    camera_info.header = img.header;
    _pubVisible.publish(img, camera_info);
  }

  void OptrisColorconvert::onPalette(const std::shared_ptr<rmw_request_id_t> request_header,
                                     const std::shared_ptr<optris_drivers2::srv::Palette::Request> req,
                                     const std::shared_ptr<optris_drivers2::srv::Palette::Response> res)
  {
    (void) request_header;

    res->success = false;

    if(req->palette > 0 && req->palette < 12)
    {
      _iBuilder.setPalette((evo::EnumOptrisColoringPalette)req->palette);
      res->success = true;
    }

    if(req->palette_scaling >=1 && req->palette_scaling <= 4)
    {
      _iBuilder.setPaletteScalingMethod((evo::EnumOptrisPaletteScalingMethod) req->palette_scaling);
      res->success = true;
    }

    if(_iBuilder.getPaletteScalingMethod() == evo::eManual &&  req->temperature_min < req->temperature_max)
    {
      _iBuilder.setManualTemperatureRange(req->temperature_min, req->temperature_max);
      res->success = true;
    }
  }

} //namespace
