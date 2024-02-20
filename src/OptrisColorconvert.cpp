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

    // 计算每个聚类的重心
    std::vector<cv::Point2f> centroids(K);
    std::vector<int> counts(K, 0);
    for (int i = 0; i < labels.rows; i++) {
        int clusterIdx = labels.at<int>(i);
        int index = i % thermalImage.cols + (i / thermalImage.cols) * thermalImage.cols;
        centroids[clusterIdx] += cv::Point2f(index % thermalImage.cols, index / thermalImage.cols);
        counts[clusterIdx]++;
    }
    float scaleX = scaled_image_width / static_cast<float>(thermalImage.cols);
    float scaleY = scaled_image_height / static_cast<float>(thermalImage.rows);

    for (int i = 0; i < K; i++) {
        centroids[i].x /= counts[i];
        centroids[i].y /= counts[i];

        // 调整重心位置到提升解析度后的图像坐标
        cv::Point2f resizedCentroid(centroids[i].x * scaleX, centroids[i].y * scaleY);

        // 获取重心位置的温度
        float temperature = _iBuilder.getTemperatureAt(static_cast<int>(centroids[i].x), static_cast<int>(centroids[i].y));
        std::string tempText = "Temp: " + std::to_string(temperature) + " C";

        // 在提升解析度后的图像上标注重心和温度
        cv::circle(resizedImage, resizedCentroid, 5, cv::Scalar(0, 255, 0), -1); // 使用绿色标注重心
        cv::putText(resizedImage, tempText, cv::Point(resizedCentroid.x, resizedCentroid.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // 設置文字參數
    //int fontFace = cv::FONT_HERSHEY_PLAIN;
    /*
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 1;
    cv::Scalar textColor(255, 255, 255); // 白色文字
    int baseline = 0;

    // 準備要顯示的文字
    std::string maxTempText = "Max: " + std::to_string(maxRegion.t) + " C";
    std::string minTempText = "Min: " + std::to_string(minRegion.t) + " C";

    // 獲取文字框大小
    cv::Size textSizeMax = cv::getTextSize(maxTempText, fontFace, fontScale, thickness, &baseline);
    cv::Size textSizeMin = cv::getTextSize(minTempText, fontFace, fontScale, thickness, &baseline);

    // 選擇文字位置
    //cv::Point maxTempPos(10, textSizeMax.height + 10); // 略高於圖像底部
    //cv::Point minTempPos(10, textSizeMax.height + textSizeMin.height + 20); // 在最高溫度資訊下方

    // 计算文字位置使其居中
    int totalTextHeight = textSizeMax.height + textSizeMin.height + 5; // 假设5像素的间距
    cv::Point maxTempPos((resizedImage.cols - textSizeMax.width) / 2, (resizedImage.rows - totalTextHeight) / 2 + textSizeMax.height);
    cv::Point minTempPos((resizedImage.cols - textSizeMin.width) / 2, maxTempPos.y + textSizeMin.height + 5);

    // 在圖像上繪製文字
    cv::putText(resizedImage, maxTempText, maxTempPos, fontFace, fontScale, textColor, thickness);
    cv::putText(resizedImage, minTempText, minTempPos, fontFace, fontScale, textColor, thickness);
    */
    

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
