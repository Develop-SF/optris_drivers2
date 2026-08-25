#include "OptrisImager.h"

#include <sys/stat.h>
#include <chrono>
#include <stdexcept>

namespace optris_drivers2
{

OptrisImager::OptrisImager() : Node("optris_imager")
{
  RCLCPP_INFO(get_logger(), "Initializing Optris Imager Node...");

  // Declare and get the XML config file parameter
  this->declare_parameter<std::string>("xml_config_file", "");
  std::string xmlConfig = this->get_parameter("xml_config_file").as_string();
  
  if(xmlConfig.empty())
  {
    RCLCPP_FATAL(get_logger(), "xml_config_file parameter is required but not set");
    throw std::runtime_error("xml_config_file parameter missing");
  }

  // Verify XML config file exists
  struct stat s;
  if(stat(xmlConfig.c_str(), &s) != 0)
  {
    RCLCPP_FATAL(get_logger(), "XML config file does not exist: %s", xmlConfig.c_str());
    throw std::runtime_error("XML config file not found: " + xmlConfig);
  }
  
  RCLCPP_INFO(get_logger(), "Using configuration file: %s", xmlConfig.c_str());

  // Read parameters from xml file
  evo::IRDeviceParams params;
  if(!evo::IRDeviceParamsReader::readXML(xmlConfig.c_str(), params))
  {
    RCLCPP_FATAL(get_logger(), "Failed to read XML configuration");
    throw std::runtime_error("Failed to read XML configuration");
  }

  // Find valid device
  evo::IRDevice* dev = evo::IRDevice::IRCreateDevice(params);
  if(!dev)
  {
    RCLCPP_FATAL(get_logger(), "UVC device with serial %ld could not be found", params.serial);
    throw std::runtime_error("Could not create IR device");
  }

  // Store device pointer for cleanup
  _dev = dev;

  RCLCPP_INFO(get_logger(), "Initializing Optris device...");

  // Automatically execute device initialization steps
  RCLCPP_INFO(get_logger(), "Executing device calibration initialization...");
  int calibration_result = system("sudo ir_download_calibration > /dev/null 2>&1");
  if (calibration_result == 0) {
    RCLCPP_INFO(get_logger(), "Calibration data initialization completed");
  } else {
    RCLCPP_WARN(get_logger(), "Calibration data initialization failed, but continuing execution");
  }

  // Execute device serial number lookup (this may trigger device re-initialization)
  RCLCPP_INFO(get_logger(), "Finding and initializing device serial number...");
  int serial_result = system("sudo ir_find_serial > /dev/null 2>&1");
  if (serial_result == 0) {
    RCLCPP_INFO(get_logger(), "Device serial number initialization completed");
  } else {
    RCLCPP_WARN(get_logger(), "Device serial number initialization failed, but continuing execution");
  }

  RCLCPP_INFO(get_logger(), "Serial: %ld", params.serial);

  _imager.init(&params, dev->getFrequency(), dev->getWidth(), dev->getHeight(), dev->controlledViaHID());

  // Calibrated temperature range to start in. Must be one of the ranges in the
  // camera's calibration file (Xi80: -20..100, 0..250, 150..900); can still be
  // switched at runtime through the temperature_range service.
  this->declare_parameter<int>("temperature_range_min", 0);
  this->declare_parameter<int>("temperature_range_max", 250);
  int tRangeMin = this->get_parameter("temperature_range_min").as_int();
  int tRangeMax = this->get_parameter("temperature_range_max").as_int();
  if(_imager.setTempRange(tRangeMin, tRangeMax))
  {
    RCLCPP_INFO(get_logger(), "Temperature range: %d..%d C", tRangeMin, tRangeMax);
  }
  else
  {
    RCLCPP_WARN(get_logger(), "Temperature range %d..%d C is not calibrated for this camera, falling back to 0..250 C", tRangeMin, tRangeMax);
    _imager.setTempRange(0, 250);
  }

  // Radiometric correction (IRImager::setRadiationParameters). Defaults keep
  // the SDK behaviour: emissivity 1.0, transmissivity 1.0, ambient measured by
  // the camera. Set emissivity to the observed object's value (e.g. ~0.9 for
  // an oxidised steel wok) for correct absolute temperatures. Live-tunable:
  //   ros2 param set <ns>/optris_imager emissivity 0.9
  this->declare_parameter<double>("emissivity", 1.0);
  this->declare_parameter<double>("transmissivity", 1.0);
  this->declare_parameter<double>("ambient_temperature", -999.0);
  applyRadiationParameters(this->get_parameter("emissivity").as_double(),
                           this->get_parameter("transmissivity").as_double(),
                           this->get_parameter("ambient_temperature").as_double());
  _param_cb = this->add_on_set_parameters_callback(
      std::bind(&OptrisImager::onParametersSet, this, std::placeholders::_1));

  float focusPos = _imager.getFocusmotorPos();
  if(focusPos < 0.f)
    RCLCPP_INFO(get_logger(), "Focus motor: not available");
  else
    RCLCPP_INFO(get_logger(), "Focus motor position: %.1f %%", focusPos);

  _imager.setClient(this);

  _bufferRaw = new unsigned char[dev->getRawBufferSize()];

  auto qos = rclcpp::QoS(
      rclcpp::QoSInitialization(
        // The history policy determines how messages are saved until taken by
        // the reader.
        // KEEP_ALL saves all messages until they are taken.
        // KEEP_LAST enforces a limit on the number of messages that are saved,
        // specified by the "depth" parameter.
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        // The next parameter represents how many messages to store in history when the
        // history policy is KEEP_LAST.
        1
    ));

  // The reliability policy can be reliable, meaning that the underlying transport layer will try
  // ensure that every message gets received in order, or best effort, meaning that the transport
  // makes no guarantees about the order or reliability of delivery.
  // Options are: SYSTEM_DEFAULT, RELIABLE, BEST_EFFORT and UNKNOWN
  rmw_qos_reliability_policy_t reliability_policy = RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT;
  qos.reliability(reliability_policy);

  _thermal_pub = this->create_publisher<sensor_msgs::msg::Image>("thermal_image", qos);
  _thermal_image.header.frame_id = "thermal_image";
  _thermal_image.height          = _imager.getHeight();
  _thermal_image.width           = _imager.getWidth();
  _thermal_image.encoding        = "mono16";
  _thermal_image.step            = _thermal_image.width * 2;
  _thermal_image.data.resize(_thermal_image.height * _thermal_image.step);

  if(_imager.hasBispectralTechnology())
  {
    _visible_pub = this->create_publisher<sensor_msgs::msg::Image>("visible_image", qos);
    _visible_image.header.frame_id = "visible_image";
    _visible_image.height          = _imager.getVisibleHeight();
    _visible_image.width           = _imager.getVisibleWidth();
    _visible_image.encoding        = "yuv422";
    _visible_image.step            = _visible_image.width * 2;
    _visible_image.data.resize(_visible_image.height * _visible_image.step);
  }

  // advertise the camera internal timer
  _timer_pub = this->create_publisher<sensor_msgs::msg::TimeReference>("optris_timer", qos);

  // advertise the internal temperature measurements
  _temp_pub = this->create_publisher<optris_drivers2::msg::Temperature> ("internal_temperature", qos);

  // advertise the flag state
  _flag_pub = this->create_publisher<optris_drivers2::msg::Flag> ("flag_state", qos);

  // provide AutoFlag service
  _sAuto  = this->create_service<optris_drivers2::srv::AutoFlag> ("auto_flag", std::bind(&OptrisImager::onAutoFlag, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // provide service to force a flag cycle manually
  _sForce = this->create_service<std_srvs::srv::Empty> ("force_flag", std::bind(&OptrisImager::onForceFlag, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // provide service to change temperature range
  _sTemp  = this->create_service<optris_drivers2::srv::TemperatureRange> ("temperature_range", std::bind(&OptrisImager::onSetTemperatureRange, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // provide service to change focus
  _sFocus  = this->create_service<optris_drivers2::srv::FocusMotorPos> ("focus_motor_pos", std::bind(&OptrisImager::onFocus, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  _img_cnt = 0;

  _dev = dev;

  _dev->startStreaming();

  // create_wall_timer changes the behaviour of spin. Services will stop working
  _run = true;
  _th = new std::thread(&OptrisImager::timer_callback, this);

  return;
}

OptrisImager::~OptrisImager()
{
  RCLCPP_INFO(get_logger(), "Shutting down OptrisImager...");
  
  // 停止線程運行
  _run = false;
  
  // 等待線程結束，設置超時以避免死鎖
  if(_th && _th->joinable()) {
    try {
      _th->join();
    } catch(const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Thread join exception: %s", e.what());
    }
    delete _th;
    _th = nullptr;
  }
  
  // 停止設備串流
  if(_dev) {
    try {
      _dev->stopStreaming();
    } catch(const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Device stop streaming exception: %s", e.what());
    }
  }

  // 清理緩衝區
  if(_bufferRaw) {
    delete [] _bufferRaw;
    _bufferRaw = nullptr;
  }
  
  RCLCPP_INFO(get_logger(), "OptrisImager shutdown complete");
}

void OptrisImager::timer_callback()
{
  auto durInSec = std::chrono::duration<double>(1.0/_imager.getMaxFramerate());
  RCLCPP_INFO(get_logger(), "Sampling rate: %lf sec", 1.0/_imager.getMaxFramerate());
  while(_run)
  {
    int retval = _dev->getFrame(_bufferRaw);
    if(retval==evo::IRIMAGER_SUCCESS)
    {
      _imager.process(_bufferRaw);
    }
    if(retval==evo::IRIMAGER_DISCONNECTED)
    {
      rclcpp::shutdown();
    }
    std::this_thread::sleep_for(durInSec);
  }
}

void OptrisImager::onThermalFrame(unsigned short* image, unsigned int w, unsigned int h, evo::IRFrameMetadata meta, void* arg)
{
  (void) arg;

  //RCLCPP_INFO(get_logger(), "onThermalFrame");
  memcpy(&_thermal_image.data[0], image, w * h * sizeof(*image));

  _thermal_image.header.frame_id = "";
  _thermal_image.header.stamp = rclcpp::Node::now() - rclcpp::Duration::from_seconds(0.05);
  _thermal_pub->publish(_thermal_image);

  _device_timer.header.frame_id=_thermal_image.header.frame_id;
  _device_timer.header.stamp = _thermal_image.header.stamp;
  //TODO: Check validity of timestamp
  _device_timer.time_ref.sec = (int32_t) (meta.timestamp / 10000000);
  _device_timer.time_ref.nanosec = (int32_t) (meta.timestamp % 10000000);
  _timer_pub->publish(_device_timer);

  _internal_temperature.header.stamp     = _thermal_image.header.stamp;
  _internal_temperature.header.frame_id  = _thermal_image.header.frame_id;
  _internal_temperature.temperature_flag = _imager.getTempFlag();
  _internal_temperature.temperature_box  = _imager.getTempBox();
  _internal_temperature.temperature_chip = _imager.getTempChip(); 
  _temp_pub->publish(_internal_temperature);
  
}

void OptrisImager::onVisibleFrame(unsigned char* image, unsigned int w, unsigned int h, evo::IRFrameMetadata meta, void* arg)
{
  (void) arg;
  (void) meta;

  if(_visible_pub->get_subscription_count()==0) return;

  memcpy(&_visible_image.data[0], image, 2 * w * h * sizeof(*image));

  _visible_image.header.frame_id  = _img_cnt;
  _visible_image.header.stamp = rclcpp::Node::now();
  _visible_pub->publish(_visible_image);
}

void OptrisImager::onFlagStateChange(evo::EnumFlagState flagstate, void* arg)
{
  (void) arg;

  optris_drivers2::msg::Flag flag;
  flag.flag_state      = flagstate;
  flag.header.frame_id = _thermal_image.header.frame_id;
  flag.header.stamp    = _thermal_image.header.stamp;
  _flag_pub->publish(flag);
}

void OptrisImager::onProcessExit(void* arg)
{
  (void) arg;
}

void OptrisImager::onAutoFlag(const std::shared_ptr<rmw_request_id_t> request_header,
                              const std::shared_ptr<optris_drivers2::srv::AutoFlag::Request> req,
                              const std::shared_ptr<optris_drivers2::srv::AutoFlag::Response> res)
{
  (void) request_header;
  RCLCPP_INFO(get_logger(), "Calling service on_auto_flag");
  _imager.setAutoFlag(req->auto_flag);
  res->is_auto_flag_active = _imager.getAutoFlag();
}

void OptrisImager::onForceFlag(const std::shared_ptr<rmw_request_id_t> request_header,
                               const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                               const std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
  (void) request_header;
  (void) req;
  (void) res;
  _imager.forceFlagEvent();
}

void OptrisImager::onSetTemperatureRange(const std::shared_ptr<rmw_request_id_t> request_header,
                                         const std::shared_ptr<optris_drivers2::srv::TemperatureRange::Request> req,
                                         const std::shared_ptr<optris_drivers2::srv::TemperatureRange::Response> res)
{
  (void) request_header;

  // Route through the parameter interface so the active range is always
  // observable via `ros2 param get <node> temperature_range_min/max`
  // (validation + flag cycle happen in onParametersSet).
  auto result = this->set_parameters_atomically({
      rclcpp::Parameter("temperature_range_min", (int)req->temperature_range_min),
      rclcpp::Parameter("temperature_range_max", (int)req->temperature_range_max)});
  if(!result.successful)
    RCLCPP_WARN(get_logger(), "temperature_range service rejected: %s", result.reason.c_str());
  res->success = result.successful;
}


void OptrisImager::onFocus(const std::shared_ptr<rmw_request_id_t> request_header,
                             const std::shared_ptr<optris_drivers2::srv::FocusMotorPos::Request> req,
                             const std::shared_ptr<optris_drivers2::srv::FocusMotorPos::Response> res)
{
    (void) request_header;
    RCLCPP_INFO(get_logger(), "Calling service on_Focus %f",req->pos);

    bool error = _imager.setFocusmotorPos(req->pos);
    //if error is false, it means that the focus motor is not available
    res->success = error;
    if(error)
      RCLCPP_INFO(get_logger(), "Focus motor position now %.1f %%", _imager.getFocusmotorPos());
    else
      RCLCPP_WARN(get_logger(), "Focus motor not available");


}

void OptrisImager::applyRadiationParameters(double emissivity, double transmissivity, double tAmbient)
{
  _imager.setRadiationParameters((float)emissivity, (float)transmissivity, (float)tAmbient);
  if(tAmbient < -273.15)
    RCLCPP_INFO(get_logger(), "Radiation parameters: emissivity=%.3f transmissivity=%.3f ambient=camera-measured", emissivity, transmissivity);
  else
    RCLCPP_INFO(get_logger(), "Radiation parameters: emissivity=%.3f transmissivity=%.3f ambient=%.1f C", emissivity, transmissivity, tAmbient);
}

rcl_interfaces::msg::SetParametersResult OptrisImager::onParametersSet(const std::vector<rclcpp::Parameter>& params)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  // Start from the committed values, overlay the incoming ones (they are not
  // stored yet while this callback runs).
  double e  = this->get_parameter("emissivity").as_double();
  double t  = this->get_parameter("transmissivity").as_double();
  double ta = this->get_parameter("ambient_temperature").as_double();
  int tMin = this->get_parameter("temperature_range_min").as_int();
  int tMax = this->get_parameter("temperature_range_max").as_int();
  bool radiation = false;
  bool range = false;
  for(const auto& p : params)
  {
    const std::string& name = p.get_name();
    if(name == "temperature_range_min" || name == "temperature_range_max")
    {
      if(p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER)
      {
        result.successful = false;
        result.reason = name + " must be an integer (C)";
        return result;
      }
      (name == "temperature_range_min" ? tMin : tMax) = (int)p.as_int();
      range = true;
      continue;
    }
    if(name == "emissivity" || name == "transmissivity")
    {
      if(p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
      {
        result.successful = false;
        result.reason = name + " must be a double (e.g. 0.9)";
        return result;
      }
      double v = p.as_double();
      if(v <= 0.0 || v > 1.0)
      {
        result.successful = false;
        result.reason = name + " must be in (0, 1]";
        return result;
      }
      (name == "emissivity" ? e : t) = v;
      radiation = true;
    }
    else if(name == "ambient_temperature")
    {
      if(p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
      {
        result.successful = false;
        result.reason = "ambient_temperature must be a double (C); < -273.15 = camera-measured";
        return result;
      }
      ta = p.as_double();
      radiation = true;
    }
  }
  if(range)
  {
    // Only ranges present in the camera's calibration file are accepted.
    if(!_imager.setTempRange(tMin, tMax))
    {
      result.successful = false;
      result.reason = "temperature range " + std::to_string(tMin) + ".." + std::to_string(tMax) +
                      " C is not calibrated for this camera (Xi80: -20..100, 0..250, 150..900)";
      return result;
    }
    _imager.forceFlagEvent(1000.f);
    RCLCPP_INFO(get_logger(), "Temperature range switched to %d..%d C (flag cycle in 1 s)", tMin, tMax);
  }
  if(radiation)
    applyRadiationParameters(e, t, ta);
  return result;
}

}
