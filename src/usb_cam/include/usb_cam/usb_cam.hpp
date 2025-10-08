// Copyright 2014 Robert Bosch, LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Robert Bosch, LLC nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#ifndef USB_CAM__USB_CAM_HPP_
#define USB_CAM__USB_CAM_HPP_

/**
 * @file usb_cam.hpp
 * @brief Defines the core UsbCam class and related data structures for V4L2 camera interaction.
 *
 * This header contains the main components for interfacing with a V4L2 device,
 * including the UsbCam class which encapsulates camera operations, and various
 * structs for managing parameters, image data, and supported formats.
 */

extern "C" {
#include <libavcodec/avcodec.h>
#include <linux/videodev2.h>
}

#include <chrono>
#include <memory>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include "usb_cam/utils.hpp"
#include "usb_cam/formats/pixel_format_base.hpp"
#include "usb_cam/formats/av_pixel_format_helper.hpp"

#include "usb_cam/formats/mjpeg.hpp"
#include "usb_cam/formats/mono.hpp"
#include "usb_cam/formats/rgb.hpp"
#include "usb_cam/formats/uyvy.hpp"
#include "usb_cam/formats/yuyv.hpp"
#include "usb_cam/formats/m420.hpp"


namespace usb_cam
{

using usb_cam::utils::io_method_t;
using usb_cam::formats::pixel_format_base;

/// @brief Add more formats here and to driver_supported_formats below as
/// they are added to this library
using usb_cam::formats::RGB8;
using usb_cam::formats::YUYV;
using usb_cam::formats::YUYV2RGB;
using usb_cam::formats::UYVY;
using usb_cam::formats::UYVY2RGB;
using usb_cam::formats::MONO8;
using usb_cam::formats::MONO16;
using usb_cam::formats::Y102MONO8;
using usb_cam::formats::RAW_MJPEG;
using usb_cam::formats::MJPEG2RGB;
using usb_cam::formats::M4202RGB;


/**
 * @brief Provides a list of all pixel format conversions supported by this driver.
 * @param args Arguments needed to construct the pixel format objects.
 * @return A vector of shared pointers to the supported pixel format handlers.
 */
std::vector<std::shared_ptr<pixel_format_base>> driver_supported_formats(
  const formats::format_arguments_t & args = formats::format_arguments_t())
{
  std::vector<std::shared_ptr<pixel_format_base>> fmts = {
    std::make_shared<RGB8>(args),
    std::make_shared<YUYV>(args),
    std::make_shared<YUYV2RGB>(args),
    std::make_shared<UYVY>(args),
    std::make_shared<UYVY2RGB>(args),
    std::make_shared<MONO8>(args),
    std::make_shared<MONO16>(args),
    std::make_shared<Y102MONO8>(args),
    std::make_shared<RAW_MJPEG>(args),
    std::make_shared<MJPEG2RGB>(args),
    std::make_shared<M4202RGB>(args),
  };
  return fmts;
}

/**
 * @struct capture_format_t
 * @brief Holds information about a specific capture format supported by the device.
 */
typedef struct capture_format_t
{
  struct v4l2_fmtdesc format; ///< V4L2 format description.
  struct v4l2_frmivalenum v4l2_fmt; ///< V4L2 frame interval enumeration.
} capture_format_t;

/**
 * @struct parameters_t
 * @brief A comprehensive structure to hold all configurable parameters for the camera node.
 */
typedef struct parameters_t
{
  std::string camera_name;
  std::string device_name;
  std::string frame_id;
  std::string io_method_name;
  std::string camera_info_url;
  std::string pixel_format_name;
  std::string av_device_format;
  int image_width;
  int image_height;
  int framerate;
  int brightness;
  int contrast;
  int saturation;
  int sharpness;
  int gain;
  int white_balance;
  int exposure;
  int focus;
  bool auto_white_balance;
  bool autoexposure;
  bool autofocus;

  parameters_t()
// *INDENT-OFF*
    : camera_name("usb_cam"),
    device_name("/dev/video0"),
    frame_id("camera"),
    io_method_name("mmap"),
    camera_info_url("package://usb_cam/config/camera_info.yaml"),
    pixel_format_name("yuyv2rgb"),
    av_device_format("YUV422P"),
    image_width(600),
    image_height(480),
    framerate(30.0),
    brightness(-1),
    contrast(-1),
    saturation(-1),
    sharpness(-1),
    gain(-1),
    white_balance(-1),
    exposure(-1),
    focus(-1),
    auto_white_balance(true),
    autoexposure(true),
    autofocus(false)
  {
  }
// *INDENT-ON*
} parameters_t;

/**
 * @struct image_t
 * @brief Represents an image buffer and its associated metadata.
 */
typedef struct image_t
{
  char * data;
  size_t width;
  size_t height;
  std::shared_ptr<pixel_format_base> pixel_format;
  size_t number_of_pixels;
  size_t bytes_per_line;
  size_t size_in_bytes;
  v4l2_format v4l2_fmt;
  struct timespec stamp;

  size_t set_number_of_pixels()
  {
    number_of_pixels = width * height;
    return number_of_pixels;
  }
  size_t set_bytes_per_line()
  {
    bytes_per_line = width * pixel_format->byte_depth() * pixel_format->channels();
    return bytes_per_line;
  }
  size_t set_size_in_bytes()
  {
    size_in_bytes = height * bytes_per_line;
    return size_in_bytes;
  }

  /// @brief make it a shorter API call to get the pixel format
  unsigned int get_format_fourcc()
  {
    return pixel_format->v4l2();
  }
} image_t;

/**
 * @class UsbCam
 * @brief Main class for handling V4L2 camera operations.
 *
 * This class provides a high-level interface to open, configure,
 * start/stop streaming, and capture frames from a V4L2 compatible device.
 */
class UsbCam
{
public:
  /**
   * @brief Construct a new UsbCam object.
   */
  UsbCam();
  /**
   * @brief Destroy the UsbCam object, ensuring the device is shut down.
   */
  ~UsbCam();

  /**
   * @brief Configure the camera device with a set of parameters.
   * This must be called before starting the stream.
   * @param parameters A struct containing all configuration settings.
   * @param io_method The I/O method to use (mmap, read, userptr).
   */
  void configure(parameters_t & parameters, const io_method_t & io_method);

  /**
   * @brief Start the camera stream.
   */
  void start();

  /**
   * @brief Stop the camera stream and release the device.
   */
  void shutdown(void);

  /**
   * @brief Grab a new image from the device and return a pointer to the data.
   * The user is responsible for managing the returned buffer.
   * @return char* A pointer to the raw image data buffer.
   */
  char * get_image();

  /**
   * @brief Grab a new image and fill a user-provided buffer.
   * @param destination A pointer to the destination buffer to fill with image data.
   */
  void get_image(char * destination);

  /**
   * @brief Get the list of formats supported by the camera device.
   * @return std::vector<capture_format_t> A vector of supported capture formats.
   */
  std::vector<capture_format_t> get_supported_formats();

  /**
   * @brief Enable or disable the camera's auto focus feature.
   * @param value 1 to enable, 0 to disable.
   * @return true if the operation was successful, false otherwise.
   */
  bool set_auto_focus(int value);

  /**
   * @brief Set a V4L2 device parameter using an integer value.
   * @param param The name of the parameter to set.
   * @param value The integer value to set.
   * @return true if the operation was successful, false otherwise.
   */
  bool set_v4l_parameter(const std::string & param, int value);

  /**
   * @brief Set a V4L2 device parameter using a string value.
   * @param param The name of the parameter to set.
   * @param value The string value to set.
   * @return true if the operation was successful, false otherwise.
   */
  bool set_v4l_parameter(const std::string & param, const std::string & value);

  /**
   * @brief Stop the video capture stream.
   */
  void stop_capturing();

  /**
   * @brief Start the video capture stream.
   */
  void start_capturing();

  /**
   * @brief Get the width of the captured image.
   * @return size_t Image width in pixels.
   */
  inline size_t get_image_width()
  {
    return m_image.width;
  }

  /**
   * @brief Get the height of the captured image.
   * @return size_t Image height in pixels.
   */
  inline size_t get_image_height()
  {
    return m_image.height;
  }

  /**
   * @brief Get the total size of the image buffer in bytes.
   * @return size_t The size of the image in bytes.
   */
  inline size_t get_image_size_in_bytes()
  {
    return m_image.size_in_bytes;
  }

  /**
   * @brief Get the total number of pixels in the image.
   * @return size_t The number of pixels (width * height).
   */
  inline size_t get_image_size_in_pixels()
  {
    return m_image.number_of_pixels;
  }

  /**
   * @brief Get the timestamp of the last captured image.
   * @return timespec The timestamp of the image.
   */
  inline timespec get_image_timestamp()
  {
    return m_image.stamp;
  }

  /**
   * @brief Get the number of bytes per line (stride) of the image.
   * @return unsigned int The number of bytes per line.
   */
  inline unsigned int get_image_step()
  {
    return m_image.bytes_per_line;
  }

  /**
   * @brief Get the name of the camera device file.
   * @return std::string The device name (e.g., "/dev/video0").
   */
  inline std::string get_device_name()
  {
    return m_device_name;
  }

  /**
   * @brief Get the currently configured pixel format handler.
   * @return std::shared_ptr<pixel_format_base> A pointer to the pixel format handler.
   */
  inline std::shared_ptr<pixel_format_base> get_pixel_format()
  {
    return m_image.pixel_format;
  }

  /**
   * @brief Get the currently configured I/O method.
   * @return usb_cam::utils::io_method_t The I/O method.
   */
  inline usb_cam::utils::io_method_t get_io_method()
  {
    return m_io;
  }

  /**
   * @brief Get the file descriptor for the camera device.
   * @return int The file descriptor.
   */
  inline int get_fd()
  {
    return m_fd;
  }

  /**
   * @brief Get a pointer to the array of V4L2 buffers.
   * @return std::shared_ptr<usb_cam::utils::buffer[]> A pointer to the buffers.
   */
  inline std::shared_ptr<usb_cam::utils::buffer[]> get_buffers()
  {
    return m_buffers;
  }

  /**
   * @brief Get the number of allocated V4L2 buffers.
   * @return unsigned int The number of buffers.
   */
  inline unsigned int number_of_buffers()
  {
    return m_number_of_buffers;
  }

  /**
   * @brief Get the AVCodec used for decompression (if any).
   * @return AVCodec* A pointer to the AVCodec.
   */
  inline AVCodec * get_avcodec()
  {
    return m_avcodec;
  }

  /**
   * @brief Get the AVDictionary of options for the codec.
   * @return AVDictionary* A pointer to the AVDictionary.
   */
  inline AVDictionary * get_avoptions()
  {
    return m_avoptions;
  }

  /**
   * @brief Get the AVCodecContext for the decompression stream.
   * @return AVCodecContext* A pointer to the AVCodecContext.
   */
  inline AVCodecContext * get_avcodec_context()
  {
    return m_avcodec_context;
  }

  /**
   * @brief Get the AVFrame used for holding decompressed video data.
   * @return AVFrame* A pointer to the AVFrame.
   */
  inline AVFrame * get_avframe()
  {
    return m_avframe;
  }

  /**
   * @brief Check if the camera is currently capturing.
   * @return true if capturing, false otherwise.
   */
  inline bool is_capturing()
  {
    return m_is_capturing;
  }

  /**
   * @brief Get the time shift between the monotonic clock and epoch time.
   * @return time_t The time shift in microseconds.
   */
  inline time_t get_epoch_time_shift_us()
  {
    return m_epoch_time_shift_us;
  }

  /**
   * @brief Get the cached list of supported formats. If not cached, it queries the device.
   * @return std::vector<capture_format_t> A vector of supported formats.
   */
  inline std::vector<capture_format_t> supported_formats()
  {
    if (m_supported_formats.size() == 0) {
      this->get_supported_formats();
    }

    return m_supported_formats;
  }

  /**
   * @brief Check if the given format is supported by this device and set it if it is.
   * @param args A struct containing the format name and other parameters.
   * @return bool true if the format is supported and set, false otherwise.
   * @throws std::invalid_argument if the format is not supported by the driver.
   */
  inline bool set_pixel_format(const formats::format_arguments_t & args)
  {
    bool result = false;

    std::shared_ptr<pixel_format_base> found_driver_format = nullptr;

    // First check if given format is supported by this driver
    for (auto driver_fmt : driver_supported_formats(args)) {
      if (driver_fmt->name() == args.name) {
        found_driver_format = driver_fmt;
      }
    }

    if (found_driver_format == nullptr) {
      // List the supported formats of this driver for the user before throwing
      std::cerr << "This driver supports the following formats:" << std::endl;
      for (auto driver_fmt : driver_supported_formats(args)) {
        std::cerr << "\t" << driver_fmt->name() << std::endl;
      }
      throw std::invalid_argument(
              "Specified format `" + args.name + "` is unsupported by this ROS driver"
      );
    }

    std::cout << "This device supports the following formats:" << std::endl;
    for (auto fmt : this->supported_formats()) {
      // Always list the devices supported formats for the user
      std::cout << "\t" << fmt.format.description << " ";
      std::cout << fmt.v4l2_fmt.width << " x " << fmt.v4l2_fmt.height << " (";
      std::cout << fmt.v4l2_fmt.discrete.denominator / fmt.v4l2_fmt.discrete.numerator << " Hz)";
      std::cout << std::endl;

      if (fmt.v4l2_fmt.pixel_format == found_driver_format->v4l2()) {
        result = true;
        m_image.pixel_format = found_driver_format;
      }
    }

    return result;
  }

  /**
   * @brief Set the pixel format from a `parameters_t` struct.
   *
   * This is a convenience function that constructs the necessary arguments
   * and calls the main `set_pixel_format` method.
   *
   * @param parameters The struct containing all camera parameters.
   * @return std::shared_ptr<pixel_format_base> A pointer to the configured pixel format handler.
   * @throws std::invalid_argument if the specified format is not supported by the device.
   */
  inline std::shared_ptr<pixel_format_base> set_pixel_format(const parameters_t & parameters)
  {
    // create format arguments structure
    formats::format_arguments_t format_args({
        parameters.pixel_format_name,
        parameters.image_width,
        parameters.image_height,
        m_image.number_of_pixels,
        parameters.av_device_format,
      });

    // Look for specified pixel format
    if (!this->set_pixel_format(format_args)) {
      throw std::invalid_argument(
              "Specified format `" + parameters.pixel_format_name + "` is unsupported by the " +
              "selected device `" + parameters.device_name + "`"
      );
    }

    return m_image.pixel_format;
  }

private:
  void init_read();
  void init_mmap();
  void init_userp();
  void init_device();

  void open_device();
  void grab_image();
  void read_frame();
  void process_image(const char * src, char * & dest, const int & bytes_used);

  void uninit_device();
  void close_device();

  std::string m_device_name;
  usb_cam::utils::io_method_t m_io;
  int m_fd;
  unsigned int m_number_of_buffers;
  std::shared_ptr<usb_cam::utils::buffer[]> m_buffers;
  image_t m_image;

  AVFrame * m_avframe;
  AVCodec * m_avcodec;
  AVDictionary * m_avoptions;
  AVCodecContext * m_avcodec_context;

  bool m_is_capturing;
  int m_framerate;
  const time_t m_epoch_time_shift_us;
  std::vector<capture_format_t> m_supported_formats;
};

}  // namespace usb_cam

#endif  // USB_CAM__USB_CAM_HPP_