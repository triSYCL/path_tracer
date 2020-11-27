#ifndef RT_SYCL_CAMERA_H
#define RT_SYCL_CAMERA_H

#include "rtweekend.hpp"

class camera {

public:

  static auto constexpr aspect_ratio = 16.0/9.0;
  static auto constexpr viewport_height = 2.0;
  static auto constexpr viewport_width = aspect_ratio*viewport_height;
  static auto constexpr focal_length = 1.0;

  ray get_ray(double u, double v) const {
    return
      { origin, lower_left_corner + u * horizontal + v * vertical - origin };
  }

private:

  point origin;
  point lower_left_corner =
    origin - horizontal/2 - vertical/2 - vec { 0, 0, focal_length };
  vec horizontal { viewport_width, 0.0, 0.0 };
  vec vertical { 0.0, viewport_height, 0.0 };

};
#endif
