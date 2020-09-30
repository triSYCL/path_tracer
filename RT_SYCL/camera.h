#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

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

  point3 origin;
  point3 lower_left_corner =
    origin - horizontal/2 - vertical/2 - vec3 { 0, 0, focal_length };
  vec3 horizontal { viewport_width, 0.0, 0.0 };
  vec3 vertical { 0.0, viewport_height, 0.0 };

};
#endif
