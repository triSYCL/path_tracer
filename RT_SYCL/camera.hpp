#ifndef RT_SYCL_CAMERA_H
#define RT_SYCL_CAMERA_H

#include "rtweekend.hpp"
#include "ray.hpp"

class camera {

public:
  camera(
    point look_from,
    point look_at,
    vec vup,
    real_t vfov,
    real_t aspect_ratio,
    real_t aperature,
    real_t focus_dist
  ){
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta/2);
    auto viewport_height = 2.0*h;
    auto viewport_width = aspect_ratio * viewport_height;

    w = unit_vector(look_from - look_at);
    u = unit_vector(sycl::cross(vup, w));
    v = sycl::cross(w, u);

    origin = look_from;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

    lens_radius = aperature / 2;
  }

  ray get_ray(real_t s, real_t t) const {
    vec rd = lens_radius * random_in_unit_disk();
    vec offset = u * rd.x() + v * rd.y();
    return ray(
      origin + offset, 
      lower_left_corner + s * horizontal + t * vertical - origin - offset);
  }

private:
  point origin;
  point lower_left_corner;
  vec horizontal;
  vec vertical;
  vec u,v,w;
  real_t lens_radius;
};

#endif
