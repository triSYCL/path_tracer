#ifndef RT_SYCL_CAMERA_H
#define RT_SYCL_CAMERA_H

#include <cmath>

#include "rtweekend.hpp"
#include "ray.hpp"

/** Camera model

    This implements:

    - https://raytracing.github.io/books/RayTracingInOneWeekend.html#positionablecamera

    - https://raytracing.github.io/books/RayTracingInOneWeekend.html#defocusblur
*/
class camera {

  /// Position of the camera
  point origin;

  /// Lower left corner of the camera plane
  point lower_left_corner;

  /// Horizontal vector of the camera plane
  vec horizontal;

  /// Vertical vector of the camera plane
  vec vertical;

  /// Right axis of the camera plane
  vec u;

  /// Vertical axis of the camera plane
  vec v;

  /// Camera viewing direction
  vec w;

  /// Size of the lens simulating the depth-of-field
  real_t lens_radius;

public:

  /** Create a parameterized camera

      \param[in] look_from is the position of the camera

      \param[in] look_at is a point the camera is looking at

      \param[in] vup is the “view up” orientation for the
      camera. {0,1,0} means the usual vertical orientation

      \param[in] degree_vfov is the vertical field-of-view in degrees

      \param[in] aspect_ratio is the ratio between the camera image
      width and the camera image height

      \param[in] aperture is the lens aperture of the camera

      \param[in] focus_dist is the focus distance
  */
  camera(point look_from,
         point look_at,
         vec vup,
         real_t degree_vfov,
         real_t aspect_ratio,
         real_t aperture,
         real_t focus_dist
         )
    : origin { look_from }
  {
    auto theta = degrees_to_radians(degree_vfov);
    auto h = std::tan(theta/2);
    auto viewport_height = 2.0*h;
    auto viewport_width = aspect_ratio * viewport_height;

    w = unit_vector(look_from - look_at);
    u = unit_vector(sycl::cross(vup, w));
    v = sycl::cross(w, u);

    horizontal = focus_dist * viewport_width*u;
    vertical = focus_dist * viewport_height*v;
    lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

    lens_radius = aperture/2;
  }

  ray get_ray(real_t s, real_t t) const {
    vec rd = lens_radius*random_in_unit_disk();
    vec offset = u * rd.x() + v * rd.y();
    return { origin + offset,
             lower_left_corner + s*horizontal + t*vertical - origin - offset };
  }
};

#endif
