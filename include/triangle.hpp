#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "material.hpp"
#include "primitives.hpp"

namespace raytracer::scene {
struct _triangle_coord {
  point v0, v1, v2;
};

class triangle : public _triangle_coord {
 public:
  triangle(const point& _v0, const point& _v1, const point& _v2,
            const material_t& mat_type)
      : _triangle_coord { _v0, _v1, _v2 }
      , material_type { mat_type } {}

  material_t material_type;
};
} // namespace raytracer::scene
#endif
