#ifndef HITTABLE_H
#define HITTABLE_H

#include <variant>

#include "hit_record.hpp"
#include "material.hpp"
#include "rtweekend.hpp"
#include "vec.hpp"

struct hittable_hit_visitor {
 private:
  task_context& ctx;
  const ray& r;
  real_t min;
  real_t max;
  hit_record& rec;
  material_t& hit_material_type;

 public:
  hittable_hit_visitor(task_context& ctx, const ray& r, real_t min, real_t max,
                       hit_record& rec, material_t& hit_material_type)
      : ctx { ctx }
      , r { r }
      , min { min }
      , max { max }
      , rec { rec }
      , hit_material_type { hit_material_type } {}

  template <typename H> bool operator()(H&& hittable) {
    return hittable.hit(ctx, r, min, max, rec, hit_material_type);
  }

  bool operator()(std::monostate) {
    assert(fase && "unreachable");
    return false;
  }
};

#include "box.hpp"
#include "constant_medium.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

using hittable_t = std::variant<std::monostate, sphere, xy_rect, triangle, box,
                                constant_medium>;
#endif
