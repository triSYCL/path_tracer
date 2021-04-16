#ifndef TEXTURE_VISITOR_HPP
#define TEXTURE_VISITOR_HPP
#include "hit_record.hpp"
#include "sycl.hpp"
#include "primitives.hpp"
#include "task_context.hpp"
#include "texture.hpp"

namespace raytracer::visitor {
struct texture_value {
 private:
  task_context& ctx;
  const hit_record& rec;

 public:
  texture_value(task_context& ctx, const hit_record& rec)
      : ctx { ctx }
      , rec { rec } {}

  // For solid texture, the color is same throughout the object
  color operator()(scene::solid_texture& t) const { return t.get_color(); }

  // Color value is different based on normalised spherical coordinates
  color operator()(scene::checker_texture& t) const {
    auto sines = sycl::sin(10 * rec.p.x()) * sycl::sin(10 * rec.p.y()) *
                 sycl::sin(10 * rec.p.z());
    auto const & [odd, even] = t.get_colors();
    return (sines < 0) ? odd : even;
  }

  color operator()(scene::image_texture& t) const {
    return t.color_at(rec.u, rec.v, ctx.texture_data);
  }

  color operator()(std::monostate) {
    assert(false && "unreachable");
    return { 0.f, 0.f, 0.f };
  }
};
} // namespace raytracer::visitor
#endif