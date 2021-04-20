#ifndef RT_SYCL_TEXTURE_HPP
#define RT_SYCL_TEXTURE_HPP
#include "hit_record.hpp"
#include "rtweekend.hpp"
#include "vec.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include "sycl.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// Solid texture consists of a single color
struct solid_texture {
  solid_texture() = default;
  solid_texture(const color& c)
      : color_value { c } {}
  solid_texture(real_t red, real_t green, real_t blue)
      : solid_texture { color { red, green, blue } } {}
  // For solid texture, the color is same throughout the sphere
  color value(auto&, const hit_record&) const { return color_value; }

 private:
  color color_value;
};

// Takes two solid_textures to create checker pattern
struct checker_texture {
  checker_texture() = default;

  checker_texture(const solid_texture& x, const solid_texture& y)
      : odd { x }
      , even { y } {}
  checker_texture(const color& c1, const color& c2)
      : odd { solid_texture { c1 } }
      , even { solid_texture { c2 } } {}
  // Color value is different based on normalised spherical coordinates
  color value(auto& ctx, const hit_record& rec) const {
    auto sines = sycl::sin(10 * rec.p.x()) * sycl::sin(10 * rec.p.y()) *
                 sycl::sin(10 * rec.p.z());
    if (sines < 0)
      return odd.value(ctx, rec);
    else
      return even.value(ctx, rec);
  }
  solid_texture odd;
  solid_texture even;
};

using texture_t = std::variant<std::monostate, checker_texture, solid_texture>;

#endif
