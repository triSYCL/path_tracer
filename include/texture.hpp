#ifndef RT_SYCL_TEXTURE_HPP
#define RT_SYCL_TEXTURE_HPP
#include "hitable.hpp"
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
  solid_texture(float red, float green, float blue)
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
    auto sines =
        sycl::sin(10 * rec.p.x()) * sycl::sin(10 * rec.p.y()) * sycl::sin(10 * rec.p.z());
    if (sines < 0)
      return odd.value(ctx, rec);
    else
      return even.value(ctx, rec);
  }
  solid_texture odd;
  solid_texture even;
};

/// A texture based on an image
struct image_texture {
  /** Create a texture from an image file

         \param[in] file_name is the path name to the image file

         \param[in] cyclic_frequency is an optional repetition rate of
         the image in the texture
 */
  static image_texture image_texture_factory(const char* file_name,
                                             float _cyclic_frequency = 1) {
    assert(!frozen);
    auto components_per_pixel = bytes_per_pixel;
    uint8_t* _data;
    int _w, _h;
    _data =
        stbi_load(file_name, &_w, &_h, &components_per_pixel, bytes_per_pixel);
    std::size_t _offset = 0;
    if (!_data) {
      std::cerr << "ERROR: Could not load texture image file '" << file_name
                << "'.\n"
                << stbi_failure_reason() << std::endl;
      _w = _h = 1;
    } else {
      auto size = bytes_per_pixel * _w * _h;
      _offset = texture_data.size() / 3;
      std::copy(_data, _data + size, std::back_inserter(texture_data));
    }
    return image_texture(_w, _h, _offset, _cyclic_frequency);
  }

  static sycl::buffer<uint8_t, 2> freeze() {
    assert(!frozen);
    frozen = true;
    return sycl::buffer<uint8_t, 2> { texture_data.data(),
                                      { texture_data.size() / 3, 3 } };
  }

  /// Get the color for the texture at the given place
  /// \todo rename this value() to color() everywhere?
  color value(auto& ctx, const hit_record& rec) const {
    // If texture data is unavailable, return solid cyan
    // The image is repeated by the repetition factor
    std::size_t i = sycl::fmod(rec.u * cyclic_frequency, (float)1) * (width - 1);
    // The image frame buffer is going downwards, so flip the y axis
    std::size_t j = (1 - sycl::fmod(rec.v * cyclic_frequency, (float)1)) * (height - 1);
    std::size_t local_offset = j * width + i;
    std::size_t pix_idx = local_offset + offset;
    auto scale = 1.f / 255;
    auto& texture_data = ctx.texture_data;
    return { texture_data[pix_idx * 3] * scale,
             texture_data[pix_idx * 3 + 1] * scale,
             texture_data[pix_idx * 3 + 2] * scale };
  }

 private:
  std::size_t width {};
  std::size_t height {};
  static constexpr auto bytes_per_pixel = 3;
  std::size_t offset;

  static std::vector<uint8_t> texture_data;
  static bool frozen;

  /// The repetition rate of the image
  float cyclic_frequency { 1 };
  image_texture(std::size_t _width, std::size_t _height, std::size_t _offset,
                float _cyclic_frequency)
      : width { _width }
      , height { _height }
      , offset { _offset }
      , cyclic_frequency { _cyclic_frequency } {}
};

using texture_t = std::variant<checker_texture, solid_texture, image_texture>;

// Start filled with the fallback texture (solid blue) for texture load error
std::vector<uint8_t> image_texture::texture_data { 0, 0, 1 };
bool image_texture::frozen = false;
#endif
