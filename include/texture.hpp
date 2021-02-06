#ifndef RT_SYCL_TEXTURE_HPP
#define RT_SYCL_TEXTURE_HPP
#include "hitable.hpp"
#include "vec.hpp"
#include <cmath>
#include <iostream>

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
  color value(const hit_record&) const { return color_value; }

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
  color value(const hit_record& rec) const {
    auto sines =
        sin(10 * rec.p.x()) * sin(10 * rec.p.y()) * sin(10 * rec.p.z());
    if (sines < 0)
      return odd.value(rec);
    else
      return even.value(rec);
  }
  solid_texture odd;
  solid_texture even;
};

/// A texture based on an image
struct image_texture {

  unsigned char* data {};
  int width {};
  int height {};

  /// The repetition rate of the image
  float cyclic_frequency { 1 };

  int bytes_per_scanline {};

  static constexpr auto bytes_per_pixel = 3;

  /** Create a texture from an image file

        \param[in] file_name is the path name to the image file

        \param[in] cyclic_frequency is an optional repetition rate of
        the image in the texture
*/
  image_texture(const char* file_name, float cyclic_frequency = 1)
      : cyclic_frequency { cyclic_frequency } {
    auto components_per_pixel = bytes_per_pixel;
    data = stbi_load(file_name, &width, &height, &components_per_pixel,
                     bytes_per_pixel);

    if (!data) {
      std::cerr << "ERROR: Could not load texture image file '" << file_name
                << "'.\n"
                << stbi_failure_reason() << std::endl;
      width = height = 0;
    }
    // \todo deallocate the image memory in the constructor

    bytes_per_scanline = bytes_per_pixel * width;
  }

  /// Get the color for the texture at the given place
  /// \todo rename this value() to color() everywhere?
  color value(const hit_record& rec) const {
    // If texture data is unavailable, return solid cyan
    if (!data)
      return { 0, 1, 1 };

    // The image is repeated by the repetition factor
    int i = std::fmod(rec.u * cyclic_frequency, 1) * (width - 1);
    // The image frame buffer is going downwards, so flip the y axis
    int j = (1 - std::fmod(rec.v * cyclic_frequency, 1)) * (height - 1);

    auto scale = 1.f / 255;
    auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;
    return { pixel[0] * scale, pixel[1] * scale, pixel[2] * scale };
  }
};

using texture_t = std::variant<checker_texture, solid_texture, image_texture>;

#endif
