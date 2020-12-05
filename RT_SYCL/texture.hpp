#ifndef RT_SYCL_TEXTURE_HPP
#define RT_SYCL_TEXTURE_HPP
#include "hitable.hpp"
#include "vec.hpp"
#include <iostream>
#include <variant>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// Solid texture consists of a single color
struct solid_texture {
    solid_texture() = default;
    solid_texture(color c)
        : color_value { c }
    {
    }
    solid_texture(double red, double green, double blue)
        : solid_texture { color { red, green, blue } }
    {
    }
    // For solid texture, the color is same throughout the sphere
    color value(const hit_record& rec) const
    {
        return color_value;
    }

private:
    color color_value;
};

// Takes two solid_textures to create checker pattern
struct checker_texture {
    checker_texture() = default;

    checker_texture(solid_texture x, solid_texture y)
        : odd { x }
        , even { y }
    {
    }
    checker_texture(color c1, color c2)
        : odd { solid_texture { c1 } }
        , even { solid_texture { c2 } }
    {
    }
    // Color value is different based on normalised spherical coordinates
    color value(const hit_record& rec) const
    {
        auto sines = sin(10 * rec.p.x()) * sin(10 * rec.p.y()) * sin(10 * rec.p.z());
        if (sines < 0)
            return odd.value(rec);
        else
            return even.value(rec);
    }
    solid_texture odd;
    solid_texture even;
};

// Takes input image as texture
struct image_texture {
    image_texture()
        : data { nullptr }
        , width { 0 }
        , height { 0 }
        , bytes_per_scanline { 0 }
    {
    }
    image_texture(const char* filename)
    {
        auto components_per_pixel = bytes_per_pixel;
        data = stbi_load(
            filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!data) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            std::cerr << stbi_failure_reason() << std::endl;
            width = height = 0;
        }

        bytes_per_scanline = bytes_per_pixel * width;
    }
    color value(const hit_record& rec) const
    {
        // If texture data is unavailable return solid cyan
        if (data == nullptr)
            return { 0, 1, 1 };

        auto u = std::clamp(rec.u, 0.0, 1.0);
        auto v = 1.0 - std::clamp(rec.v, 0.0, 1.0);

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        if (i >= width)
            i = width - 1;
        if (j >= height)
            j = height - 1;

        const auto color_scale = 1.0 / 255.0;
        auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return { color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2] };
    }
    unsigned char* data;
    int bytes_per_pixel = 3;
    int width;
    int height;
    int bytes_per_scanline;
};

using texture_t = std::variant<checker_texture, solid_texture, image_texture>;

#endif
