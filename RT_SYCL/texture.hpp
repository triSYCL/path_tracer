#ifndef RT_SYCL_TEXTURE_HPP
#define RT_SYCL_TEXTURE_HPP
#include "vec3.hpp"
#include <iostream>
#include <variant>
#include <vector>

//Solid texture consists of a single color
struct solid_texture {
    solid_texture() = default;
    solid_texture(color c)
        : color_value { c }
    {
    }
    solid_texture(double red, double green, double blue)
        : solid_texture { color(red, green, blue) }
    {
    }

    color value(double u, double v, const vec3& p) const
    {
        return color_value;
    }

private:
    color color_value;
};

//takes two solid_textures to create checker pattern
struct checker_texture {
    checker_texture() = default;
    checker_texture(solid_texture x, solid_texture y)
        : odd { x }
        , even { y }
    {
    }
    checker_texture(color c1, color c2)
        : even { solid_texture { c1 } }
        , odd { solid_texture { c2 } }
    {
    }
    color value(double u, double v, const point3& p) const
    {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd.value(u, v, p);
        else
            return even.value(u, v, p);
    }
    solid_texture odd;
    solid_texture even;
};

//takes input image as texture
struct image_texture {
    image_texture()
        : data { nullptr }
        , width(0)
        , bytes_per_scanline(0)
    {
    }
    image_texture(const char* filename)
    {
        /*auto components_per_pixel = bytes_per_pixel;
        data = stbi_load(
            filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!data) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            width = height = 0;
        }

        bytes_per_scanline = bytes_per_pixel * width;
        */
    }
    ~image_texture()
    {
        delete data;
    }
    color value(double u, double v, const point3& p) const
    {
        //if texture data is unavailble return solid cyan
        if (data == nullptr)
            return { 0, 1, 1 };

        /*u = clamp(u, 0.0, 1.0);
        v = 1.0 - clamp(v, 0.0, 1.0);

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        if (i >= width)
            i = width - 1;
        if (j >= height)
            j = height - 1;

        const auto color_scale = 1.0 / 255.0;
        auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
        */
    }
    unsigned char* data;
    int bytes_per_pixel = 3;
    int width;
    int bytes_per_scanline;
};

//to visit value() based on the texture
struct call_value {
    call_value(double u, double v, const vec3& p)
        : u { u }
        , v { v }
        , p { p }
    {
    }
    double u;
    double v;
    vec3 p;
    color operator()(const checker_texture& n) { return n.value(u, v, p); }
    color operator()(const image_texture& i) { return i.value(u, v, p); }
    color operator()(const solid_texture& s) { return s.value(u, v, p); }
};

using Texture = std::variant<checker_texture, solid_texture>;

#endif
