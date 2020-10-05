#include "vec3.hpp"
#include <iostream>
#include <variant>
#include <vector>

struct solid_texture {
    solid_texture() = default;
    solid_texture(color c)
        : color_value{c}
    {
    }
    solid_texture(double red, double green, double blue)
        : solid_texture{color(red, green, blue)}
    {
    }

    color value(double u, double v, const vec3& p) const
    {
        return color_value;
    }

private:
    color color_value;
};

struct checker_texture {
    //checker_texture(int x, int y) : color_value(x), color_type(y){}
    checker_texture(solid_texture x, solid_texture y)
        : odd{x}
        , even{y}
    {
    }
    checker_texture(color c1, color c2)
        : even { solid_texture { c1 }  }
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

struct call_value {
    call_value(double u, double v, const vec3& p)
        : u{u}
        , v{v}
        , p{p}
    {}
    double u;
    double v;
    vec3 p;
    color operator()(const checker_texture& n) { return n.value(u, v, p); }
    //int operator()(const image_texture& i) { return i.value(u,v); }
    color operator()(const solid_texture& s) { return s.value(u, v, p); }
};
