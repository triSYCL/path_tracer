#ifndef SPHERE_H
#define SPHERE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
//#include"hitable.hpp"
#include "material.hpp"
#include "vec3.hpp"


/* Computes normalised values of theta and phi. The input point p
corresponds to a point on a unit sphere centered at origin */
std::pair<double,double> get_sphere_uv(const point3& p)
{
    // phi is the angle around the axis
    auto phi = atan2(p.z(), p.x());
    // theta is the angle down from the pole
    auto theta = asin(p.y());
    // theta and phi together constitute the spherical coordinates
    // phi is between -pi and pi , u is between 0 and 1
    auto u = 1 - (phi + pi) / (2 * pi);
    // theta is between -pi/2 and pi/2 , v is between 0 and 1
    auto v = (theta + pi / 2) / pi;
    return { u, v };
}

class sphere {
public:
    sphere() = default;

    sphere(const vec3& cen, real_t r, Material_t mat_type)
        : center { cen }
        , radius { r }
        , Material_type { mat_type }
    {}

    // Constructor for lambertian sphere with color
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& color)
        : center { cen }
        , radius { r }
        , material_type { mat_type }
        , lambertian_albedo { solid_texture(color) }
    {
    }

    // Constructor for lambertian sphere with texture
    sphere(const vec3& cen, real_t r, material_t mat_type, texture_t& texture)
        : center { cen }
        , radius { r }
        , material_type { mat_type }
        , lambertian_albedo { texture }
    {
    }

    // Constructor for metal sphere with color
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& mat_color, real_t f)
        : center { cen }
        , radius { r }
        , material_type { mat_type }
        , albedo { mat_color }
        , fuzz { std::clamp(f, 0.0, 1.0) }
    {
    }

    // Constructor for dielectric sphere with color
    sphere(const vec3& cen, real_t r, material_t mat_type, real_t ref_idx)
        : center { cen }
        , radius { r }
        , material_type { mat_type }
        , refraction_index { ref_idx }
    {
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, Material_t& hit_material_type) const
    {
        // Storing data in hit_record
        //rec.material_type = material_type;
        rec.center = center;
        rec.radius = radius;
        hit_material_type = Material_type;
        
        /*(P(t)-C).(P(t)-C)=r^2
        in the above sphere equation P(t) is the point on sphere hit by the ray
        (A+tb−C)⋅(A+tb−C)=r^2
        (t^2)b⋅b + 2tb⋅(A−C) + (A−C)⋅(A−C)−r^2 = 0
        There can 0 or 1 or 2 real roots*/
        vec3 oc = r.origin() - center; // oc = A-C
        auto a = sycl::dot(r.direction(), r.direction());
        auto b = sycl::dot(oc, r.direction());
        auto c = sycl::dot(oc, oc) - radius * radius;
        auto discriminant = b * b - a * c;
        // Real roots if discriminant is positive
        if (discriminant > 0) {
            // First root
            auto temp = (-b - sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                // Ray hits the sphere at p
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                /* Update u and v values in the hit record. Normal of a
                point is calculated as above. Its the same way the point is 
                transformed into a point on unit sphere centered at origin.*/
                std::tie(rec.u, rec.v) = get_sphere_uv((point3)rec.normal);
                return true;
            }
            // Second root
            temp = (-b + sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                // Ray hits the sphere at p
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                // Update u and v values in the hit record
                std::tie(rec.u, rec.v) = get_sphere_uv((point3)rec.normal);
                return true;
            }
        }
        // No real roots
        return false;
    }

    // Geometry properties
    vec3 center;
    real_t radius;

    // Material properties
    material_t material_type;
    Material_t Material_type;
    texture_t lambertian_albedo;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

#endif