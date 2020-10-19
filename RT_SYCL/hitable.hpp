#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec3.hpp"

enum class material_t { Lambertian,
    Metal,
    Dielectric };

class hit_record {
public:
    bool scatter_material(const ray& r_in, vec3& attenuation, ray& scattered)
    {
        switch (material_type) {
        case material_t::Lambertian:
            // Scattered ray is from point p in the direction of normal + random unit vector
            scattered = ray(p, normal + random_unit_vector());
            attenuation = std::visit([&](auto&& arg) { return arg.value(u, v, p); }, lambertian_albedo);
            return true;
        case material_t::Metal: {
            // Reflected is the reflected ray of r_in about the normal
            vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
            // Scattered ray depends on the value of fuzz
            scattered = ray(p, reflected + fuzz * random_in_unit_sphere());
            attenuation = albedo;
            return sycl::dot(scattered.direction(), normal) > 0;
        }
        case material_t::Dielectric:
            return false;
        default:
            return false;
        }
    }
    vec3 center;
    real_t radius;

    double t;
    point3 p;
    vec3 normal;
    double u;
    double v;

    // material properties
    material_t material_type;
    vec3 albedo;
    texture_t lambertian_albedo;
    real_t fuzz;
    real_t refraction_index;
};

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

template <class derived>
struct crtp {
    derived& underlying()
    {
        return static_cast<derived&>(*this);
    }
    const derived& underlying() const
    {
        return static_cast<const derived&>(*this);
    }
};

template <class geometry>
class hitable : crtp<geometry> {
public:
    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        return this->underlying().hit(r, min, max, rec);
    }
};

class sphere : public hitable<sphere> {
public:
    sphere() = default;
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

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        // Storing data in hit_record
        rec.material_type = material_type;
        rec.center = center;
        rec.radius = radius;
        if (material_type == material_t::Lambertian) {
            rec.lambertian_albedo = lambertian_albedo;
        }
        if (material_type == material_t::Metal) {
            rec.albedo = albedo;
            rec.fuzz = fuzz;
        }
        if (material_type == material_t::Dielectric) {
            rec.refraction_index = refraction_index;
        }

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
    texture_t lambertian_albedo;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

#endif
