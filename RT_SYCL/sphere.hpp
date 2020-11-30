#ifndef SPHERE_H
#define SPHERE_H

#include "material.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec.hpp"

/* Computes normalised values of theta and phi. The input vector p
corresponds to a vector passing through the centre of the a sphere 
and the hipoint on the surface of the sphere */
std::pair<double, double> mercator_coordinates(const vec& p)
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

    sphere(const point& cen, real_t r, material_t mat_type)
        : center { cen }
        , radius { r }
        , material_type { mat_type }
    {
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec, material_t& hit_material_type) const
    {
        hit_material_type = material_type;

        /*(P(t)-C).(P(t)-C)=r^2
        in the above sphere equation P(t) is the point on sphere hit by the ray
        (A+tb−C)⋅(A+tb−C)=r^2
        (t^2)b⋅b + 2tb⋅(A−C) + (A−C)⋅(A−C)−r^2 = 0
        There can 0 or 1 or 2 real roots*/
        vec oc = r.origin() - center; // oc = A-C
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
                vec outward_normal = (rec.p - center) / radius;
                // To set if hit point is on the front face and the outward normal in rec
                rec.set_face_normal(r, outward_normal);
                /* Update u and v values in the hit record. Normal of a
                point is calculated as above. This vector is used to also 
                used to get the mercator coordinates of the hitpoint.*/
                std::tie(rec.u, rec.v) = mercator_coordinates(rec.normal);
                return true;
            }
            // Second root
            temp = (-b + sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                // Ray hits the sphere at p
                rec.p = r.at(rec.t);
                //rec.normal = (rec.p - center) / radius;
                vec outward_normal = (rec.p - center) / radius;
                rec.set_face_normal(r, outward_normal);
                // Update u and v values in the hit record
                std::tie(rec.u, rec.v) = mercator_coordinates(rec.normal);
                return true;
            }
        }
        // No real roots
        return false;
    }

    // Geometry properties
    point center;
    real_t radius;

    // Material properties
    material_t material_type;
};

#endif
