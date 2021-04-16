#ifndef HITABLE_VISITOR_HPP
#define HITABLE_VISITOR_HPP

#include "hitable.hpp"
#include "sycl.hpp"
#include "task_context.hpp"
#include "visit.hpp"

namespace raytracer::visitor {
inline bool
badouel_ray_triangle_intersec(const ray& r,
                              raytracer::scene::_triangle_coord const& tri,
                              real_t min, real_t max, hit_record& rec) {
  // Get triangle edge vectors and plane normal
  auto u = tri.v1 - tri.v0;
  auto v = tri.v2 - tri.v0;
  vec outward_normal = sycl::cross(u, v);

  auto w0 = r.origin() - tri.v0;
  auto a = -sycl::dot(outward_normal, w0);
  auto b = sycl::dot(outward_normal, r.direction());

  // ray is parallel to triangle plane
  if (sycl::fabs(b) < 0.000001f)
    return false;

  // intersection point of ray with triangle
  real_t length = a / b;
  if (length < 0)
    return false;
  else if (length < min || length > max)
    return false;

  vec hit_pt = r.at(length);
  auto uu = sycl::dot(u, u);
  auto uv = sycl::dot(u, v);
  auto vv = sycl::dot(v, v);
  auto w = hit_pt - tri.v0;
  auto wu = sycl::dot(w, u);
  auto wv = sycl::dot(w, v);
  auto D = uv * uv - uu * vv;

  auto s = (uv * wv - vv * wu) / D;
  auto t = (uv * wu - uu * wv) / D;
  if (s < 0.0f || s > 1.0f || t < 0.0f || (s + t) > 1.0f)
    return false;

  rec.set_face_normal(r, outward_normal);
  rec.t = length;
  rec.p = hit_pt;
  return true;
};

inline bool
moller_trumbore_triangle_intersec(const ray& r,
                                  raytracer::scene::_triangle_coord const& tri,
                                  real_t min, real_t max, hit_record& rec) {
  constexpr auto epsilon = 0.0000001f;

  // Get triangle edge vectors and plane normal
  auto edge1 = tri.v1 - tri.v0;
  auto edge2 = tri.v2 - tri.v0;
  vec h = sycl::cross(r.direction(), edge2);
  auto a = sycl::dot(edge1, h);
  auto a_abs = sycl::fabs(a);

  if (a_abs < epsilon)
    return false; // This ray is parallel to this triangle

  auto a_pos = a > 0.f;

  auto s = r.origin() - tri.v0;
  auto u = sycl::dot(s, h);
  auto u_pos = u > 0.f;

  if ((u_pos xor a_pos) || sycl::fabs(u) > a_abs)
    return false;

  auto q = sycl::cross(s, edge1);
  auto v = sycl::dot(r.direction(), q);
  auto v_pos = v > 0.f;
  if ((v_pos xor a_pos) || (sycl::fabs(u + v) > a_abs))
    return false;

  auto length = sycl::dot(edge2, q) / a;

  if (length < min || length > max)
    return false;

  vec hit_pt = r.at(length);

  rec.set_face_normal(r, sycl::cross(edge1, edge2));
  rec.t = length;
  rec.p = hit_pt;
  return true;
};

template <auto IntersectionStrategy = moller_trumbore_triangle_intersec>
struct _hitable_hit {
 private:
  using material_t = scene::material_t;
  task_context& ctx;
  const ray& r;
  real_t min;
  real_t max;
  hit_record& rec;
  material_t& hit_material_type;

 private:
 public:
  _hitable_hit(task_context& ctx, const ray& r, real_t min, real_t max,
               hit_record& rec, material_t& _hmt)
      : ctx { ctx }
      , r { r }
      , min { min }
      , max { max }
      , rec { rec }
      , hit_material_type { _hmt } {}

  bool operator()(scene::box const& b) const {
    hit_record temp_rec;
    scene::material_t temp_material_type;
    auto hit_anything = false;
    auto closest_so_far = max;
    // Checking if the ray hits any of the sides
    for (const auto& side : b.sides) {
      if (dev_visit(_hitable_hit(ctx, r, min, closest_so_far, temp_rec,
                                 temp_material_type),
                    side)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
        hit_material_type = temp_material_type;
      }
    }
    return hit_anything;
  }

  bool operator()(scene::constant_medium const& cm) const {
    auto& rng = ctx.rng;
    hit_material_type = cm.phase_function;
    material_t temp_material_type;
    hit_record rec1, rec2;
    if (!dev_visit(
            _hitable_hit(ctx, r, -infinity, infinity, rec1, temp_material_type),
            cm.boundary)) {
      return false;
    }

    if (!dev_visit(_hitable_hit(ctx, r, rec1.t + 0.0001f, infinity, rec2,
                                temp_material_type),
                   cm.boundary)) {
      return false;
    }

    if (rec1.t < min)
      rec1.t = min;
    if (rec2.t > max)
      rec2.t = max;
    if (rec1.t >= rec2.t)
      return false;
    if (rec1.t < 0)
      rec1.t = 0;

    const auto ray_length = sycl::length(r.direction());
    /// Distance between the two hitpoints affect of probability
    /// of the ray hitting a smoke particle
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = cm.neg_inv_density * sycl::log(rng.real());

    /// With lower density, hit_distance has higher probabilty
    /// of being greater than distance_inside_boundary
    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec { 1, 0, 0 }; // arbitrary
    rec.front_face = true;        // also arbitrary
    return true;
  }

  template <bool x_plane, bool y_plane, bool z_plane>
  bool operator()(scene::rect<x_plane, y_plane, z_plane> const& pl) {
    hit_material_type = pl.material_type;
    auto origin =
        scene::rect<x_plane, y_plane, z_plane>::plan_basis(r.origin());
    auto direction =
        scene::rect<x_plane, y_plane, z_plane>::plan_basis(r.direction());

    auto t = (pl.k - origin.z()) / direction.z();
    if (t < min || t > max)
      return false;
    auto a = origin.x() + t * direction.x();
    auto b = origin.y() + t * direction.y();
    if (a < pl.a0 || a > pl.a1 || b < pl.b0 || b > pl.b1)
      return false;
    rec.u = (a - pl.a0) / (pl.a1 - pl.a0);
    rec.v = (b - pl.b0) / (pl.b1 - pl.b0);
    rec.t = t;
    rec.p = r.at(rec.t);
    vec outward_normal = vec(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    return true;
  }

  /// Compute ray interaction with sphere
  bool operator()(scene::sphere const& s) {
    hit_material_type = s.material_type;

    auto actual_center = s.center(r.time());

    /*(P(t)-C).(P(t)-C)=r^2
    in the above sphere equation P(t) is the point on sphere hit by the ray
    (A+tb−C)⋅(A+tb−C)=r^2
    (t^2)b⋅b + 2tb⋅(A−C) + (A−C)⋅(A−C)−r^2 = 0
    There can 0 or 1 or 2 real roots*/
    vec oc = r.origin() - actual_center; // oc = A-C
    auto a = sycl::dot(r.direction(), r.direction());
    auto b = sycl::dot(oc, r.direction());
    auto c = sycl::dot(oc, oc) - s.radius * s.radius;
    auto discriminant = b * b - a * c;
    // Real roots if discriminant is positive
    if (discriminant > 0) {
      // First root
      auto temp = (-b - sycl::sqrt(discriminant)) / a;
      if (temp < max && temp > min) {
        rec.t = temp;
        // Ray hits the sphere at p
        rec.p = r.at(rec.t);
        vec outward_normal = (rec.p - actual_center) / s.radius;
        // To set if hit point is on the front face and the outward normal in
        // rec
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
        vec outward_normal = (rec.p - actual_center) / s.radius;
        rec.set_face_normal(r, outward_normal);
        // Update u and v values in the hit record
        std::tie(rec.u, rec.v) = mercator_coordinates(rec.normal);
        return true;
      }
    }
    // No real roots
    return false;
  }

  bool operator()(scene::triangle const & tri) {
    hit_material_type = tri.material_type;
    return IntersectionStrategy(r, tri, min, max, rec);
  }

  bool operator()(std::monostate) {
    assert(fase && "unreachable");
    return false;
  }
};

using hitable_hit = _hitable_hit<>;

} // namespace raytracer::visitor

#endif