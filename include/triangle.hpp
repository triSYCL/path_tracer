#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "material.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec.hpp"

struct _triangle_coord {
  point v0, v1, v2;
};

inline bool badouel_ray_triangle_intersec(const ray& r,
                                          _triangle_coord const& tri,
                                          real_t min, real_t max,
                                          hit_record& rec) {
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

inline bool moller_trumbore_triangle_intersec(const ray& r,
                                              _triangle_coord const& tri,
                                              real_t min, real_t max,
                                              hit_record& rec) {
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

// A triangle based on 3 points
template <auto IntersectionStrategy = moller_trumbore_triangle_intersec>
class _triangle : public _triangle_coord {
 public:
  _triangle() = default;
  _triangle(const point& _v0, const point& _v1, const point& _v2,
            const material_t& mat_type)
      : _triangle_coord { _v0, _v1, _v2 }
      , material_type { mat_type } {}

  /// Compute ray interaction with triangle
  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
           material_t& hit_material_type, LocalPseudoRNG&) const {
    hit_material_type = material_type;
    return IntersectionStrategy(r, *this, min, max, rec);
  }

  material_t material_type;
};

using triangle = _triangle<>;
#endif
