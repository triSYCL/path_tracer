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

auto badouel_ray_triangle_intersec = [](const ray& r, _triangle_coord const & tri, real_t min, real_t max, hit_record& rec)->bool
{
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

// A triangle based on 3 points
template<auto IntersectionStrategy = badouel_ray_triangle_intersec>
class _triangle:public _triangle_coord {
 public:
  _triangle() = default;
  _triangle(const point& _v0, const point& _v1, const point& _v2,
		   const material_t& mat_type)
	  : _triangle_coord{_v0, _v1, _v2}
	  , material_type { mat_type } {}

  /// Compute ray interaction with triangle
  bool hit(const ray& r, real_t min, real_t max, hit_record& rec,
		   material_t& hit_material_type) const {
	hit_material_type = material_type;
	return IntersectionStrategy(r, *this, min, max, rec);
  }

  material_t material_type;
};

using triangle = _triangle<badouel_ray_triangle_intersec>;
#endif
