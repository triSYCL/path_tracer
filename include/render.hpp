#include <array>
#include <type_traits>
#include <vector>

#include "box.hpp"
#include "build_parameters.hpp"
#include "camera.hpp"
#include "constant_medium.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "rtweekend.hpp"
#include "sphere.hpp"
#include "sycl.hpp"
#include "texture.hpp"
#include "triangle.hpp"
#include "vec.hpp"
#include "visit.hpp"

using hittable_t =
	std::variant<sphere, xy_rect, triangle, box, constant_medium>;
namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
} // namespace constants

template<int width, int height, int samples, int depth>
auto hit_world (const ray& r, real_t min, real_t max,
								 hit_record& rec,
				sycl::accessor<hittable_t, 1, sycl::access::mode::read,
							   sycl::access::target::global_buffer> const & hittable_acc,
								 material_t& material_type){
  // Check if ray hits anything in the world
  hit_record temp_rec;
  material_t temp_material_type;
  auto hit_anything = false;
  auto closest_so_far = max;
  // Checking if the ray hits any of the spheres
  for (auto i = 0; i < hittable_acc.get_count(); i++) {
	if (dev_visit(
			[&](auto&& arg) {
			  return arg.hit(r, min, closest_so_far, temp_rec,
							 temp_material_type);
			},
			hittable_acc[i])) {
	  hit_anything = true;
	  closest_so_far = temp_rec.t;
	  rec = temp_rec;
	  material_type = temp_material_type;
	}
  }
  return hit_anything;
};

template<int width, int height, int samples, int depth>
inline auto get_color (const ray& r,
					   sycl::accessor<hittable_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>const & hittable_acc){
  ray cur_ray = r;
  color cur_attenuation { 1.0f, 1.0f, 1.0f };
  ray scattered;
  color emitted;
  material_t material_type;
  for (auto i = 0; i < depth; i++) {
	hit_record rec;
	if (hit_world<width, height, samples, depth>(cur_ray, real_t { 0.001f }, infinity, rec, hittable_acc,
				  material_type)) {
	  emitted = dev_visit([&](auto&& arg) { return arg.emitted(rec); },
						  material_type);
	  if (dev_visit(
			  [&](auto&& arg) {
				return arg.scatter(cur_ray, rec, cur_attenuation, scattered);
			  },
			  material_type)) {
		// On hitting the object, the ray gets scattered
		cur_ray = scattered;
	  } else {
		// Ray did not get scattered or reflected
		return emitted;
	  }
	} else {
	  /**
	   * If ray doesn't hit anything during iteration linearly blend white and
	   * blue color depending on the height of the y coordinate after scaling
	   * the ray direction to unit length. While -1.0f < y < 1.0f, hit_pt is
	   * between 0 and 1. This produces a blue to white gradient in the
	   * background
	   */
	  vec unit_direction = unit_vector(cur_ray.direction());
	  auto hit_pt = 0.5f * (unit_direction.y() + 1.0f);
	  color c = (1.0f - hit_pt) * color { 1.0f, 1.0f, 1.0f } +
				hit_pt * color { 0.5f, 0.7f, 1.0f };
	  return emitted + cur_attenuation * c;
	}
  }
  // If not returned within max_depth return black
  return color { 0.0f, 0.0f, 0.0f };
};

template <int width, int height, typename T>
void executor(trisycl::handler& cgh, const T& render_kernel) {
  if constexpr (buildparams::use_single_task) {
	cgh.single_task([render_kernel](){
	  for (int x_coord = 0; x_coord != width; ++x_coord)
		for (int y_coord = 0; y_coord != height; ++y_coord) {
		  render_kernel(x_coord, y_coord);
		}
	});
  } else {
	const auto global = sycl::range<2>(width, height);
	const auto local = sycl::range<2>(constants::TileX, constants::TileY);
	cgh.parallel_for_work_group(global, local, [render_kernel](sycl::group<2> g) {
		g.parallel_for_work_item([&](sycl::h_item<2> item){
			auto gid = item.get_global_id();
			const auto x_coord = gid[0];
			const auto y_coord = gid[1];
			render_kernel(x_coord, y_coord);
		});
	});
  }
}

// Render function to call the render kernel
template <int width, int height, int samples>
void render(sycl::queue queue, std::array<color, width*height>& fb, std::vector<hittable_t>& hittables, camera& cam) {
  auto constexpr depth = 50;
  auto frame_buf = sycl::buffer<color, 2>(fb.data(), sycl::range<2>(height, width));
  auto hittables_buf =
	  sycl::buffer<hittable_t, 1>(hittables.data(), sycl::range<1>(hittables.size()));
  // Submit command group on device
  queue.submit([&](sycl::handler& cgh) {
	// Get memory access
	auto frame_ptr = frame_buf.get_access<sycl::access::mode::discard_write>(cgh);
	auto hittables_acc = hittables_buf.get_access<sycl::access::mode::read>(cgh);
	// Construct kernel functor

	executor<width, height>(cgh, [=](int x_coord, int y_coord){
		color final_color(0.0f, 0.0f, 0.0f);
		for (auto i = 0; i < samples; i++) {
		  const auto u = (x_coord + random_float()) / width;
		  const auto v = (y_coord + random_float()) / height;
		  // u and v are points on the viewport
		  ray r = cam.get_ray(u, v);
		  final_color += get_color<width, height, samples, depth>(r, hittables_acc);
		}
		final_color /= static_cast<real_t>(samples);

		// Write final color to the frame buffer global memory
		frame_ptr[y_coord][x_coord] = final_color;
	});
  });
}
