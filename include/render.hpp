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

template <int width, int height, int samples, int depth>
inline auto render_pixel(int x_coord, int y_coord, camera const& cam,
						 hittable_t const* hittable_ptr, int nb_hittable,
						 color* fb_ptr) {
  auto get_color = [&](const ray& r) {
	auto hit_world = [&](const ray& r, hit_record& rec,
						 material_t& material_type) {
	  hit_record temp_rec;
	  material_t temp_material_type;
	  auto hit_anything = false;
	  auto closest_so_far = infinity;
	  // Checking if the ray hits any of the spheres
	  for (auto i = 0; i < nb_hittable; i++) {
		if (dev_visit(
				[&](auto&& arg) {
				  return arg.hit(r, 0.001f, closest_so_far, temp_rec,
								 temp_material_type);
				},
				hittable_ptr[i])) {
		  hit_anything = true;
		  closest_so_far = temp_rec.t;
		  rec = temp_rec;
		  material_type = temp_material_type;
		}
	  }
	  return hit_anything;
	};

	ray cur_ray = r;
	color cur_attenuation { 1.0f, 1.0f, 1.0f };
	ray scattered;
	color emitted;
	material_t material_type;
	for (auto i = 0; i < depth; i++) {
	  hit_record rec;
	  if (hit_world(cur_ray, rec, material_type)) {
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

  color final_color(0.0f, 0.0f, 0.0f);
  for (auto i = 0; i < samples; i++) {
	const auto u = (x_coord + random_float()) / width;
	const auto v = (y_coord + random_float()) / height;
	// u and v are points on the viewport
	ray r = cam.get_ray(u, v);
	final_color += get_color(r);
  }
  final_color /= static_cast<real_t>(samples);

  // Write final color to the frame buffer global memory
  fb_ptr[y_coord * width + x_coord] = final_color;
}

template <int width> int find_immediate_least_divider(int val) {
  for (int i = val; i > 0; --i) {
	if ((val % i) == 0)
	  return i;
  }
  return 0;
}

// Render function to call the render kernel
template <int width, int height, int samples>
void render(sycl::queue& queue, std::array<color, width * height>& fb,
			std::vector<hittable_t>& hittables, camera& cam) {
  auto constexpr depth = 50;
  const auto nb_hittable = hittables.size();
  auto frame_buf =
	  sycl::buffer<color, 2>(fb.data(), sycl::range<2>(height, width));
  auto hittables_buf = sycl::buffer<hittable_t, 1>(hittables.data(),
												   sycl::range<1>(nb_hittable));

  const auto& dev = queue.get_device();
  const auto comp_unit =
	  (buildparams::use_single_task)
		  ? 1
		  : dev.get_info<sycl::info::device::max_compute_units>();
  const auto wg_size =
	  (buildparams::use_single_task)
		  ? 1
		  : dev.get_info<sycl::info::device::max_work_group_size>();

  const auto parallel_exec = comp_unit * wg_size;

  const auto total_work = width * height;

  const auto ideal_pg_height = height / parallel_exec;
  const auto pg_height = (ideal_pg_height == 0) ? 1 : ideal_pg_height;
  const auto pg_width =
	  (ideal_pg_height == 0)
		  ? find_immediate_least_divider<width>(total_work / parallel_exec)
		  : width;
  const auto nb_pg_line = width / pg_width;
  const auto required_pg_line =
	  height / ideal_pg_height + ((height % ideal_pg_height) ? 1 : 0);
  const auto required_pg = required_pg_line * nb_pg_line;

  // Submit command group on device
  queue.submit([=, &hittables_buf, &frame_buf, &cam](sycl::handler& cgh) {
	auto fb_acc = frame_buf.get_access<sycl::access::mode::discard_write>(cgh);
	auto hittables_acc =
		hittables_buf.get_access<sycl::access::mode::read>(cgh);
	hittable_t const* hittable_ptr = hittables_acc.get_pointer();
	color* fb_ptr = fb_acc.get_pointer();
	const auto global = sycl::range<1>(required_pg);
	const auto local = sycl::range<1>(wg_size);
	cgh.parallel_for_work_group(global, local, [=](sycl::group<1> g) {
	  g.parallel_for_work_item([&](sycl::h_item<1> item) {
		auto gid = item.get_global_id()[0];
		const auto grid_x = gid % nb_pg_line;
		const auto grid_y = gid / nb_pg_line;
		const auto start_x = grid_x * pg_width;
		const auto start_y = grid_y * pg_height;
		const auto max_x = start_x + pg_width;
		const auto th_max_y = start_y + pg_height;
		const auto max_y = (th_max_y > height) ? height : th_max_y;
		for (auto y = start_y; y < max_y; ++y) {
		  for (auto x = start_x; x < max_x; ++x) {
			render_pixel<width, height, samples, depth>(x, y, cam, hittable_ptr,
														nb_hittable, fb_ptr);
		  }
		}
	  });
	});
  });
}
