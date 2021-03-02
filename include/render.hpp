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
                         color* fb_ptr, LocalPseudoRNG& rng) {
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
                                 temp_material_type, rng);
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
                  return arg.scatter(cur_ray, rec, cur_attenuation, scattered,
                                     rng);
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
        return cur_attenuation * c;
      }
    }
    // If not returned within max_depth return black
    return color { 0.0f, 0.0f, 0.0f };
  };

  color final_color(0.0f, 0.0f, 0.0f);
  for (auto i = 0; i < samples; i++) {
    const auto u = (x_coord + rng.float_t()) / width;
    const auto v = (y_coord + rng.float_t()) / height;
    // u and v are points on the viewport
    ray r = cam.get_ray(u, v, rng);
    final_color += get_color(r);
  }
  final_color /= static_cast<real_t>(samples);

  // Write final color to the frame buffer global memory
  fb_ptr[y_coord * width + x_coord] = final_color;
}

template <int width, int height, int samples, int depth>
inline void executor(sycl::handler& cgh, camera const& cam_ptr,
                     hittable_t const* hittable_ptr, size_t nb_hittable,
                     color* fb_ptr) {
  if constexpr (buildparams::use_single_task) {
    cgh.single_task([=] {
      LocalPseudoRNG rng;
      for (int x_coord = 0; x_coord != width; ++x_coord)
        for (int y_coord = 0; y_coord != height; ++y_coord) {
          render_pixel<width, height, samples, depth>(
              x_coord, y_coord, cam_ptr, hittable_ptr, nb_hittable, fb_ptr, rng);
        }
    });
  } else {
    const auto global = sycl::range<2>(height, width);

    cgh.parallel_for(global, [=](sycl::item<2> item) {
      auto gid = item.get_id();
      const auto x_coord = gid[1];
      const auto y_coord = gid[0];
      auto init_generator_state = std::hash<std::size_t>{}(item.get_linear_id());
      LocalPseudoRNG rng(init_generator_state);
      render_pixel<width, height, samples, depth>(
          x_coord, y_coord, cam_ptr, hittable_ptr, nb_hittable, fb_ptr, rng);
    });
  }
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

  // Submit command group on device
  queue.submit([&](sycl::handler& cgh) {
    auto fb_acc = frame_buf.get_access<sycl::access::mode::discard_write>(cgh);
    auto hittables_acc =
        hittables_buf.get_access<sycl::access::mode::read>(cgh);

    hittable_t const* hittable_ptr = hittables_acc.get_pointer();
    color* fb_ptr = fb_acc.get_pointer();

    executor<width, height, samples, depth>(cgh, cam, hittable_ptr, nb_hittable,
                                            fb_ptr);

  });
}
