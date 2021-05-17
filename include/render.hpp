#include <array>
#include <type_traits>
#include <variant>
#include <vector>

#include "build_parameters.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "sycl.hpp"
#include "texture.hpp"
#include "vec.hpp"
#include "visit.hpp"

inline auto render_pixel(int width, int height, int depth, int samples, int x_coord, int y_coord,
                         camera const& cam, auto& hittable_acc, auto fb_acc) {
  auto get_color = [&](const ray& r) {
    auto hit_world = [&](const ray& r, hit_record& rec,
                         material_t& material_type) {
      hit_record temp_rec;
      material_t temp_material_type;
      auto hit_anything = false;
      auto closest_so_far = infinity;
      // Checking if the ray hits any of the spheres
      for (auto i = 0; i < hittable_acc.get_count(); i++) {
        if (dev_visit(monostate_dispatch(
                          [&](auto&& object) {
                            return object.hit(r, 0.001f, closest_so_far,
                                              temp_rec, temp_material_type);
                          },
                          false),
                      hittable_acc[i])) {
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
        emitted = dev_visit(
            monostate_dispatch([&](auto&& mat) { return mat.emitted(rec); },
                               color { 0.f, 0.f, 0.f }),
            material_type);
        if (dev_visit(monostate_dispatch(
                          [&](auto&& mat) {
                            return mat.scatter(cur_ray, rec, cur_attenuation,
                                               scattered);
                          },
                          false),
                      material_type)) {
          // On hitting the object, the ray gets scattered
          cur_ray = scattered;
        } else {
          // Ray did not get scattered or reflected
          return emitted;
        }
      } else {
        /**
         If ray doesn't hit anything during iteration linearly blend white and
         blue color depending on the height of the y coordinate after scaling
         the ray direction to unit length. While -1.0f < y < 1.0f, hit_pt is
         between 0 and 1. This produces a blue to white gradient in the
         background
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
  uint32_t seed = std::hash<int> {}((x_coord << 16) | (y_coord & 0xFFFF));
  LocalPseudoRNG rng(seed);
  color final_color(0.0f, 0.0f, 0.0f);
  for (auto i = 0; i < samples; i++) {
    const auto u = (x_coord + rng.real()) / width;
    const auto v = (y_coord + rng.real()) / height;
    // u and v are points on the viewport
    ray r = cam.get_ray(u, v, rng);
    final_color += get_color(r);
  }
  final_color /= static_cast<real_t>(samples);

  // Write final color to the frame buffer global memory
  fb_acc[y_coord][x_coord] = final_color;
}

struct PixelRender;

inline void executor(int width, int height, int depth, int samples, sycl::handler& cgh,
                     camera const& cam_ptr, auto& hittable_acc, auto& fb_acc) {
  if constexpr (buildparams::use_single_task) {
    cgh.single_task<PixelRender>([=] {
      for (int x_coord = 0; x_coord != width; ++x_coord)
        for (int y_coord = 0; y_coord != height; ++y_coord) {
          render_pixel(width, height, depth, samples, x_coord, y_coord,
                                       cam_ptr, hittable_acc, fb_acc);
        }
    });
  } else {
    const auto global = sycl::range<2>(height, width);

    cgh.parallel_for<PixelRender>(global, [=](sycl::item<2> item) {
      auto gid = item.get_id();
      const auto x_coord = gid[1];
      const auto y_coord = gid[0];
      render_pixel(width, height, depth, samples, x_coord, y_coord,
                                   cam_ptr, hittable_acc, fb_acc);
    });
  }
}

// Render function to call the render kernel
void render(int width, int height, int depth, int samples, sycl::queue& queue,
            sycl::buffer<color, 2>& frame_buf,
            std::vector<hittable_t>& hittables, camera& cam) {
  const auto nb_hittable = hittables.size();
  auto hittables_buf = sycl::buffer<hittable_t, 1>(hittables.data(),
                                                   sycl::range<1>(nb_hittable));

  // Submit command group on device
  queue.submit([&](sycl::handler& cgh) {
    auto fb_acc = frame_buf.get_access<sycl::access::mode::discard_write>(cgh);
    auto hittables_acc =
        hittables_buf.get_access<sycl::access::mode::read>(cgh);
    executor(width, height, depth, samples, cgh, cam, hittables_acc,
                             fb_acc);
  });
}
