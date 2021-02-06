#include "sycl.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <math.h>
#include <thread>
#include <vector>

#include "render.hpp"

// Function to save image data in ppm format
void save_image(int width, int height, auto& fb_data) {
  std::cout << "P3\n" << width << " " << height << "\n255\n";
  for (int y = height - 1; y >= 0; y--) {
    for (int x = 0; x < width; x++) {
      auto pixel_index = y * width + x;
      int r = static_cast<int>(
          256 * std::clamp(sycl::sqrt(fb_data[pixel_index].x()), 0.0f, 0.999f));
      int g = static_cast<int>(
          256 * std::clamp(sycl::sqrt(fb_data[pixel_index].y()), 0.0f, 0.999f));
      int b = static_cast<int>(
          256 * std::clamp(sycl::sqrt(fb_data[pixel_index].z()), 0.0f, 0.999f));
      std::cout << r << " " << g << " " << b << "\n";
    }
  }
}

int main() {
  // Frame buffer dimensions
  constexpr auto width = buildparams::output_width;
  constexpr auto height = buildparams::output_height;

  // Allocate frame buffer on host
  std::array<color, (width * height)> fb;

  /// Graphical objects
  std::vector<hittable_t> hittables;

  // Generating a checkered ground and some random spheres
  texture_t t =
      checker_texture(color { 0.2f, 0.3f, 0.1f }, color { 0.9f, 0.9f, 0.9f });
  material_t m = lambertian_material(t);
  hittables.emplace_back(sphere(point { 0, -1000, 0 }, 1000, m));
  t = checker_texture(color { 0.9f, 0.9f, 0.9f }, color { 0.4f, 0.2f, 0.1f });

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      // Based on a random variable , the material type is chosen
      auto choose_mat = random_float();
      // Spheres are placed at a point randomly displaced from a,b
      point center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());
      if (sycl::length((center - point(4, 0.2f, 0))) > 0.9f) {
        if (choose_mat < 0.4f) {
          // Lambertian
          auto albedo = randomvec() * randomvec();
          hittables.emplace_back(
              sphere(center, 0.2f, lambertian_material(albedo)));
        } else if (choose_mat < 0.8f) {
          // Lambertian movig spheres
          auto albedo = randomvec() * randomvec();
          auto center2 = center + point { 0, random_float(0, 0.25f), 0 };
          hittables.emplace_back(sphere(center, center2, 0.0f, 1.0f, 0.2f,
                                        lambertian_material(albedo)));
        } else if (choose_mat < 0.95f) {
          // metal
          auto albedo = randomvec(0.5f, 1);
          auto fuzz = random_float(0, 0.5f);
          hittables.emplace_back(
              sphere(center, 0.2f, metal_material(albedo, fuzz)));
        } else {
          // glass
          hittables.emplace_back(
              sphere(center, 0.2f,
                     dielectric_material(1.5f, color { 1.0f, 1.0f, 1.0f })));
        }
      }
    }
  }

  // Pyramid
  hittables.emplace_back(
      triangle(point { 6.5f, 0.0f, 1.30f }, point { 6.25f, 0.50f, 1.05f },
               point { 6.5f, 0.0f, 0.80f },
               lambertian_material(color(0.68f, 0.50f, 0.1f))));
  hittables.emplace_back(
      triangle(point { 6.0f, 0.0f, 1.30f }, point { 6.25f, 0.50f, 1.05f },
               point { 6.5f, 0.0f, 1.30f },
               lambertian_material(color(0.89f, 0.73f, 0.29f))));
  hittables.emplace_back(triangle(
      point { 6.5f, 0.0f, 0.80f }, point { 6.25f, 0.50f, 1.05f },
      point { 6.0f, 0.0f, 0.80f }, lambertian_material(color(0.0f, 0.0f, 1))));
  hittables.emplace_back(triangle(
      point { 6.0f, 0.0f, 0.80f }, point { 6.25f, 0.50f, 1.05f },
      point { 6.0f, 0.0f, 1.30f }, lambertian_material(color(0.0f, 0.0f, 1))));

  // Glowing ball
  hittables.emplace_back(
      sphere(point { 4, 1, 0 }, 0.2f, lightsource_material(color(10, 0, 10))));

  // Four large spheres of metal, dielectric and Lambertian material types
  t = image_texture("../images/Xilinx.jpg");
  hittables.emplace_back(xy_rect(2, 4, 0, 1, -1, lambertian_material(t)));
  hittables.emplace_back(
      sphere(point { 4, 1, 2.25f }, 1, lambertian_material(t)));
  hittables.emplace_back(
      sphere(point { 0, 1, 0 }, 1,
             dielectric_material(1.5f, color { 1.0f, 0.5f, 0.5f })));
  hittables.emplace_back(sphere(point { -4, 1, 0 }, 1,
                                lambertian_material(color(0.4f, 0.2f, 0.1f))));
  hittables.emplace_back(sphere(point { 0, 1, -2.25f }, 1,
                                metal_material(color(0.7f, 0.6f, 0.5f), 0.0f)));

  t = image_texture { "../images/SYCL.png", 5 };

  // Add a sphere with a SYCL logo in the background
  hittables.emplace_back(
      sphere { point { -60, 3, 5 }, 4, lambertian_material { t } });

  // Add a metallic monolith
  hittables.emplace_back(
      box { point { 6.5f, 0, -1.5f }, point { 7.0f, 3.0f, -1.0f },
            metal_material { color { 0.7f, 0.6f, 0.5f }, 0.25f } });

  // Add a smoke ball
  sphere smoke_sphere =
      sphere { point { 5, 1, 3.5f }, 1,
               lambertian_material { color { 0.75f, 0.75f, 0.75f } } };
  hittables.emplace_back(
      constant_medium { smoke_sphere, 1, color { 1, 1, 1 } });

  // SYCL queue
  sycl::queue myQueue;

  // Camera setup
  /// Position of the camera
  point look_from { 13, 3, 3 };
  /// The center of the scene
  point look_at { 0, -1, 0 };
  // Make the camera oriented upwards
  vec vup { 0, 1, 0 };

  /// Vertical angle of view in degree
  real_t angle = 40;
  // Lens aperture. 0 if not depth-of-field
  real_t aperture = 0.04f;
  // Make the focus on the point we are looking at
  real_t focus_dist = length(look_at - look_from);
  camera cam {
    look_from, look_at,    vup,  angle, static_cast<real_t>(width) / height,
    aperture,  focus_dist, 0.0f, 1.0f
  };

  // Sample per pixel
  constexpr auto samples = 100;

  // SYCL render kernel
  render<width, height, samples>(myQueue, fb, hittables, cam);

  // Save image to file
  save_image(width, height, fb);

  return 0;
}
