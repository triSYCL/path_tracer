#include <SYCL/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <thread>
#include <vector>

#include "render.hpp"

// Function to save image data in ppm format
template <int width, int height>
void save_image(color* fb_data)
{
    std::cout << "P3\n"
              << width << " " << height << "\n255\n";
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            auto pixel_index = y * width + x;
            int r = static_cast<int>(256 * std::clamp(sycl::sqrt(fb_data[pixel_index].x()), 0.0, 0.999));
            int g = static_cast<int>(256 * std::clamp(sycl::sqrt(fb_data[pixel_index].y()), 0.0, 0.999));
            int b = static_cast<int>(256 * std::clamp(sycl::sqrt(fb_data[pixel_index].z()), 0.0, 0.999));
            std::cout << r << " " << g << " " << b << "\n";
        }
    }
}

int main()
{
    // Frame buffer dimensions
    constexpr auto width = 800;
    constexpr auto height = 480;

    // Allocate frame buffer on host
    std::vector<color> fb(width*height);

    /// Graphical objects
    std::vector<hittable_t> hittables;

    // Generating a checkered ground and some random spheres
    texture_t t = checker_texture(color { 0.2, 0.3, 0.1 }, color { 0.9, 0.9, 0.9 });
    material_t m = lambertian_material(t);
    hittables.emplace_back(sphere(point { 0, -1000, 0 }, 1000, m));
    t = checker_texture(color { 0.9, 0.9, 0.9 }, color { 0.4, 0.2, 0.1 });

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            // Based on a random variable , the material type is chosen
            auto choose_mat = random_double();
            // Spheres are placed at a point randomly displaced from a,b
            point center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
            if (sycl::length((center - point(4, 0.2, 0))) > 0.9) {
                if (choose_mat < 0.4) {
                    // Lambertian
                    auto albedo = randomvec() * randomvec();
                    hittables.emplace_back(sphere(center, 0.2, lambertian_material(albedo)));
                }else if(choose_mat < 0.8){
                    // Lambertian movig spheres
                    auto albedo = randomvec() * randomvec();
                    auto center2 = center + point { 0, random_double(0, 0.25), 0 };
                    hittables.emplace_back(moving_sphere(center, center2, 0.0, 1.0, 0.2, lambertian_material(albedo)));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = randomvec(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    hittables.emplace_back(sphere(center, 0.2, metal_material(albedo, fuzz)));
                } else {
                    //glass
                    hittables.emplace_back(sphere(center, 0.2, dielectric_material(1.5, color {1.0, 1.0, 1.0 })));
                }
            }
        }
    }

    // Pyramid
    hittables.emplace_back(triangle(point { 6.5, 0.0, 1.30 }, point { 6.25, 0.50, 1.05 }, point { 6.5, 0.0, 0.80 }, lambertian_material(color(0.68, 0.50, 0.1))));
    hittables.emplace_back(triangle(point { 6.0, 0.0, 1.30 }, point { 6.25, 0.50, 1.05 }, point { 6.5, 0.0, 1.30 }, lambertian_material(color(0.89, 0.73, 0.29))));
    hittables.emplace_back(triangle(point { 6.5, 0.0, 0.80 }, point { 6.25, 0.50, 1.05 }, point { 6.0, 0.0, 0.80 }, lambertian_material(color(0.0, 0.0, 1))));
    hittables.emplace_back(triangle(point { 6.0, 0.0, 0.80 }, point { 6.25, 0.50, 1.05 }, point { 6.0, 0.0, 1.30 }, lambertian_material(color(0.0, 0.0, 1))));

    // Glowing ball
    hittables.emplace_back(sphere(point { 4, 1, 0 }, 0.2, lightsource_material(color(10, 0, 10))));

    // Four large spheres of metal, dielectric and Lambertian material types
    t = image_texture("../images/Xilinx.jpg");
    hittables.emplace_back(xy_rect(2, 4, 0, 1, -1, lambertian_material(t)));
    hittables.emplace_back(sphere(point { 4, 1, 2.25 }, 1, lambertian_material(t)));
    hittables.emplace_back(sphere(point { 0, 1, 0 }, 1, dielectric_material(1.5, color {1.0, 0.5, 0.5 })));
    hittables.emplace_back(sphere(point { -4, 1, 0 }, 1, lambertian_material(color(0.4, 0.2, 0.1))));
    hittables.emplace_back(sphere(point { 0, 1, -2.25 }, 1, metal_material(color(0.7, 0.6, 0.5), 0.0)));

    t = image_texture { "../images/SYCL.png", 5 };

    // Add a sphere with a SYCL logo in the background
    hittables.emplace_back(sphere {  point { -60, 3, 5 }, 4,
                                     lambertian_material { t } });

    // Add a metallic monolith
    hittables.emplace_back(box { point{ 6.5, 0, -1.5 }, point { 7.0, 3.0, -1.0 }, 
                                     metal_material { color { 0.7, 0.6, 0.5 }, 0.25 } });

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
    real_t aperture = 0.04;
    // Make the focus on the point we are looking at
    real_t focus_dist = length(look_at - look_from);
    camera cam { look_from, look_at, vup, angle,
                 static_cast<real_t>(width)/height,
                 aperture, focus_dist, 0.0, 1.0 };

    // Sample per pixel
    constexpr auto samples = 100;

    // SYCL render kernel
    render<width, height, samples>(myQueue, fb.data(),
                                   hittables.data(), hittables.size(), cam);

    // Save image to file
    save_image<width, height>(fb.data());

    return 0;
}
