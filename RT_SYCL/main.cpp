#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <thread>
#include <vector>

#include <SYCL/sycl.hpp>

#include "camera.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "sphere.hpp"
#include "rectangle.hpp"
#include "ray.hpp"
#include "rtweekend.hpp"
#include "texture.hpp"
#include "vec3.hpp"

using int_type = std::uint32_t;
using hittable_t = std::variant<sphere>;
namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
}

template <int width, int height, int samples, int depth, int num_hittables>
class render_kernel {
public:
    render_kernel(sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> frame_ptr,
        sycl::accessor<sphere, 1, sycl::access::mode::read, sycl::access::target::global_buffer> hitable_ptr)
        : m_frame_ptr { frame_ptr }
        , m_hitable_ptr { hitable_ptr }
    {
    }

    void operator()(sycl::nd_item<2> item)
    {
        // Get our Ids
        const auto x_coord = item.get_global_id(0);
        const auto y_coord = item.get_global_id(1);
        // map the 2D indices to a single linear, 1D index
        const auto pixel_index = y_coord * width + x_coord;

        // Color sampling for antialiasing
        vec3 final_color(0.0, 0.0, 0.0);
        for (auto i = 0; i < samples; i++) {
            const auto u = (x_coord + random_double()) / width;
            const auto v = (y_coord + random_double()) / height;
            // u and v are points on the viewport
            ray r = get_ray(u, v);
            final_color += color(r, m_hitable_ptr.get_pointer(), depth);
        }
        final_color /= static_cast<real_t>(samples);

        // Write final color to the frame buffer global memory
        m_frame_ptr[pixel_index] = final_color;
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    // Check if ray hits anything in the world
    bool hit_world(const ray& r, real_t min, real_t max, hit_record& rec, sphere* spheres, material_t& material_type)
    {
        hit_record temp_rec;
        material_t temp_material_type;
        auto hit_anything = false;
        auto closest_so_far = max;
        // Checking if the ray hits any of the spheres
        for (auto i = 0; i < num_hittables; i++) {
            if (spheres[i].hit(r, min, closest_so_far, temp_rec, temp_material_type)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                material_type = temp_material_type;
            }
        }
        return hit_anything;
    }

    vec3 color(const ray& r, sphere* spheres, int max_depth)
    {
        ray cur_ray = r;
        vec3 cur_attenuation(1.0, 1.0, 1.0);
        ray scattered;
        material_t material_type;
        for (auto i = 0; i < max_depth; i++) {
            hit_record rec;
            if (hit_world(cur_ray, real_t { 0.001 }, infinity, rec, spheres,material_type)) {
                if(std::visit([&](auto&& arg) { return arg.scatter(cur_ray, rec, cur_attenuation, scattered); }, material_type)){
                    // On hitting the sphere, the ray gets scattered
                    cur_ray = scattered;
                } else {
                    // Ray did not get scattered or reflected
                    return vec3(0.0, 0.0, 0.0);
                }
            } else {
                /* If ray doesn't hit anything during iteration linearly blend white and 
                blue color depending on the height of the y coordinate after scaling the 
                ray direction to unit length. While -1.0 < y < 1.0, hit_pt is between 0 
                and 1. This produces a blue to white gradient in the background */
                vec3 unit_direction = unit_vector(cur_ray.direction());
                auto hit_pt = 0.5 * (unit_direction.y() + 1.0);
                vec3 c = (1.0 - hit_pt) * vec3(1.0, 1.0, 1.0) + hit_pt * vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        // If not returned within max_depth return black
        return vec3(0.0, 0.0, 0.0);
    }

    /* Computes ray from camera passing through 
    viewport local coordinates (s,t) based on viewport 
    width, height and focus distance */
    ray get_ray(real_t s, real_t t)
    {
        auto theta = degrees_to_radians(20);
        auto h = tan(theta / 2);
        auto aspect_ratio = 16.0 / 9.0;
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;
        auto focal_length = 1.0;
        vec3 look_from = { 13, 2, 3 };
        auto focus_dist = 10;

        vec3 w = unit_vector(look_from - vec3(0, 0, 0));
        vec3 u = unit_vector(sycl::cross(vec3(0, 1, 0), w));
        vec3 v = sycl::cross(w, u);

        // Camera arguments
        origin = look_from;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        auto lens_radius = 0.1 / 2;

        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    // Accessor objects
    sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> m_frame_ptr;
    sycl::accessor<sphere, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_hitable_ptr;
};

// Render function to call the render kernel
template <int width, int height, int samples, int num_hittables>
void render(sycl::queue queue, vec3* fb_data, const sphere* spheres)
{
    constexpr auto num_pixels = width * height;
    auto const depth = 5;
    auto frame_buf = sycl::buffer<vec3, 1>(fb_data, sycl::range<1>(num_pixels));
    auto sphere_buf = sycl::buffer<sphere, 1>(spheres, sycl::range<1>(num_hittables));
    // Submit command group on device
    queue.submit([&](sycl::handler& cgh) {
        // Get memory access
        auto frame_ptr = frame_buf.get_access<sycl::access::mode::write>(cgh);
        auto spheres_ptr = sphere_buf.get_access<sycl::access::mode::read>(cgh);
        // Setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(constants::TileX, constants::TileY);
        const auto index_space = sycl::nd_range<2>(global, local);
        // Construct kernel functor
        auto render_func = render_kernel<width, height, samples, depth, num_hittables>(frame_ptr, spheres_ptr);
        // Execute kernel
        cgh.parallel_for(index_space, render_func);
    });
}

// Function to save image data in ppm format
template <int width, int height>
void save_image(vec3* fb_data)
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
    constexpr auto num_pixels = width * height;
    constexpr auto num_hittables = 487;
    constexpr auto samples = 100;
    std::vector<sphere> spheres;

    // Generating a checkered ground and some random spheres
    texture_t t = checker_texture(color { 0.2, 0.3, 0.1 }, color { 0.9, 0.9, 0.9 });
    material_t m = lambertian_material(t);
    spheres.emplace_back(vec3 { 0, -1000, 0 }, 1000, m);

    // //spheres.push_back(sphere(vec3(0, -1000, 0), 1000, material_t::Lambertian, color(0.2, 0.2, 0.2)));
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            // Based on a random variable , the material type is chosen
            auto choose_mat = random_double();
            // Spheres are placed at a point randomly displaced from a,b
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
            if (sycl::length((center - point3(4, 0.2, 0))) > 0.9) {
                if (choose_mat < 0.8) {
                    // lambertian
                    auto albedo = randomvec3() * randomvec3();
                    //spheres.emplace_back(center, 0.2, material_t::Lambertian, albedo);
                    spheres.emplace_back(center,0.2,lambertian_material(albedo));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = randomvec3(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    //spheres.emplace_back(center, 0.2, material_t::Metal, albedo, fuzz);
                    spheres.emplace_back(center, 0.2, metal_material(albedo,fuzz));
                }else{
                    //glass
                    spheres.emplace_back(center,0.2,dielectric_material(1.5));
                }
            }
        }
    }

    // Three large spheres of metal and lambertian material types
    t = image_texture("../RT_SYCL/Xilinx.jpg");
    spheres.emplace_back(point3 { 4, 1, 2.25 }, 1,lambertian_material(t));
    spheres.emplace_back(point3 { 0, 1, 0 }, 1,dielectric_material(1.5));
    spheres.emplace_back(point3 { -4, 1, 0 }, 1,lambertian_material(color(0.4,0.2,0.1)));
    spheres.emplace_back(point3 { 4, 1, 0 }, 1, metal_material(color(0.7,0.6,0.5),0.0));

    // SYCL queue
    sycl::queue myQueue;

    // Allocate frame buffer on host
    std::vector<vec3> fb(num_pixels);

    camera cam;

    // Sycl render kernel
    render<width, height, samples, num_hittables>(myQueue, fb.data(), spheres.data());

    // Save image to file
    save_image<width, height>(fb.data());

    return 0;
}
