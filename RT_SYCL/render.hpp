#include "camera.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "rtweekend.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "triangle.hpp"
#include "vec.hpp"
#include <SYCL/sycl.hpp>

using hittable_t = std::variant<sphere, xy_rect, triangle>;
namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
}

template <int width, int height, int samples, int depth, int num_hittables>
class render_kernel {
public:
    render_kernel(sycl::accessor<color, 1, sycl::access::mode::write, sycl::access::target::global_buffer> frame_ptr,
        sycl::accessor<hittable_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> hitable_ptr)
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
        color final_color(0.0, 0.0, 0.0);
        for (auto i = 0; i < samples; i++) {
            const auto u = (x_coord + random_double()) / width;
            const auto v = (y_coord + random_double()) / height;
            // u and v are points on the viewport
            ray r = get_ray(u, v);
            final_color += get_color(r, m_hitable_ptr.get_pointer(), depth);
        }
        final_color /= static_cast<real_t>(samples);

        // Write final color to the frame buffer global memory
        m_frame_ptr[pixel_index] = final_color;
    }

private:
    point origin;
    point lower_left_corner;
    vec horizontal;
    vec vertical;

    // Check if ray hits anything in the world
    bool hit_world(const ray& r, real_t min, real_t max, hit_record& rec, hittable_t* hittables, material_t& material_type)
    {
        hit_record temp_rec;
        material_t temp_material_type;
        auto hit_anything = false;
        auto closest_so_far = max;
        // Checking if the ray hits any of the spheres
        for (auto i = 0; i < num_hittables; i++) {
            //if (hittables[i].hit(r, min, closest_so_far, temp_rec, temp_material_type)) {
            if (std::visit([&](auto&& arg) { return arg.hit(r, min, closest_so_far, temp_rec, temp_material_type); }, hittables[i])) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                material_type = temp_material_type;
            }
        }
        return hit_anything;
    }

    color get_color(const ray& r, hittable_t* hittables, int max_depth)
    {
        ray cur_ray = r;
        color cur_attenuation { 1.0, 1.0, 1.0 };
        ray scattered;
        color emitted;
        material_t material_type;
        for (auto i = 0; i < max_depth; i++) {
            hit_record rec;
            if (hit_world(cur_ray, real_t { 0.001 }, infinity, rec, hittables, material_type)) {
                emitted = std::visit([&](auto&& arg) { return arg.emitted(rec); }, material_type);
                if (std::visit([&](auto&& arg) { return arg.scatter(cur_ray, rec, cur_attenuation, scattered); }, material_type)) {
                    // On hitting the sphere, the ray gets scattered
                    cur_ray = scattered;
                } else {
                    // Ray did not get scattered or reflected
                    return emitted;
                }
            } else {
                /* If ray doesn't hit anything during iteration linearly blend white and 
                blue color depending on the height of the y coordinate after scaling the 
                ray direction to unit length. While -1.0 < y < 1.0, hit_pt is between 0 
                and 1. This produces a blue to white gradient in the background */
                vec unit_direction = unit_vector(cur_ray.direction());
                auto hit_pt = 0.5 * (unit_direction.y() + 1.0);
                color c = (1.0 - hit_pt) * color { 1.0, 1.0, 1.0 } + hit_pt * color { 0.5, 0.7, 1.0 };
                return emitted + cur_attenuation * c;
            }
        }
        // If not returned within max_depth return black
        return color { 0.0, 0.0, 0.0 };
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
        point look_from = { 13, 2, 3 };
        auto focus_dist = 10;
        auto aperature = 15;

        vec w = unit_vector(look_from - point(0, 0, 0));
        vec u = unit_vector(sycl::cross(vec(0, 1, 0), w));
        vec v = sycl::cross(w, u);

        // Camera arguments
        origin = look_from;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        auto lens_radius = 0.1 / aperature;

        vec rd = lens_radius * random_in_unit_disk();
        vec offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    // Accessor objects
    sycl::accessor<color, 1, sycl::access::mode::write, sycl::access::target::global_buffer> m_frame_ptr;
    sycl::accessor<hittable_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_hitable_ptr;
};

// Render function to call the render kernel
template <int width, int height, int samples, int num_hittables>
void render(sycl::queue queue, color* fb_data, const hittable_t* hittables)
{
    constexpr auto num_pixels = width * height;
    auto const depth = 50;
    auto frame_buf = sycl::buffer<color, 1>(fb_data, sycl::range<1>(num_pixels));
    auto hittables_buf = sycl::buffer<hittable_t, 1>(hittables, sycl::range<1>(num_hittables));
    // Submit command group on device
    queue.submit([&](sycl::handler& cgh) {
        // Get memory access
        auto frame_ptr = frame_buf.get_access<sycl::access::mode::write>(cgh);
        auto hittables_ptr = hittables_buf.get_access<sycl::access::mode::read>(cgh);
        // Setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(constants::TileX, constants::TileY);
        const auto index_space = sycl::nd_range<2>(global, local);
        // Construct kernel functor
        auto render_func = render_kernel<width, height, samples, depth, num_hittables>(frame_ptr, hittables_ptr);
        // Execute kernel
        cgh.parallel_for(index_space, render_func);
    });
}
