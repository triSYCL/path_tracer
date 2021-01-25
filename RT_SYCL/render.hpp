#include "box.hpp"
#include "camera.hpp"
#include "constant_medium.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rectangle.hpp"
#include "rtweekend.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "triangle.hpp"
#include "vec.hpp"
#include "visit.hpp"
#include "sycl.hpp"

using hittable_t = std::variant<sphere, xy_rect, triangle, box, constant_medium>;
namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
}

template <int width, int height, int samples, int depth>
class render_kernel {
public:
    render_kernel(sycl::accessor<color, 1, sycl::access::mode::write, sycl::access::target::global_buffer> frame_ptr,
        sycl::accessor<hittable_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> hitable_ptr, int num_hittables,
        camera& cam)
        : m_frame_ptr { frame_ptr }
        , m_hitable_ptr { hitable_ptr }
        , num_hittables { num_hittables }
        , m_camera { cam }
    {
    }

#ifdef USE_SINGLE_TASK
    void operator()()
#else
    void operator()(sycl::nd_item<2> item)
#endif
    {
#ifdef USE_SINGLE_TASK
        for (int x_coord = 0; x_coord != width; ++x_coord)
            for (int y_coord = 0; y_coord != height; ++y_coord) {
#else
        // Get our Ids
        const auto x_coord = item.get_global_id(0);
        const auto y_coord = item.get_global_id(1);
#endif
        // map the 2D indices to a single linear, 1D index
        const auto pixel_index = y_coord * width + x_coord;

        // Color sampling for antialiasing
        color final_color(0.0f, 0.0f, 0.0f);
        for (auto i = 0; i < samples; i++) {
            const auto u = (x_coord + random_float()) / width;
            const auto v = (y_coord + random_float()) / height;
            // u and v are points on the viewport
            ray r = m_camera.get_ray(u, v);
            final_color += get_color(r, m_hitable_ptr.get_pointer(), depth);
        }
        final_color /= static_cast<real_t>(samples);

        // Write final color to the frame buffer global memory
        m_frame_ptr[pixel_index] = final_color;
#ifdef USE_SINGLE_TASK
        }
#endif
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
            if (dev_visit([&](auto&& arg) { return arg.hit(r, min, closest_so_far, temp_rec, temp_material_type); }, hittables[i])) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                material_type = temp_material_type;
            }
        }
        return hit_anything;
    }

    /// Compute the color of the ray
    color get_color(const ray& r, hittable_t* hittables, int max_depth)
    {
        ray cur_ray = r;
        color cur_attenuation { 1.0f, 1.0f, 1.0f };
        ray scattered;
        color emitted;
        material_t material_type;
        for (auto i = 0; i < max_depth; i++) {
            hit_record rec;
            if (hit_world(cur_ray, real_t { 0.001f }, infinity, rec, hittables, material_type)) {
                emitted = dev_visit([&](auto&& arg) { return arg.emitted(rec); }, material_type);
                if (dev_visit([&](auto&& arg) { return arg.scatter(cur_ray, rec, cur_attenuation, scattered); }, material_type)) {
                    // On hitting the object, the ray gets scattered
                    cur_ray = scattered;
                } else {
                    // Ray did not get scattered or reflected
                    return emitted;
                }
            } else {
                /**
                 * If ray doesn't hit anything during iteration linearly blend white and 
                 * blue color depending on the height of the y coordinate after scaling the 
                 * ray direction to unit length. While -1.0f < y < 1.0f, hit_pt is between 0 
                 * and 1. This produces a blue to white gradient in the background 
                */
                vec unit_direction = unit_vector(cur_ray.direction());
                auto hit_pt = 0.5f * (unit_direction.y() + 1.0f);
                color c = (1.0f - hit_pt) * color { 1.0f, 1.0f, 1.0f } + hit_pt * color { 0.5f, 0.7f, 1.0f };
                return emitted + cur_attenuation * c;
            }
        }
        // If not returned within max_depth return black
        return color { 0.0f, 0.0f, 0.0f };
    }

    // Accessor objects
    sycl::accessor<color, 1, sycl::access::mode::write, sycl::access::target::global_buffer> m_frame_ptr;
    sycl::accessor<hittable_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_hitable_ptr;
    int num_hittables;
    camera m_camera;
};

// Render function to call the render kernel
template <int width, int height, int samples>
void render(sycl::queue queue, color* fb_data, const hittable_t* hittables, int num_hittables, camera& cam)
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
        // Construct kernel functor
        render_kernel<width, height, samples, depth> render_func =
          { frame_ptr, hittables_ptr, num_hittables, cam };
        // Execute kernel
#ifdef USE_SINGLE_TASK
        // Use a single task iterating on all pixels
        cgh.single_task(render_func);
#else
        // Setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(constants::TileX, constants::TileY);
        const auto index_space = sycl::nd_range<2>(global, local);
        // Launch 1 work-item per pixel in parallel
        cgh.parallel_for(index_space, render_func);
#endif
    });
}
