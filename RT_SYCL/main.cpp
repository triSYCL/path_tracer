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
#include "ray.hpp"
#include "rtweekend.hpp"
#include "vec3.hpp"

using int_type = std::uint32_t;

namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
}

using real_t = double;

enum class material_t { Lambertian,
    Metal,
    Dielectric };

//struct hit_record {
//    double t;
//    point3 p;
//    vec3 normal;
//};

class hit_record {
public:
    hit_record() = default;

    bool scatter_material(const ray& r_in, vec3& attenuation, ray& scattered)
    {
        switch (material_type) {
        case material_t::Lambertian: {
            vec3 scatter_direction = normal + random_unit_vector();
            scattered = ray(p, scatter_direction);
            attenuation = albedo;
            return true;
        }
        case material_t::Metal: {
            vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
            scattered = ray(p, reflected + fuzz * random_in_unit_sphere());
            attenuation = albedo;
            return (dot(scattered.direction(), normal) > 0);
        }
        case material_t::Dielectric: {
        }
        default:
            return false;
        }
    }
    vec3 center;
    real_t radius;

    double t;
    point3 p;
    vec3 normal;

    // material properties
    material_t material_type;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

struct xorwow_state_t {
    // xorshift values (160 bits)
    int_type x;
    int_type y;
    int_type z;
    int_type w;
    int_type v;
    // sequence value
    int_type d;
};

inline int_type xorwow(xorwow_state_t* state)
{
    int_type t = (state->x ^ (state->x >> 2));
    state->x = state->y;
    state->y = state->z;
    state->z = state->w;
    state->v = (state->v ^ (state->v << 4)) ^ (t ^ (t << 1));
    state->d = state->d + 362437;
    return state->v + state->d;
}

inline int_type mueller_hash(int_type x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

inline xorwow_state_t get_initial_xorwow_state(int_type seed)
{
    xorwow_state_t state;
    state.d = !seed ? 4294967295 : seed;
    for (auto i = 0; i < 8; i++) {
        state.x = mueller_hash(state.d);
        state.y = mueller_hash(state.x);
        state.z = mueller_hash(state.y);
        state.w = mueller_hash(state.z);
        state.v = mueller_hash(state.w);
        state.d = mueller_hash(state.v);
    }
    return state;
}

template <class rng_t, class state_t, typename data_t = double>
inline data_t rand_uniform(rng_t rng, state_t* state)
{
    auto a = rng(state) >> 9;
    auto res = data_t { 0.0 };
    *(reinterpret_cast<int_type*>(&res)) = a | 0x3F800000;
    return res - data_t { 1.0 };
}

template <class derived>
struct crtp {
    derived& underlying()
    {
        return static_cast<derived&>(*this);
    }
    const derived& underlying() const
    {
        return static_cast<const derived&>(*this);
    }
};

template <class geometry>
class hitable : crtp<geometry> {
public:
    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        return this->underlying().hit(r, min, max, rec);
    }
};

class sphere : public hitable<sphere> {
public:
    sphere() = default;
    sphere(const vec3& cen, real_t r)
        : center(cen)
        , radius(r)
    {
        material_type = material_t::Lambertian;
    }
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& color)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , albedo(color)
    {
    }
    sphere(const vec3& cen, real_t r, material_t mat_type, const vec3& mat_color, real_t f)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , albedo(mat_color)
        , fuzz(clamp(f, 0.0, 1.0))
    {
    }

    sphere(const vec3& cen, real_t r, material_t mat_type, real_t ref_idx)
        : center(cen)
        , radius(r)
        , material_type(mat_type)
        , refraction_index(ref_idx)
    {
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
        //$#rec.material_type = material_t::Lambertian;
        rec.material_type = material_type;
        rec.center = center;
        rec.radius = radius;
        if (material_type == material_t::Lambertian) {
            rec.albedo = albedo;
        }
        if (material_type == material_t::Metal) {
            rec.albedo = albedo;
            rec.fuzz = fuzz;
        }
        if (material_type == material_t::Dielectric) {
            rec.refraction_index = refraction_index;
        }
        vec3 oc = r.origin() - center;
        auto a = dot(r.direction(), r.direction());
        auto b = dot(oc, r.direction());
        auto c = dot(oc, oc) - radius * radius;
        auto discriminant = b * b - a * c;
        //std::cout<<"center "<<center<<std::endl;
        if (discriminant > 0) {
            auto temp = (-b - sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
            temp = (-b + sycl::sqrt(discriminant)) / a;
            if (temp < max && temp > min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
        }
        return false;
    }

    // geometry properties
    vec3 center;
    real_t radius;

    //material properties
    material_t material_type;
    vec3 albedo;
    real_t fuzz;
    real_t refraction_index;
};

template <int width, int height>
class render_init_kernel {
    template <typename data_t>
    using write_accessor_t = sycl::accessor<data_t, 1, sycl::access::mode::write,
        sycl::access::target::global_buffer>;

public:
    render_init_kernel(write_accessor_t<xorwow_state_t> rand_states_ptr)
        : m_rand_states_ptr(rand_states_ptr)
    {
    }

    void operator()(sycl::nd_item<2> item)
    {
        const auto x_coord = item.get_global_id(0);
        const auto y_coord = item.get_global_id(1);

        const auto pixel_index = y_coord * width + x_coord;

        const auto state = get_initial_xorwow_state(pixel_index);
        m_rand_states_ptr[pixel_index] = state;
    }

private:
    write_accessor_t<xorwow_state_t> m_rand_states_ptr;
};

template <int width, int height, int samples, int depth, int num_spheres, class hitable>
class render_kernel {
    //template <typename data_t>
    //using read_accessor_t = sycl::accessor<data_t, 1, sycl::access::mode::read,
    //                                      sycl::access::target::global_buffer>;
public:
    render_kernel(sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> frame_ptr,
        sycl::accessor<hitable, 1, sycl::access::mode::read, sycl::access::target::global_buffer> hitable_ptr,
        sycl::accessor<xorwow_state_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> rand_states_ptr)
        : m_frame_ptr { frame_ptr }
        , m_hitable_ptr { hitable_ptr }
        , m_rand_states_ptr { rand_states_ptr }
    {
    }

    void operator()(sycl::nd_item<2> item)
    {
        // get our Ids
        const auto x_coord = item.get_global_id(0);
        const auto y_coord = item.get_global_id(1);
        // map the 2D indices to a single linear, 1D index
        const auto pixel_index = y_coord * width + x_coord;

        // initialize local (for the current thread) random state
        auto local_rand_state = m_rand_states_ptr[pixel_index];
        // create a 'rng' function object using a lambda
        auto rng = [](xorwow_state_t* state) { return xorwow(state); };
        // capture the rand generator state -> return a uniform value
        auto randf = [&local_rand_state, rng]() { return rand_uniform(rng, &local_rand_state); };

        //color sampling
        vec3 final_color(0.0, 0.0, 0.0);
        for (auto i = 0; i < samples; i++) {
            const auto u = (x_coord + random_double()) / static_cast<real_t>(width);
            const auto v = (y_coord + random_double()) / static_cast<real_t>(height);
            ray r = get_ray(u, v);
            final_color += color(r, m_hitable_ptr.get_pointer(), depth);
        }
        final_color /= static_cast<real_t>(samples);

        // write final color to the frame buffer global memory
        m_frame_ptr[pixel_index] = final_color;
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;

    bool hit_world(const ray& r, real_t min, real_t max, hit_record& rec, sphere* spheres)
    {
        auto temp_rec = hit_record {};
        auto hit_anything = false;
        auto closest_so_far = max;
        for (auto i = 0; i < num_spheres; i++) {
            if (spheres[i].hit(r, min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    vec3 color(const ray& r, sphere* spheres, int max_depth)
    {
        ray cur_ray = r;
        vec3 cur_attenuation(1.0, 1.0, 1.0);
        ray scattered;
        for (auto i = 0; i < max_depth; i++) {
            hit_record rec;
            if (hit_world(cur_ray, real_t { 0.001 }, infinity, rec, spheres)) {
                if (rec.scatter_material(cur_ray, cur_attenuation, scattered)) {
                    cur_ray = scattered;
                    ;
                } else {
                    return vec3(0.0, 0.0, 0.0);
                }
            } else {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                auto hit_pt = 0.5 * (unit_direction.y() + 1.0);
                vec3 c = (1.0 - hit_pt) * vec3(1.0, 1.0, 1.0) + hit_pt * vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        return vec3(0.0, 0.0, 0.0);
    }

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
        vec3 u = unit_vector(cross(vec3(0, 1, 0), w));
        vec3 v = cross(w, u);

        origin = look_from;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        auto lens_radius = 0.1 / 2;

        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    /* accessor objects */
    sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> m_frame_ptr;
    sycl::accessor<hitable, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_hitable_ptr;
    sycl::accessor<xorwow_state_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_rand_states_ptr;
};

template <int width, int height>
void render_init(sycl::queue& queue, xorwow_state_t* rand_states)
{
    constexpr auto num_pixels = width * height;
    // allocate memory on device
    auto rand_states_buf = sycl::buffer<xorwow_state_t, 1>(rand_states, sycl::range<1>(num_pixels));
    // submit command group on device
    queue.submit([&](sycl::handler& cgh) {
        auto rand_states_ptr = rand_states_buf.get_access<sycl::access::mode::write>(cgh);
        // setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(constants::TileX, constants::TileY);
        const auto index_space = sycl::nd_range<2>(sycl::range<2>(width, height), sycl::range<2>(constants::TileX, constants::TileY));
        // construct kernel functor
        auto render_init_func = render_init_kernel<width, height>(rand_states_ptr);
        // execute kernel
        cgh.parallel_for(index_space, render_init_func);
    });
}

template <int width, int height, int samples, int num_spheres, class hitable>
void render(sycl::queue queue, vec3* fb_data, const hitable* spheres, xorwow_state_t* rand_states)
{
    constexpr auto num_pixels = width * height;
    auto const depth = 5;
    auto frame_buf = sycl::buffer<vec3, 1>(fb_data, sycl::range<1>(num_pixels));
    auto sphere_buf = sycl::buffer<sphere, 1>(spheres, sycl::range<1>(num_spheres));
    auto rand_states_buf = sycl::buffer<xorwow_state_t, 1>(rand_states, sycl::range<1>(num_pixels));
    // submit command group on device
    queue.submit([&](sycl::handler& cgh) {
        // get memory access
        auto frame_ptr = frame_buf.get_access<sycl::access::mode::write>(cgh);
        auto spheres_ptr = sphere_buf.get_access<sycl::access::mode::read>(cgh);
        auto rand_states_ptr = rand_states_buf.get_access<sycl::access::mode::read>(cgh);
        // setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(constants::TileX, constants::TileY);
        const auto index_space = sycl::nd_range<2>(global, local);
        // construct kernel functor
        auto render_func = render_kernel<width, height, samples, depth, num_spheres, hitable>(frame_ptr, spheres_ptr, rand_states_ptr);
        // execute kernel
        cgh.parallel_for(index_space, render_func);
    });
}

template <int width, int height>
void save_image(vec3* fb_data)
{
    std::cout << "P3\n"
              << width << " " << height << "\n255\n";
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            auto pixel_index = y * width + x;
            int r = static_cast<int>(256 * clamp(sycl::sqrt(fb_data[pixel_index].x()), 0.0, 0.999));
            int g = static_cast<int>(256 * clamp(sycl::sqrt(fb_data[pixel_index].y()), 0.0, 0.999));
            int b = static_cast<int>(256 * clamp(sycl::sqrt(fb_data[pixel_index].z()), 0.0, 0.999));
            std::cout << r << " " << g << " " << b << "\n";
        }
    }
}

int main()
{
    //frame buffer dimensions
    constexpr auto width = 800;
    constexpr auto height = 480;
    constexpr auto num_pixels = width * height;
    constexpr auto num_spheres = 459;
    constexpr auto samples = 100;
    std::vector<sphere> spheres;

    spheres.push_back(sphere(vec3(0, -1000, 0), 1000, material_t::Lambertian, color(0.2, 0.2, 0.2)));
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    spheres.push_back(sphere(center, 0.2, material_t::Lambertian, albedo));
//Undefined                    count++;
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    spheres.push_back(sphere(center, 0.2, material_t::Metal, albedo, fuzz));
//Undefined                    count++;
                }
            }
        }
    }
    spheres.push_back(sphere(point3(4, 1, 0), 1, material_t::Metal, color(0.7, 0.6, 0.5), 0.0));
    spheres.push_back(sphere(point3(-4, 1, 0), 1, material_t::Lambertian, color(0.4, 0.2, 0.1)));
//Undefined    std::cerr << check * 10000 << std::endl;

    // spheres.push_back(sphere(vec3(0.0, 0.0, -1.0), 0.5,material_t::Lambertian,color(0.1,0.2,0.5))); // (small) center sphere
    // spheres.push_back(sphere(vec3(0.0, -100.5, -1.0), 100,material_t::Lambertian,color(0.2,0.2,0.2))); // (large) ground sphere
    // spheres.push_back(sphere(vec3(-1.0, -0.05, -3), 0.5,material_t::Metal,color(0.8,0.8,0.8),0.1));
    // spheres.push_back(sphere(vec3(1.0, -0.1, 3), 0.5,material_t::Metal,color(0.8,0.6,0.2),0.5));

    //sycl queue
    sycl::queue myQueue;

    std::vector<xorwow_state_t> rand_states(num_pixels);
    render_init<width, height>(myQueue, rand_states.data());

    //allocate frame buffer on host
    std::vector<vec3> fb(num_pixels);

    camera cam;

    //sycl render kernel
    render<width, height, samples, num_spheres, class sphere>(myQueue, fb.data(), spheres.data(), rand_states.data());

    //save image to file
    save_image<width, height>(fb.data());

    return 0;
}
