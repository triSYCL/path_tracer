#include "camera.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"
#include <SYCL/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <thread>
#include <vector>


using int_type = uint32_t;
namespace constants {
static constexpr auto TileX = 8;
static constexpr auto TileY = 8;
}

using real_t = double;

struct hit_record {
    double t;
    point3 p;
    vec3 normal;
};

struct xorwow_state_t{
  // xorshift values (160 bits)
  int_type x;
  int_type y;
  int_type z;
  int_type w;
  int_type v;
  // sequence value
  int_type d;
};

inline int_type xorwow(xorwow_state_t* state) {
  int_type t = (state->x ^ (state->x >> 2));
  state->x = state->y;
  state->y = state->z;
  state->z = state->w;
  state->v = (state->v ^ (state->v << 4)) ^ (t ^ (t << 1));
  state->d = state->d + 362437;
  return state->v + state->d;
}

inline int_type mueller_hash(int_type x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x);
  return x;
}

inline xorwow_state_t get_initial_xorwow_state(int_type seed) {
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
inline data_t rand_uniform(rng_t rng, state_t* state) {
  auto a = rng(state) >> 9;
  auto res = data_t{0.0};
  *(reinterpret_cast<int_type*>(&res)) = a | 0x3F800000;
  return res - data_t{1.0};
}

class sphere {
public:
    sphere() : center(vec3{0,0,-1}), radius(0.5) {
        //center = vec3{0,0,-1};
        //radius = 0.5;
    }
    sphere(const vec3& cen, real_t r) : center(cen), radius(r) {
        std::cerr<<"constructor"<<std::endl;
    }

    bool hit(const ray& r, real_t min, real_t max, hit_record& rec) const
    {
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
};

template <int width, int height>
class render_init_kernel {
  template <typename data_t>
  using write_accessor_t = sycl::accessor<data_t, 1, sycl::access::mode::write,
                                          sycl::access::target::global_buffer>;

 public:
  render_init_kernel(write_accessor_t<xorwow_state_t> rand_states_ptr)
      : m_rand_states_ptr(rand_states_ptr) {}

  void operator()(sycl::nd_item<2> item) {
    const auto x_coord = item.get_global_id(0);
    const auto y_coord = item.get_global_id(1);

    const auto pixel_index = y_coord * width + x_coord;

    const auto state = get_initial_xorwow_state(pixel_index);
    m_rand_states_ptr[pixel_index] = state;
  }

 private:
  write_accessor_t<xorwow_state_t> m_rand_states_ptr;
};

template <int width, int height, int samples,int depth,int num_spheres>
class render_kernel {
    //template <typename data_t>
    //using read_accessor_t = sycl::accessor<data_t, 1, sycl::access::mode::read,
    //                                      sycl::access::target::global_buffer>;
public:
    render_kernel(sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> frame_ptr,
        sycl::accessor<sphere, 1, sycl::access::mode::read, sycl::access::target::global_buffer> spheres_ptr,
        sycl::accessor<xorwow_state_t,1,sycl::access::mode::read, sycl::access::target::global_buffer>rand_states_ptr)
        : m_frame_ptr { frame_ptr }
        , m_spheres_ptr { spheres_ptr }
        , m_rand_states_ptr {rand_states_ptr}
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
        auto randf = [&local_rand_state, rng]() {  return rand_uniform(rng, &local_rand_state); };

        //color sampling
        vec3 final_color(0.0,0.0,0.0);
        for (auto i = 0; i < samples; i++) {
            const auto u = (x_coord + randf()) / static_cast<real_t>(width);
            const auto v = (y_coord + randf()) / static_cast<real_t>(height);
            ray r = get_ray(u, v);
            final_color += color(r, m_spheres_ptr.get_pointer(),depth);
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

    vec3 color(const ray& r, sphere* spheres,int max_depth)
    {
        static const auto max_real = std::numeric_limits<real_t>::max();
        ray cur_ray = r;
        vec3 cur_attenuation(1.0, 1.0, 1.0);
        for(auto i = 0; i<max_depth; i++){
            hit_record rec;
            if (hit_world(cur_ray, real_t { 0.001 }, max_real, rec, spheres)) {
                vec3 target = rec.p + rec.normal + random_in_unit_sphere();
                cur_attenuation *= 0.5;
                cur_ray = ray(rec.p, target-rec.p);;
            }
            else{
                vec3 unit_direction = unit_vector(cur_ray.direction());
                auto hit_pt = 0.5 * (unit_direction.y() + 1.0);
                vec3 c = (1.0 - hit_pt) * vec3(1.0, 1.0, 1.0) + hit_pt * vec3(0.5, 0.7, 1.0);
            }
        }
        return vec3(0.0,0.0,0.0);
    }

    ray get_ray(real_t u, real_t v)
    {

        auto aspect_ratio = 16.0 / 9.0;
        auto viewport_height = 2.0;
        auto viewport_width = aspect_ratio * viewport_height;
        auto focal_length = 1.0;

        origin = point3(0, 0, 0);
        horizontal = vec3(viewport_width, 0.0, 0.0);
        vertical = vec3(0.0, viewport_height, 0.0);
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

    /* accessor objects */
    sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::global_buffer> m_frame_ptr;
    sycl::accessor<sphere, 1, sycl::access::mode::read, sycl::access::target::global_buffer> m_spheres_ptr;
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
        const auto local = sycl::range<2>(constants::TileX,constants::TileY);
        const auto index_space = sycl::nd_range<2>(sycl::range<2>(width, height), sycl::range<2>(constants::TileX,constants::TileY));
        // construct kernel functor
        auto render_init_func = render_init_kernel<width, height>(rand_states_ptr);
    // execute kernel
    cgh.parallel_for(index_space, render_init_func);
  });
}

template <int width, int height,int samples, int num_spheres>
void render(sycl::queue queue, vec3* fb_data, const sphere* spheres,xorwow_state_t* rand_states)
{
    constexpr auto num_pixels = width * height;
    auto const depth = 50;
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
        auto render_func = render_kernel<width, height, samples, depth, num_spheres>(frame_ptr, spheres_ptr,rand_states_ptr);
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
            int r = static_cast<int>(255.999 * fb_data[pixel_index].x());
            int g = static_cast<int>(255.999 * fb_data[pixel_index].y());
            int b = static_cast<int>(255.999 * fb_data[pixel_index].z());
            std::cout << r << " " << g << " " << b << "\n";
        }
    }
}

int main()
{
    //frame buffer dimensions
    constexpr auto width = 400;
    constexpr auto height = 240;
    constexpr auto num_pixels = width * height;
    constexpr auto num_spheres = 3;
    constexpr auto samples = 100;
    std::vector<sphere> spheres;
    spheres.push_back(sphere(vec3(0.0, 0.0, -1.0), 0.5)); // (small) center sphere
    spheres.push_back(sphere(vec3(0.0, -100.5, -1.0), 100)); // (large) ground sphere
    spheres.push_back(sphere(vec3(0, 0.0, -0.4), 0.1));

    auto vectors = std::vector<sphere>();

    //sycl queue
    sycl::queue myQueue;

    std::vector<xorwow_state_t> rand_states(num_pixels);
    render_init<width, height>(myQueue, rand_states.data());

    //allocate frame buffer on host
    std::vector<vec3> fb(num_pixels);

    camera cam;

    //sycl render kernel
    render<width, height,samples, num_spheres>(myQueue, fb.data(), spheres.data(),rand_states.data());

    //save image to file
    save_image<width, height>(fb.data());

    return 0;
}