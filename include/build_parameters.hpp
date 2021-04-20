#ifndef BUILD_PARAMETERS_HPP
#define BUILD_PARAMETERS_HPP
namespace buildparams {

#ifdef USE_SINGLE_TASK
constexpr bool use_single_task = true;
#else
constexpr bool use_single_task = false;
#endif

#ifdef USE_SYCL_COMPILER
constexpr bool use_sycl_compiler = USE_SYCL_COMPILER;
#else
constexpr bool use_sycl_compiler = false;
#endif

constexpr int output_width = OUTPUT_WIDTH;
constexpr int output_height = OUTPUT_HEIGHT;
constexpr int samples = SAMPLES;
} // namespace buildparams

#endif // BUILD_PARAMETERS_HPP
