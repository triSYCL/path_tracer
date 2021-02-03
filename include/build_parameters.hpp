#ifndef BUILD_PARAMETERS_HPP
#define BUILD_PARAMETERS_HPP
namespace buildparams {
#if USE_SINGLE_TASK
constexpr bool use_single_task = true;
#else
constexpr bool use_single_task = false;
#endif
}


#endif // BUILD_PARAMETERS_HPP
