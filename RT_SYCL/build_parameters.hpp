#ifndef BUILD_PARAMETERS_HPP
#define BUILD_PARAMETERS_HPP
namespace buildparams {
#if USE_SINGLE_TASK
constexpr bool SINGLE_TASK = true;
#else
constexpr bool SINGLE_TASK = false;
#endif
}


#endif // BUILD_PARAMETERS_HPP
