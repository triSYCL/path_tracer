#ifndef RT_SYCL_RTWEEKEND_HPP
#define RT_SYCL_RTWEEKEND_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <variant>
#include <vector>

namespace {
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
std::mt19937 generator;
}

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees)
{
    return degrees * pi / 180.0f;
}

inline float random_float()
{
    return distribution(generator);
}

inline float random_float(float min, float max)
{
    // Returns a random real in (min,max).
    return min + (max - min) * random_float();
}

// Common Headers
#include "ray.hpp"
#include "vec.hpp"

#endif
