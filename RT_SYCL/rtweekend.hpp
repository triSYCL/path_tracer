#ifndef RT_SYCL_RTWEEKEND_HPP
#define RT_SYCL_RTWEEKEND_HPP

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

namespace
{
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 generator;
}

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

inline double random_double()
{
    return distribution(generator);
}

inline double random_double(double min, double max)
{
    // Returns a random real in (min,max).
    return min + (max - min) * random_double();
}

// Common Headers
#include "ray.hpp"
#include "vec3.hpp"

#endif
