#ifndef TASK_CONTEXT_HPP
#define TASK_CONTEXT_HPP

#include "localrandom.hpp"
#include "sycl.hpp"

namespace raytracer::visitor {
/**
 @brief Used as a poorman's cooperative ersatz of device global variable
        The task context is (manually) passed through all the visitors
 */
struct task_context {
  raytracer::random::LocalPseudoRNG rng;
  // See image_texture in texture.hpp for more details
  sycl::global_ptr<uint8_t> texture_data;
};
} // namespace raytracer::visitor
#endif