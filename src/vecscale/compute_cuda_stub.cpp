#include "vecscale/compute_cuda.hpp"

namespace vecscale {

bool topk_cosine_similarity_cuda(
    const Matrix&,
    const Matrix&,
    std::size_t,
    SearchResult*,
    std::string* error_message) {
    if (error_message != nullptr) {
        *error_message = "CUDA backend not built. Reconfigure with -DVECSCALE_ENABLE_CUDA=ON.";
    }
    return false;
}

}  // namespace vecscale
