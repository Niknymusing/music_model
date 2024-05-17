#include <torch/extension.h>
#include <vector>

__global__ void weighted_sum_update_kernel(
    float* target, const float* source, const float* weights, int num_params, int num_models) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        target[idx] = 0.0;
        for (int i = 0; i < num_models; i++) {
            target[idx] += weights[i] * source[idx + i * num_params];
        }
    }
}

void weighted_sum_update(
    torch::Tensor target,
    std::vector<torch::Tensor> sources,
    torch::Tensor weights) {
    
    int num_params = target.numel();
    int num_models = sources.size();

    // Concatenate all source tensors into one contiguous block
    auto concatenated_sources = torch::cat(sources, 0).contiguous();

    const int threads = 1024;
    const int blocks = (num_params + threads - 1) / threads;

    weighted_sum_update_kernel<<<blocks, threads>>>(
        target.data_ptr<float>(),
        concatenated_sources.data_ptr<float>(),
        weights.data_ptr<float>(),
        num_params,
        num_models);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_sum_update", &weighted_sum_update, "Weighted Sum Update (CUDA)");
}
