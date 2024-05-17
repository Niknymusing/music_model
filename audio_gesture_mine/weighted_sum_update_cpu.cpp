#include <torch/extension.h>
#include <vector>
#include <algorithm>

void weighted_sum_update_cpu(
    torch::Tensor target,
    std::vector<torch::Tensor> sources,
    torch::Tensor weights) {
    
    // Ensure all tensors are contiguous
    target = target.contiguous();
    weights = weights.contiguous();
    for (auto& source : sources) {
        source = source.contiguous();
    }

    // Get pointers to data
    float* target_data = target.data_ptr<float>();
    const float* weights_data = weights.data_ptr<float>();
    std::vector<const float*> sources_data;
    for (const auto& source : sources) {
        sources_data.push_back(source.data_ptr<float>());
    }

    // Get sizes
    int64_t num_params = target.numel();
    int64_t num_models = sources.size();

    // Perform the weighted sum update
    std::fill(target_data, target_data + num_params, 0.0);
    for (int64_t i = 0; i < num_params; ++i) {
        for (int64_t j = 0; j < num_models; ++j) {
            target_data[i] += weights_data[j] * sources_data[j][i];
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_sum_update_cpu", &weighted_sum_update_cpu, "Weighted Sum Update (CPU)");
}
