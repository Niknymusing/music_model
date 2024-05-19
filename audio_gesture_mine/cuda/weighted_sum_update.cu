#include <torch/extension.h>
#include <vector>

__global__ void weighted_sum_update_kernel(
    float* target, const float* const* sources, const float* weights, const int* param_offsets, int total_params, int num_models) {

    extern __shared__ float s_weights[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_models) {
        s_weights[tid] = weights[tid];
    }
    __syncthreads();

    if (idx < total_params) {
        float temp = 0.0;
        for (int i = 0; i < num_models; ++i) {
            int offset_idx = 0;
            while (idx >= param_offsets[offset_idx + 1]) {
                offset_idx++;
            }

            int local_idx = idx - param_offsets[offset_idx];
            temp += s_weights[i] * sources[i][param_offsets[offset_idx] + local_idx];
        }
        target[idx] = temp;
    }
}

void weighted_sum_update(
    torch::Tensor target,
    std::vector<torch::Tensor> source_params,
    torch::Tensor weights,
    torch::Tensor param_offsets) {

    int total_params = target.numel();
    int num_models = source_params.size();

    std::vector<const float*> source_ptrs(num_models);
    for (int i = 0; i < num_models; ++i) {
        source_ptrs[i] = source_params[i].data_ptr<float>();
    }

    auto source_ptrs_tensor = torch::from_blob(source_ptrs.data(), {num_models}, torch::kFloat32).to(weights.device());

    const int threads = 1024;
    const int blocks = (total_params + threads - 1) / threads;

    weighted_sum_update_kernel<<<blocks, threads, num_models * sizeof(float)>>>(
        target.data_ptr<float>(),
        source_ptrs_tensor.data_ptr<const float*>(),
        weights.data_ptr<float>(),
        param_offsets.data_ptr<int>(),
        total_params,
        num_models);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_sum_update", &weighted_sum_update, "Weighted Sum Update (CUDA)");
}
