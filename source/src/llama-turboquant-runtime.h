#pragma once

#include "llama-turboquant-codec.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

struct llama_turboquant_runtime_buffer {
    ggml_tensor * tensor = nullptr;
    ggml_type type = GGML_TYPE_F16;
    uint32_t n_row_el = 0;
    uint32_t kv_size = 0;
    bool transposed = false;
};

struct llama_turboquant_runtime_row_view {
    const uint8_t * packed = nullptr;
    size_t packed_size = 0;
    uint32_t row_index = 0;
};

struct llama_turboquant_runtime_request {
    llama_turboquant_runtime runtime = LLAMA_TURBOQUANT_RUNTIME_AUTO;
    llama_turboquant_codec_params codec_params;
    llama_turboquant_runtime_buffer buffer;
    std::vector<llama_turboquant_runtime_row_view> rows;
};

bool llama_turboquant_runtime_materialize(const llama_turboquant_runtime_request & request, std::string & reason);
bool llama_turboquant_runtime_sync_shadow(const llama_turboquant_runtime_request & request, std::string & reason);
