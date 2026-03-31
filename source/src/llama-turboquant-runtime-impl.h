#pragma once

#include "ggml-backend.h"
#include "llama-turboquant-runtime.h"

#include <cstring>
#include <string>
#include <vector>

struct llama_turboquant_runtime_context {
    llama_turboquant_runtime runtime = LLAMA_TURBOQUANT_RUNTIME_AUTO;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
};

using llama_turboquant_backend_materialize_fn = bool (*)(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t n_row_el,
        uint32_t kv_size,
        bool transposed,
        uint32_t group_size,
        uint32_t residual_bits,
        bool qjl,
        const uint8_t * packed_rows,
        size_t packed_row_size,
        const uint32_t * row_indices,
        size_t n_rows);

using llama_turboquant_backend_sync_fn = bool (*)(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        uint32_t kv_size,
        uint32_t n_row_el,
        bool transposed,
        uint32_t group_size,
        uint32_t residual_bits,
        bool qjl,
        const uint8_t * packed_rows,
        size_t packed_row_size,
        const uint32_t * row_indices,
        size_t n_rows);

inline bool llama_turboquant_runtime_linearize_rows(
        const llama_turboquant_runtime_request & request,
        std::vector<uint8_t> & packed_rows,
        std::vector<uint32_t> & row_indices,
        size_t & packed_row_size,
        std::string & reason) {
    if (request.rows.empty()) {
        packed_rows.clear();
        row_indices.clear();
        packed_row_size = 0;
        reason.clear();
        return true;
    }

    packed_row_size = request.rows[0].packed_size;
    if (packed_row_size == 0) {
        reason = "TurboQuant runtime row payload is empty";
        return false;
    }

    packed_rows.resize(request.rows.size() * packed_row_size);
    row_indices.resize(request.rows.size());

    for (size_t i = 0; i < request.rows.size(); ++i) {
        const auto & row = request.rows[i];
        if (row.packed == nullptr) {
            reason = "TurboQuant runtime row pointer is null";
            return false;
        }
        if (row.packed_size != packed_row_size) {
            reason = "TurboQuant runtime rows have inconsistent packed sizes";
            return false;
        }

        std::memcpy(packed_rows.data() + i * packed_row_size, row.packed, packed_row_size);
        row_indices[i] = row.row_index;
    }

    reason.clear();
    return true;
}

inline bool llama_turboquant_runtime_validate_rows(
        const llama_turboquant_runtime_request & request,
        size_t & packed_row_size,
        std::string & reason) {
    if (request.rows.empty()) {
        packed_row_size = 0;
        reason.clear();
        return true;
    }

    packed_row_size = request.rows[0].packed_size;
    if (packed_row_size == 0) {
        reason = "TurboQuant runtime row payload is empty";
        return false;
    }

    for (const auto & row : request.rows) {
        if (row.packed == nullptr) {
            reason = "TurboQuant runtime row pointer is null";
            return false;
        }
        if (row.packed_size != packed_row_size) {
            reason = "TurboQuant runtime rows have inconsistent packed sizes";
            return false;
        }
    }

    reason.clear();
    return true;
}

bool llama_turboquant_runtime_materialize_native_hip(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason);

bool llama_turboquant_runtime_materialize_native_vulkan(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason);

bool llama_turboquant_runtime_sync_native_vulkan(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason);
