#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_turboquant_codec_params {
    uint32_t group_size = 64;
    uint32_t residual_bits = 1;
    bool qjl = true;
};

GGML_API bool ggml_turboquant_codec_supports_type(ggml_type type);

GGML_API size_t ggml_turboquant_row_size(size_t n_el, const ggml_turboquant_codec_params & params);

GGML_API void ggml_turboquant_pack_row(
        const void * src,
        ggml_type type,
        size_t n_el,
        const ggml_turboquant_codec_params & params,
        std::vector<uint8_t> & dst);

GGML_API void ggml_turboquant_unpack_row(
        const uint8_t * src,
        ggml_type type,
        size_t n_el,
        const ggml_turboquant_codec_params & params,
        void * dst);
