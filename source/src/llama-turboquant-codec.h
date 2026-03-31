#pragma once

#include "../ggml/src/ggml-turboquant-codec.h"
#include "llama.h"

#include <vector>

using llama_turboquant_codec_params = ggml_turboquant_codec_params;

inline bool llama_turboquant_codec_supports_type(ggml_type type) {
    return ggml_turboquant_codec_supports_type(type);
}

inline size_t llama_turboquant_row_size(size_t n_el, const llama_turboquant_codec_params & params) {
    return ggml_turboquant_row_size(n_el, params);
}

inline void llama_turboquant_pack_row(
        const void * src,
        ggml_type type,
        size_t n_el,
        const llama_turboquant_codec_params & params,
        std::vector<uint8_t> & dst) {
    ggml_turboquant_pack_row(src, type, n_el, params, dst);
}

inline void llama_turboquant_unpack_row(
        const uint8_t * src,
        ggml_type type,
        size_t n_el,
        const llama_turboquant_codec_params & params,
        void * dst) {
    ggml_turboquant_unpack_row(src, type, n_el, params, dst);
}
