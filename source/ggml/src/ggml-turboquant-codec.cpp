#include "ggml-turboquant-codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

static void ggml_turboquant_row_to_f32(
        const void * src,
        ggml_type type,
        size_t n_el,
        std::vector<float> & dst) {
    dst.resize(n_el);

    switch (type) {
        case GGML_TYPE_F32:
            std::memcpy(dst.data(), src, n_el * sizeof(float));
            return;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(src), dst.data(), (int64_t) n_el);
            return;
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(src), dst.data(), (int64_t) n_el);
            return;
        default:
            throw std::runtime_error("TurboQuant codec only supports F32/F16/BF16 rows");
    }
}

static void ggml_turboquant_row_from_f32(
        const std::vector<float> & src,
        ggml_type type,
        void * dst) {
    switch (type) {
        case GGML_TYPE_F32:
            std::memcpy(dst, src.data(), src.size() * sizeof(float));
            return;
        case GGML_TYPE_F16:
            ggml_fp32_to_fp16_row(src.data(), static_cast<ggml_fp16_t *>(dst), (int64_t) src.size());
            return;
        case GGML_TYPE_BF16:
            ggml_fp32_to_bf16_row(src.data(), static_cast<ggml_bf16_t *>(dst), (int64_t) src.size());
            return;
        default:
            throw std::runtime_error("TurboQuant codec only supports F32/F16/BF16 rows");
    }
}

static void ggml_turboquant_write_bits(uint8_t * dst, size_t bit_offset, uint32_t value, uint32_t n_bits) {
    for (uint32_t b = 0; b < n_bits; ++b) {
        const size_t bit = bit_offset + b;
        const size_t byte_index = bit / 8;
        const uint8_t bit_mask = (uint8_t) (1u << (bit % 8));

        if ((value >> b) & 1u) {
            dst[byte_index] |= bit_mask;
        } else {
            dst[byte_index] &= (uint8_t) ~bit_mask;
        }
    }
}

static uint32_t ggml_turboquant_read_bits(const uint8_t * src, size_t bit_offset, uint32_t n_bits) {
    uint32_t value = 0;

    for (uint32_t b = 0; b < n_bits; ++b) {
        const size_t bit = bit_offset + b;
        const size_t byte_index = bit / 8;
        const uint8_t bit_mask = (uint8_t) (1u << (bit % 8));

        if (src[byte_index] & bit_mask) {
            value |= 1u << b;
        }
    }

    return value;
}

bool ggml_turboquant_codec_supports_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

size_t ggml_turboquant_row_size(size_t n_el, const ggml_turboquant_codec_params & params) {
    if (params.group_size == 0) {
        throw std::runtime_error("TurboQuant group size must be greater than zero");
    }

    size_t total = 0;
    for (size_t i = 0; i < n_el; i += params.group_size) {
        const size_t group_el = std::min<size_t>(params.group_size, n_el - i);
        total += sizeof(ggml_fp16_t);
        total += (group_el + 7) / 8;
        if (params.qjl && params.residual_bits > 0) {
            total += sizeof(ggml_fp16_t);
            total += (group_el * params.residual_bits + 7) / 8;
        }
    }

    return total;
}

void ggml_turboquant_pack_row(
        const void * src,
        ggml_type type,
        size_t n_el,
        const ggml_turboquant_codec_params & params,
        std::vector<uint8_t> & dst) {
    if (!ggml_turboquant_codec_supports_type(type)) {
        throw std::runtime_error("unsupported tensor type for TurboQuant row packing");
    }
    if (params.group_size == 0) {
        throw std::runtime_error("TurboQuant group size must be greater than zero");
    }

    std::vector<float> values;
    std::vector<float> reconstructed(n_el, 0.0f);
    ggml_turboquant_row_to_f32(src, type, n_el, values);

    dst.assign(ggml_turboquant_row_size(n_el, params), 0);
    size_t offset = 0;

    for (size_t group_start = 0; group_start < n_el; group_start += params.group_size) {
        const size_t group_el = std::min<size_t>(params.group_size, n_el - group_start);

        float primary_scale = 0.0f;
        for (size_t i = 0; i < group_el; ++i) {
            primary_scale = std::max(primary_scale, std::fabs(values[group_start + i]));
        }

        const ggml_fp16_t primary_scale_f16 = ggml_fp32_to_fp16(primary_scale);
        std::memcpy(dst.data() + offset, &primary_scale_f16, sizeof(primary_scale_f16));
        offset += sizeof(primary_scale_f16);

        uint8_t * sign_bytes = dst.data() + offset;
        const size_t sign_bytes_size = (group_el + 7) / 8;
        std::memset(sign_bytes, 0, sign_bytes_size);

        for (size_t i = 0; i < group_el; ++i) {
            const float value = values[group_start + i];
            const bool positive = std::signbit(value) == 0;
            ggml_turboquant_write_bits(sign_bytes, i, positive ? 1u : 0u, 1);
            reconstructed[group_start + i] = positive ? primary_scale : -primary_scale;
        }
        offset += sign_bytes_size;

        if (params.qjl && params.residual_bits > 0) {
            float residual_scale = 0.0f;
            for (size_t i = 0; i < group_el; ++i) {
                residual_scale = std::max(residual_scale, std::fabs(values[group_start + i] - reconstructed[group_start + i]));
            }

            const ggml_fp16_t residual_scale_f16 = ggml_fp32_to_fp16(residual_scale);
            std::memcpy(dst.data() + offset, &residual_scale_f16, sizeof(residual_scale_f16));
            offset += sizeof(residual_scale_f16);

            uint8_t * residual_bytes = dst.data() + offset;
            const size_t residual_bytes_size = (group_el * params.residual_bits + 7) / 8;
            std::memset(residual_bytes, 0, residual_bytes_size);

            const uint32_t levels = 1u << params.residual_bits;
            for (size_t i = 0; i < group_el; ++i) {
                const float residual = residual_scale > 0.0f ? (values[group_start + i] - reconstructed[group_start + i]) / residual_scale : 0.0f;
                const float clamped = std::max(-1.0f, std::min(1.0f, residual));
                const float qf = levels <= 1 ? 0.0f : ((clamped + 1.0f) * 0.5f * float(levels - 1));
                const uint32_t q = levels <= 1 ? 0u : (uint32_t) std::lround(qf);
                ggml_turboquant_write_bits(residual_bytes, i * params.residual_bits, q, params.residual_bits);
            }

            offset += residual_bytes_size;
        }
    }
}

void ggml_turboquant_unpack_row(
        const uint8_t * src,
        ggml_type type,
        size_t n_el,
        const ggml_turboquant_codec_params & params,
        void * dst) {
    if (!ggml_turboquant_codec_supports_type(type)) {
        throw std::runtime_error("unsupported tensor type for TurboQuant row unpacking");
    }
    if (params.group_size == 0) {
        throw std::runtime_error("TurboQuant group size must be greater than zero");
    }

    std::vector<float> values(n_el, 0.0f);
    size_t offset = 0;

    for (size_t group_start = 0; group_start < n_el; group_start += params.group_size) {
        const size_t group_el = std::min<size_t>(params.group_size, n_el - group_start);

        ggml_fp16_t primary_scale_f16;
        std::memcpy(&primary_scale_f16, src + offset, sizeof(primary_scale_f16));
        const float primary_scale = ggml_fp16_to_fp32(primary_scale_f16);
        offset += sizeof(primary_scale_f16);

        const uint8_t * sign_bytes = src + offset;
        offset += (group_el + 7) / 8;

        for (size_t i = 0; i < group_el; ++i) {
            const bool positive = ggml_turboquant_read_bits(sign_bytes, i, 1) != 0;
            values[group_start + i] = positive ? primary_scale : -primary_scale;
        }

        if (params.qjl && params.residual_bits > 0) {
            ggml_fp16_t residual_scale_f16;
            std::memcpy(&residual_scale_f16, src + offset, sizeof(residual_scale_f16));
            const float residual_scale = ggml_fp16_to_fp32(residual_scale_f16);
            offset += sizeof(residual_scale_f16);

            const uint8_t * residual_bytes = src + offset;
            offset += (group_el * params.residual_bits + 7) / 8;

            const uint32_t levels = 1u << params.residual_bits;
            for (size_t i = 0; i < group_el; ++i) {
                const uint32_t q = ggml_turboquant_read_bits(residual_bytes, i * params.residual_bits, params.residual_bits);
                const float deq = levels <= 1 ? 0.0f : (-1.0f + 2.0f * float(q) / float(levels - 1));
                values[group_start + i] += residual_scale * deq;
            }
        }
    }

    ggml_turboquant_row_from_f32(values, type, dst);
}
