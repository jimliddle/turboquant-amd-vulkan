#version 450

#include "types.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform parameter {
    uint n_row_el;
    uint kv_size;
    uint packed_row_size;
    uint n_rows;
    uint group_size;
    uint residual_bits;
    uint qjl;
    uint transposed;
} p;

layout(binding = 0) readonly buffer PackedRows {
    uint packed_rows[];
};

layout(binding = 1) readonly buffer RowIndices {
    uint row_indices[];
};

layout(binding = 2) writeonly buffer Dst {
    D_TYPE data_d[];
};

uint tq_read_byte(uint byte_offset) {
    const uint word = packed_rows[byte_offset >> 2];
    const uint shift = (byte_offset & 3u) << 3u;
    return (word >> shift) & 0xffu;
}

uint tq_read_u16(uint byte_offset) {
    return tq_read_byte(byte_offset) | (tq_read_byte(byte_offset + 1u) << 8u);
}

uint tq_read_bits(uint bit_offset, uint n_bits) {
    uint value = 0u;
    for (uint b = 0u; b < n_bits; ++b) {
        const uint bit = bit_offset + b;
        const uint byte_offset = bit >> 3u;
        const uint shift = bit & 7u;
        value |= ((tq_read_byte(byte_offset) >> shift) & 1u) << b;
    }
    return value;
}

float tq_read_f16(uint byte_offset) {
    return unpackHalf2x16(tq_read_u16(byte_offset)).x;
}

void tq_store(uint index, float value) {
#if defined(DATA_D_F16)
    data_d[index] = float16_t(value);
#elif defined(DATA_D_F32)
    data_d[index] = value;
#elif defined(DATA_D_BF16)
    data_d[index] = uint16_t(fp32_to_bf16(value));
#else
#error unsupported TurboQuant destination type
#endif
}

void main() {
    const uint el = gl_GlobalInvocationID.x;
    const uint row_slot = gl_GlobalInvocationID.y;

    if (el >= p.n_row_el || row_slot >= p.n_rows) {
        return;
    }

    const uint row_index = row_indices[row_slot];
    const uint group_index = el / p.group_size;
    const uint group_start = group_index * p.group_size;
    const uint group_el = min(p.group_size, p.n_row_el - group_start);
    const uint local_index = el - group_start;

    const uint full_sign_bytes = (p.group_size + 7u) >> 3u;
    const uint full_residual_bytes = (p.group_size * p.residual_bits + 7u) >> 3u;
    const uint full_group_bytes = 2u + full_sign_bytes + ((p.qjl != 0u && p.residual_bits > 0u) ? (2u + full_residual_bytes) : 0u);

    const uint sign_bytes = (group_el + 7u) >> 3u;
    const uint row_base = row_index * p.packed_row_size + group_index * full_group_bytes;
    const uint sign_base = row_base + 2u;

    const float primary_scale = tq_read_f16(row_base);
    const bool positive = tq_read_bits(sign_base * 8u + local_index, 1u) != 0u;
    float value = positive ? primary_scale : -primary_scale;

    if (p.qjl != 0u && p.residual_bits > 0u) {
        const uint residual_scale_base = sign_base + sign_bytes;
        const uint residual_data_base = residual_scale_base + 2u;
        const float residual_scale = tq_read_f16(residual_scale_base);
        const uint levels = 1u << p.residual_bits;
        const uint q = tq_read_bits(residual_data_base * 8u + local_index * p.residual_bits, p.residual_bits);
        const float deq = levels <= 1u ? 0.0f : (-1.0f + 2.0f * float(q) / float(levels - 1u));
        value += residual_scale * deq;
    }

    const uint dst_index = p.transposed != 0u
            ? row_index + el * p.kv_size
            : row_index * p.n_row_el + el;

    tq_store(dst_index, value);
}
