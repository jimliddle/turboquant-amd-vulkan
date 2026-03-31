#include "../src/llama-turboquant-codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float max_abs_error(const std::vector<float> & a, const std::vector<float> & b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-turboquant-codec: %s\n", msg);
        std::exit(1);
    }
}

static std::vector<float> make_signal(size_t n) {
    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = float(i);
        result[i] = 0.35f * std::sin(0.13f * x) + 0.6f * std::cos(0.07f * x);
    }
    return result;
}

static void test_roundtrip_f32() {
    const auto src = make_signal(257);
    const llama_turboquant_codec_params params = {
        /*.group_size    =*/ 64,
        /*.residual_bits =*/ 1,
        /*.qjl           =*/ true,
    };

    std::vector<uint8_t> packed;
    llama_turboquant_pack_row(src.data(), GGML_TYPE_F32, src.size(), params, packed);

    std::vector<float> dst(src.size(), 0.0f);
    llama_turboquant_unpack_row(packed.data(), GGML_TYPE_F32, src.size(), params, dst.data());

    require(packed.size() == llama_turboquant_row_size(src.size(), params), "packed row size mismatch for F32");
    require(max_abs_error(src, dst) < 1.5f, "unexpectedly large F32 roundtrip error");
}

static void test_roundtrip_f16() {
    const auto src_f32 = make_signal(96);
    std::vector<ggml_fp16_t> src(src_f32.size());
    std::vector<ggml_fp16_t> dst(src_f32.size());
    ggml_fp32_to_fp16_row(src_f32.data(), src.data(), (int64_t) src.size());

    const llama_turboquant_codec_params params = {
        /*.group_size    =*/ 32,
        /*.residual_bits =*/ 2,
        /*.qjl           =*/ true,
    };

    std::vector<uint8_t> packed;
    llama_turboquant_pack_row(src.data(), GGML_TYPE_F16, src.size(), params, packed);
    llama_turboquant_unpack_row(packed.data(), GGML_TYPE_F16, src.size(), params, dst.data());

    std::vector<float> dst_f32(src.size());
    ggml_fp16_to_fp32_row(dst.data(), dst_f32.data(), (int64_t) dst.size());

    require(packed.size() == llama_turboquant_row_size(src.size(), params), "packed row size mismatch for F16");
    require(max_abs_error(src_f32, dst_f32) < 1.5f, "unexpectedly large F16 roundtrip error");
}

static void test_residual_path_changes_encoding() {
    const auto src = make_signal(64);

    const llama_turboquant_codec_params base = {
        /*.group_size    =*/ 64,
        /*.residual_bits =*/ 0,
        /*.qjl           =*/ false,
    };
    const llama_turboquant_codec_params residual = {
        /*.group_size    =*/ 64,
        /*.residual_bits =*/ 2,
        /*.qjl           =*/ true,
    };

    std::vector<uint8_t> packed_base;
    std::vector<uint8_t> packed_residual;
    std::vector<float> decoded_base(src.size(), 0.0f);
    std::vector<float> decoded_residual(src.size(), 0.0f);

    llama_turboquant_pack_row(src.data(), GGML_TYPE_F32, src.size(), base, packed_base);
    llama_turboquant_pack_row(src.data(), GGML_TYPE_F32, src.size(), residual, packed_residual);
    llama_turboquant_unpack_row(packed_base.data(), GGML_TYPE_F32, src.size(), base, decoded_base.data());
    llama_turboquant_unpack_row(packed_residual.data(), GGML_TYPE_F32, src.size(), residual, decoded_residual.data());

    require(packed_residual.size() > packed_base.size(), "residual path should increase packed size");
    require(max_abs_error(decoded_base, decoded_residual) > 0.0f,
            "residual path should change the decoded reconstruction");
    require(max_abs_error(src, decoded_residual) <= max_abs_error(src, decoded_base),
            "residual path should not reconstruct worse than the base path on this signal");
}

int main() {
    test_roundtrip_f32();
    test_roundtrip_f16();
    test_residual_path_changes_encoding();

    std::printf("test-turboquant-codec: OK\n");
    return 0;
}
