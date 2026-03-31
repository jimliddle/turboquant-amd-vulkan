#include "llama-turboquant-runtime-impl.h"

bool llama_turboquant_runtime_materialize_native_vulkan(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason) {
#if defined(GGML_USE_VULKAN)
    if (ctx.device == nullptr || ctx.backend == nullptr) {
        reason = "TurboQuant Vulkan native materialization requires an initialized Vulkan backend context";
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ctx.device);
    if (reg == nullptr) {
        reason = "TurboQuant Vulkan native materialization could not resolve the Vulkan backend registry";
        return false;
    }

    auto * fn = (llama_turboquant_backend_materialize_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_turboquant_materialize");
    if (fn == nullptr) {
        reason = "TurboQuant Vulkan native materialization entry point is not exported by the Vulkan backend";
        return false;
    }

    static thread_local std::vector<uint32_t> row_indices;
    size_t packed_row_size = 0;

    if (!llama_turboquant_runtime_validate_rows(request, packed_row_size, reason)) {
        return false;
    }

    size_t run_begin = 0;
    while (run_begin < request.rows.size()) {
        size_t run_end = run_begin + 1;
        while (run_end < request.rows.size()) {
            const auto & prev = request.rows[run_end - 1];
            const auto & curr = request.rows[run_end];
            if (curr.row_index != prev.row_index + 1) {
                break;
            }
            if (curr.packed != prev.packed + packed_row_size) {
                break;
            }
            ++run_end;
        }

        const size_t run_size = run_end - run_begin;
        row_indices.resize(run_size);
        for (size_t i = 0; i < run_size; ++i) {
            row_indices[i] = request.rows[run_begin + i].row_index;
        }

        if (!fn(
                ctx.backend,
                request.buffer.tensor,
                request.buffer.type,
                request.buffer.n_row_el,
                request.buffer.kv_size,
                request.buffer.transposed,
                request.codec_params.group_size,
                request.codec_params.residual_bits,
                request.codec_params.qjl,
                request.rows[run_begin].packed,
                packed_row_size,
                row_indices.data(),
                row_indices.size())) {
            reason = "TurboQuant Vulkan native materialization entry point reported failure";
            return false;
        }

        run_begin = run_end;
    }

    reason.clear();
    return true;
#else
    GGML_UNUSED(request);
    GGML_UNUSED(ctx);
    reason = "TurboQuant Vulkan native materialization is unavailable because this build does not include GGML Vulkan support";
    return false;
#endif
}

bool llama_turboquant_runtime_sync_native_vulkan(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason) {
#if defined(GGML_USE_VULKAN)
    if (ctx.device == nullptr || ctx.backend == nullptr) {
        reason = "TurboQuant Vulkan shadow sync requires an initialized Vulkan backend context";
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ctx.device);
    if (reg == nullptr) {
        reason = "TurboQuant Vulkan shadow sync could not resolve the Vulkan backend registry";
        return false;
    }

    auto * fn = (llama_turboquant_backend_sync_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_turboquant_sync_shadow");
    if (fn == nullptr) {
        reason = "TurboQuant Vulkan shadow sync entry point is not exported by the Vulkan backend";
        return false;
    }

    static thread_local std::vector<uint32_t> row_indices;
    size_t packed_row_size = 0;

    if (!llama_turboquant_runtime_validate_rows(request, packed_row_size, reason)) {
        return false;
    }

    size_t run_begin = 0;
    while (run_begin < request.rows.size()) {
        size_t run_end = run_begin + 1;
        while (run_end < request.rows.size()) {
            const auto & prev = request.rows[run_end - 1];
            const auto & curr = request.rows[run_end];
            if (curr.row_index != prev.row_index + 1) {
                break;
            }
            if (curr.packed != prev.packed + packed_row_size) {
                break;
            }
            ++run_end;
        }

        const size_t run_size = run_end - run_begin;
        row_indices.resize(run_size);
        for (size_t i = 0; i < run_size; ++i) {
            row_indices[i] = request.rows[run_begin + i].row_index;
        }

        if (!fn(
                ctx.backend,
                request.buffer.tensor,
                request.buffer.kv_size,
                request.buffer.n_row_el,
                request.buffer.transposed,
                request.codec_params.group_size,
                request.codec_params.residual_bits,
                request.codec_params.qjl,
                request.rows[run_begin].packed,
                packed_row_size,
                row_indices.data(),
                row_indices.size())) {
            reason = "TurboQuant Vulkan shadow sync entry point reported failure";
            return false;
        }

        run_begin = run_end;
    }

    reason.clear();
    return true;
#else
    GGML_UNUSED(request);
    GGML_UNUSED(ctx);
    reason = "TurboQuant Vulkan shadow sync is unavailable because this build does not include GGML Vulkan support";
    return false;
#endif
}
