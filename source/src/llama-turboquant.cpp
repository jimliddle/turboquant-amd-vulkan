#include "llama-turboquant.h"

llama_turboquant_plan llama_turboquant_make_plan(const llama_context_params & params, bool offload_kqv) {
    llama_turboquant_plan plan = {
        /*.enabled  =*/ params.kv_cache_codec == LLAMA_KV_CACHE_CODEC_TURBOQUANT,
        /*.fallback =*/ false,
        /*.runtime  =*/ params.turboquant_runtime,
        /*.reason   =*/ "",
    };

    if (!plan.enabled) {
        return plan;
    }

    if (params.turboquant_group_size == 0) {
        plan.fallback = true;
        plan.reason = "TurboQuant group size must be greater than zero";
        return plan;
    }

    if (params.turboquant_residual_bits > 8) {
        plan.fallback = true;
        plan.reason = "TurboQuant residual bits must be in the range [0, 8]";
        return plan;
    }

    if (!offload_kqv) {
        plan.fallback = true;
        plan.reason = "TurboQuant currently targets offloaded KV execution paths";
        return plan;
    }

    if (plan.runtime == LLAMA_TURBOQUANT_RUNTIME_AUTO) {
#if defined(GGML_USE_HIP)
        plan.runtime = LLAMA_TURBOQUANT_RUNTIME_HIP;
#elif defined(GGML_USE_VULKAN)
        plan.runtime = LLAMA_TURBOQUANT_RUNTIME_VULKAN;
#endif
    }

    switch (plan.runtime) {
        case LLAMA_TURBOQUANT_RUNTIME_HIP:
#if !defined(GGML_USE_HIP)
            plan.fallback = true;
            plan.reason = "TurboQuant HIP runtime requested, but this build does not include GGML HIP support";
#endif
            break;
        case LLAMA_TURBOQUANT_RUNTIME_VULKAN:
#if !defined(GGML_USE_VULKAN)
            plan.fallback = true;
            plan.reason = "TurboQuant Vulkan runtime requested, but this build does not include GGML Vulkan support";
#endif
            break;
        case LLAMA_TURBOQUANT_RUNTIME_AUTO:
            plan.fallback = true;
            plan.reason = "TurboQuant runtime auto-selection could not resolve a compiled GPU backend";
            break;
    }

    if (plan.fallback && plan.reason.empty()) {
        plan.reason = "TurboQuant kernels are not available for the selected runtime";
    }

    return plan;
}
