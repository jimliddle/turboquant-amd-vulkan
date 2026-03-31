#include "llama-turboquant-runtime.h"
#include "llama-turboquant-runtime-impl.h"

#include "ggml-backend.h"

#include <memory>
#include <utility>

namespace {

using ggml_backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

struct llama_turboquant_runtime_backend_entry {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_ptr backend = { nullptr, ggml_backend_free };
};

struct llama_turboquant_runtime_scratch {
    std::vector<uint8_t> decoded_rows;
    std::vector<uint8_t> column_data;
};

static ggml_backend_t llama_turboquant_runtime_get_backend(ggml_backend_dev_t device, std::string & reason);

static ggml_backend_buffer_t llama_turboquant_runtime_tensor_buffer(const ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

static ggml_backend_dev_t llama_turboquant_runtime_tensor_device(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = llama_turboquant_runtime_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    if (buft == nullptr) {
        return nullptr;
    }

    return ggml_backend_buft_get_device(buft);
}

static llama_turboquant_runtime llama_turboquant_runtime_from_device(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return LLAMA_TURBOQUANT_RUNTIME_AUTO;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg == nullptr) {
        return LLAMA_TURBOQUANT_RUNTIME_AUTO;
    }

    const char * reg_name = ggml_backend_reg_name(reg);
    if (reg_name == nullptr) {
        return LLAMA_TURBOQUANT_RUNTIME_AUTO;
    }

    const std::string name(reg_name);
    if (name == "HIP") {
        return LLAMA_TURBOQUANT_RUNTIME_HIP;
    }
    if (name == "Vulkan") {
        return LLAMA_TURBOQUANT_RUNTIME_VULKAN;
    }

    return LLAMA_TURBOQUANT_RUNTIME_AUTO;
}

static bool llama_turboquant_runtime_resolve(
        const llama_turboquant_runtime_request & request,
        llama_turboquant_runtime & effective_runtime,
        ggml_backend_dev_t & device,
        std::string & reason) {
    if (request.buffer.tensor == nullptr) {
        reason = "TurboQuant runtime request is missing a destination tensor";
        return false;
    }

    if (request.buffer.n_row_el == 0) {
        reason = "TurboQuant runtime request has zero row width";
        return false;
    }

    device = llama_turboquant_runtime_tensor_device(request.buffer.tensor);
    const llama_turboquant_runtime tensor_runtime = llama_turboquant_runtime_from_device(device);

    effective_runtime = request.runtime;
    if (effective_runtime == LLAMA_TURBOQUANT_RUNTIME_AUTO) {
        effective_runtime = tensor_runtime;
    }

    if (effective_runtime == LLAMA_TURBOQUANT_RUNTIME_AUTO) {
        reason.clear();
        return true;
    }

    if (device == nullptr) {
        reason = std::string("TurboQuant ") + llama_turboquant_runtime_name(effective_runtime) +
                " runtime requested, but the destination tensor has no backend device";
        return false;
    }

    if (tensor_runtime != effective_runtime) {
        reason = std::string("TurboQuant ") + llama_turboquant_runtime_name(effective_runtime) +
                " runtime requested for tensor on " + ggml_backend_dev_name(device) + " backend";
        return false;
    }

    reason.clear();
    return true;
}

static bool llama_turboquant_runtime_prepare_context(
        const llama_turboquant_runtime_request & request,
        llama_turboquant_runtime requested_runtime,
        llama_turboquant_runtime_context & ctx,
        std::string & reason) {
    if (!llama_turboquant_runtime_resolve(request, ctx.runtime, ctx.device, reason)) {
        return false;
    }

    if (requested_runtime != LLAMA_TURBOQUANT_RUNTIME_AUTO && ctx.runtime != requested_runtime) {
        reason = std::string("TurboQuant runtime context requested for ") +
                llama_turboquant_runtime_name(requested_runtime) + ", but resolved runtime is " +
                llama_turboquant_runtime_name(ctx.runtime);
        return false;
    }

    if (ctx.runtime == LLAMA_TURBOQUANT_RUNTIME_AUTO) {
        ctx.backend = nullptr;
        reason.clear();
        return true;
    }

    ctx.backend = llama_turboquant_runtime_get_backend(ctx.device, reason);
    return ctx.backend != nullptr;
}

static ggml_backend_ptr llama_turboquant_runtime_init_backend(ggml_backend_dev_t device, std::string & reason) {
    if (device == nullptr) {
        reason = "TurboQuant runtime staging requires a valid backend device";
        return { nullptr, ggml_backend_free };
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        reason = std::string("failed to initialize ") + ggml_backend_dev_name(device) + " backend";
        return { nullptr, ggml_backend_free };
    }

    reason.clear();
    return { backend, ggml_backend_free };
}

static ggml_backend_t llama_turboquant_runtime_get_backend(ggml_backend_dev_t device, std::string & reason) {
    static thread_local std::vector<llama_turboquant_runtime_backend_entry> cache;

    for (auto & entry : cache) {
        if (entry.device == device) {
            return entry.backend.get();
        }
    }

    ggml_backend_ptr backend = llama_turboquant_runtime_init_backend(device, reason);
    if (!backend) {
        return nullptr;
    }

    cache.push_back({
        /*.device  =*/ device,
        /*.backend =*/ std::move(backend),
    });

    reason.clear();
    return cache.back().backend.get();
}

static void llama_turboquant_runtime_set_tensor(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    if (backend != nullptr) {
        ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
    } else {
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
}

static llama_turboquant_runtime_scratch & llama_turboquant_runtime_get_scratch() {
    static thread_local llama_turboquant_runtime_scratch scratch;
    return scratch;
}

static bool llama_turboquant_runtime_materialize_reference(
        const llama_turboquant_runtime_request & request,
        ggml_backend_t backend,
        std::string & reason) {
    const size_t row_size = ggml_row_size(request.buffer.type, request.buffer.n_row_el);
    const size_t el_size = ggml_type_size(request.buffer.type);
    auto & scratch = llama_turboquant_runtime_get_scratch();

    scratch.decoded_rows.resize(request.rows.size() * row_size);
    for (size_t i = 0; i < request.rows.size(); ++i) {
        llama_turboquant_unpack_row(
                request.rows[i].packed,
                request.buffer.type,
                request.buffer.n_row_el,
                request.codec_params,
                scratch.decoded_rows.data() + i * row_size);
    }

    if (!request.buffer.transposed) {
        size_t run_begin = 0;

        while (run_begin < request.rows.size()) {
            size_t run_end = run_begin + 1;
            while (run_end < request.rows.size() &&
                    request.rows[run_end].row_index == request.rows[run_end - 1].row_index + 1) {
                ++run_end;
            }

            llama_turboquant_runtime_set_tensor(
                    backend,
                    request.buffer.tensor,
                    scratch.decoded_rows.data() + run_begin * row_size,
                    request.rows[run_begin].row_index * row_size,
                    (run_end - run_begin) * row_size);

            run_begin = run_end;
        }
    } else {
        scratch.column_data.resize(request.rows.size() * el_size);
        for (uint32_t j = 0; j < request.buffer.n_row_el; ++j) {
            for (size_t i = 0; i < request.rows.size(); ++i) {
                std::memcpy(
                        scratch.column_data.data() + i * el_size,
                        scratch.decoded_rows.data() + i * row_size + j * el_size,
                        el_size);
            }

            size_t run_begin = 0;
            while (run_begin < request.rows.size()) {
                size_t run_end = run_begin + 1;
                while (run_end < request.rows.size() &&
                        request.rows[run_end].row_index == request.rows[run_end - 1].row_index + 1) {
                    ++run_end;
                }

                llama_turboquant_runtime_set_tensor(
                        backend,
                        request.buffer.tensor,
                        scratch.column_data.data() + run_begin * el_size,
                        (request.rows[run_begin].row_index + j * request.buffer.kv_size) * el_size,
                        (run_end - run_begin) * el_size);

                run_begin = run_end;
            }
        }
    }

    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
    }

    reason.clear();
    return true;
}

static bool llama_turboquant_runtime_materialize_staged(
        const llama_turboquant_runtime_request & request,
        const llama_turboquant_runtime_context & ctx,
        std::string & reason) {
    return llama_turboquant_runtime_materialize_reference(request, ctx.backend, reason);
}

static bool llama_turboquant_runtime_materialize_cpu(
        const llama_turboquant_runtime_request & request,
        std::string & reason) {
    return llama_turboquant_runtime_materialize_reference(request, nullptr, reason);
}

static bool llama_turboquant_runtime_materialize_device_upload(
        const llama_turboquant_runtime_request & request,
        llama_turboquant_runtime runtime,
        std::string & reason) {
    llama_turboquant_runtime_context ctx;
    if (!llama_turboquant_runtime_prepare_context(request, runtime, ctx, reason)) {
        return false;
    }

    if (ctx.runtime != runtime) {
        return llama_turboquant_runtime_materialize_cpu(request, reason);
    }

    std::string native_reason;
    switch (runtime) {
        case LLAMA_TURBOQUANT_RUNTIME_HIP:
            if (llama_turboquant_runtime_materialize_native_hip(request, ctx, native_reason)) {
                reason.clear();
                return true;
            }
            break;
        case LLAMA_TURBOQUANT_RUNTIME_VULKAN:
            if (llama_turboquant_runtime_materialize_native_vulkan(request, ctx, native_reason)) {
                reason.clear();
                return true;
            }
            break;
        case LLAMA_TURBOQUANT_RUNTIME_AUTO:
        default:
            break;
    }

    const bool staged_ok = llama_turboquant_runtime_materialize_staged(request, ctx, reason);
    if (!staged_ok) {
        if (!native_reason.empty()) {
            reason = native_reason + "; " + reason;
        }
        return false;
    }

    if (!native_reason.empty()) {
        reason = native_reason + "; using staged upload fallback";
    } else {
        reason.clear();
    }
    return true;
}

static bool llama_turboquant_runtime_materialize_hip(
        const llama_turboquant_runtime_request & request,
        std::string & reason) {
#if defined(GGML_USE_HIP)
    return llama_turboquant_runtime_materialize_device_upload(request, LLAMA_TURBOQUANT_RUNTIME_HIP, reason);
#else
    GGML_UNUSED(request);
    reason = "TurboQuant HIP runtime requested, but this build does not include GGML HIP support";
    return false;
#endif
}

static bool llama_turboquant_runtime_materialize_vulkan(
        const llama_turboquant_runtime_request & request,
        std::string & reason) {
#if defined(GGML_USE_VULKAN)
    return llama_turboquant_runtime_materialize_device_upload(request, LLAMA_TURBOQUANT_RUNTIME_VULKAN, reason);
#else
    GGML_UNUSED(request);
    reason = "TurboQuant Vulkan runtime requested, but this build does not include GGML Vulkan support";
    return false;
#endif
}

static bool llama_turboquant_runtime_sync_vulkan(
        const llama_turboquant_runtime_request & request,
        std::string & reason) {
#if defined(GGML_USE_VULKAN)
    llama_turboquant_runtime_context ctx;
    if (!llama_turboquant_runtime_prepare_context(request, LLAMA_TURBOQUANT_RUNTIME_VULKAN, ctx, reason)) {
        return false;
    }

    if (ctx.runtime != LLAMA_TURBOQUANT_RUNTIME_VULKAN) {
        reason = "TurboQuant Vulkan shadow sync requested for a non-Vulkan tensor";
        return false;
    }

    return llama_turboquant_runtime_sync_native_vulkan(request, ctx, reason);
#else
    GGML_UNUSED(request);
    reason = "TurboQuant Vulkan shadow sync requested, but this build does not include GGML Vulkan support";
    return false;
#endif
}

} // namespace

bool llama_turboquant_runtime_materialize(const llama_turboquant_runtime_request & request, std::string & reason) {
    llama_turboquant_runtime effective_runtime = LLAMA_TURBOQUANT_RUNTIME_AUTO;
    ggml_backend_dev_t device = nullptr;
    if (!llama_turboquant_runtime_resolve(request, effective_runtime, device, reason)) {
        return false;
    }

    GGML_UNUSED(device);

    switch (effective_runtime) {
        case LLAMA_TURBOQUANT_RUNTIME_HIP:
            return llama_turboquant_runtime_materialize_hip(request, reason);
        case LLAMA_TURBOQUANT_RUNTIME_VULKAN:
            return llama_turboquant_runtime_materialize_vulkan(request, reason);
        case LLAMA_TURBOQUANT_RUNTIME_AUTO:
        default:
            return llama_turboquant_runtime_materialize_cpu(request, reason);
    }
}

bool llama_turboquant_runtime_sync_shadow(const llama_turboquant_runtime_request & request, std::string & reason) {
    llama_turboquant_runtime effective_runtime = LLAMA_TURBOQUANT_RUNTIME_AUTO;
    ggml_backend_dev_t device = nullptr;
    if (!llama_turboquant_runtime_resolve(request, effective_runtime, device, reason)) {
        return false;
    }

    GGML_UNUSED(device);

    switch (effective_runtime) {
        case LLAMA_TURBOQUANT_RUNTIME_VULKAN:
            return llama_turboquant_runtime_sync_vulkan(request, reason);
        case LLAMA_TURBOQUANT_RUNTIME_HIP:
            reason = "TurboQuant shadow sync is not implemented for HIP yet";
            return false;
        case LLAMA_TURBOQUANT_RUNTIME_AUTO:
        default:
            reason = "TurboQuant shadow sync requires a backend runtime";
            return false;
    }
}
