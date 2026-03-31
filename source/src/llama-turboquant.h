#pragma once

#include "llama.h"

#include <string>

struct llama_turboquant_plan {
    bool enabled = false;
    bool fallback = false;
    llama_turboquant_runtime runtime = LLAMA_TURBOQUANT_RUNTIME_AUTO;
    std::string reason;
};

llama_turboquant_plan llama_turboquant_make_plan(const llama_context_params & params, bool offload_kqv);
