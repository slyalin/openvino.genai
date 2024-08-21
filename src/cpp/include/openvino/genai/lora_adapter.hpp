// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"

#define DEBUG_PRINT(X) do { std::cerr << "[ DEBUG ] " << X << "\n"; } while(false)

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS AdapterController;
struct AdapterControllerImpl;

// Inmutable LoRA Adapter that carries the adaptation matrices and the default alpha value
class OPENVINO_GENAI_EXPORTS Adapter {
    class Impl;
    std::shared_ptr<Impl> m_pimpl;
    friend AdapterController;
    friend AdapterControllerImpl;
    friend bool operator== (const Adapter& a, const Adapter& b);
public:
    explicit Adapter(const std::string& path, float default_alpha);
    explicit Adapter(const std::string& path);
    Adapter() = default;

    operator bool() const {
        return bool(m_pimpl);
    }

    // TODO: Mapping between names of layers in a model and tensor names in the adapter
};

bool OPENVINO_GENAI_EXPORTS operator== (const Adapter& a, const Adapter& b);

struct OPENVINO_GENAI_EXPORTS AdaptersConfig {
public:
    bool is_dynamic = false;    // false -- parameters cannot be changed during inference, this config should match for every generation
    std::vector<Adapter> adapters;
    std::vector<float> alphas;
    std::set<std::string> modules;  // additional modules that can be patched, from LoRA config "target_modules": ["q_proj", "v_proj"] etc.
    ov::element::Type adapter_element_type = ov::element::dynamic; // optional element type for adapter tensors in case if multiple adapters have various types or they are not known in advance

    AdaptersConfig (const std::vector<Adapter>& adapters = {}, bool is_dynamic = false) : is_dynamic(is_dynamic), adapters(adapters) {}
    AdaptersConfig (const Adapter& adapter, bool is_dynamic = false);

    AdaptersConfig (const Adapter& adapter, bool alpha, bool is_dynamic = false) : is_dynamic(is_dynamic) {
        add(adapter, alpha);
    }

    AdaptersConfig& add(const Adapter& adapter, float alpha);
    AdaptersConfig& add(const Adapter& adapter);
    AdaptersConfig& set(const Adapter& adapter, float alpha);
    AdaptersConfig& set(const Adapter& adapter);
    AdaptersConfig& remove(const Adapter);

    // Returns true if it is not a trivial config
    operator bool() const {
        return is_dynamic || !adapters.empty();
    }
};


class OPENVINO_GENAI_EXPORTS AdapterController {
    // FIXME: Should hold AdaptersConfig to compare with previsly set config and to hold Adapter objects

    std::shared_ptr<AdapterControllerImpl> m_pimpl;
    friend AdapterControllerImpl;
    //static std::shared_ptr<Adapter::Impl> get_adapter_impl(const Adapter& adapter);
public:
    AdapterController() = default;
    AdapterController(std::shared_ptr<ov::Model> model, const AdaptersConfig& config, const std::string& prefix);

    // Call it every time when adapter config is changed; if adapter was configured as a static one, this call is not required
    void apply(ov::InferRequest& request, const AdaptersConfig& config);

    // Apply the same config that was used last time (in initialization or in previous call to apply).
    void apply(ov::InferRequest& request);

    operator bool() const {
        return bool(m_pimpl);
    }
};



}  // namespace genai
}  // namespace ov
