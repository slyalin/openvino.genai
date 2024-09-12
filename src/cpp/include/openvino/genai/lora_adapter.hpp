// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>
#include <optional>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"


namespace ov {
namespace genai {

// FIXME: Remove or move to a dedicated common header
#ifdef NDEBUG
    #define DEBUG_PRINT(X) do {} while(false)
#else
    #define DEBUG_PRINT(X) do { std::cerr << "[ DEBUG ] " << X << "\n"; } while(false)
#endif

class OPENVINO_GENAI_EXPORTS AdapterController;
struct AdapterControllerImpl;

// Inmutable LoRA Adapter that carries the adaptation matrices and the default alpha value
class OPENVINO_GENAI_EXPORTS Adapter {
    class Impl;
    std::shared_ptr<Impl> m_pimpl;
    friend AdapterController;
    friend AdapterControllerImpl;
    friend bool operator== (const Adapter& a, const Adapter& b);
    friend bool operator< (const Adapter& a, const Adapter& b);
public:
    explicit Adapter(const std::string& path, float default_alpha);
    explicit Adapter(const std::string& path);
    Adapter() = default;
    std::optional<float> get_default_alpha() const;

    operator bool() const {
        return bool(m_pimpl);
    }

    // TODO: Mapping between names of layers in a model and tensor names in the adapter
};

bool OPENVINO_GENAI_EXPORTS operator== (const Adapter& a, const Adapter& b);


struct OPENVINO_GENAI_EXPORTS AdapterConfig {
    enum Mode {
        MODE_AUTO,          // Automatically selected (depends on the place where this mode is applied and device selection)
        MODE_DYNAMIC,       // A, B, alpha are fully variable
        MODE_STATIC_RANK,   // A and B have static shape, alpha is variable // FIXME: WA to unlock experiments, gives a unique perf level
        MODE_STATIC,        // A, B and alpha are constants
        MODE_FUSE           // A, B and alpha are constants, fused to main matrix W
    };

    Mode get_mode() const { return mode; }
    void set_mode(Mode);

    AdapterConfig (const Adapter& adapter, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<Adapter>{adapter}, mode) {}
    AdapterConfig (const Adapter& adapter, float alpha, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<std::pair<Adapter, float>>{{adapter, alpha}}, mode) {}
    AdapterConfig (const Adapter& adapter, double alpha, Mode mode = MODE_AUTO) : AdapterConfig(adapter, float(alpha), mode) {}
    AdapterConfig (const std::vector<Adapter>& adapters, Mode mode = MODE_AUTO);
    AdapterConfig (const std::vector<std::pair<Adapter, float>>& adapters, Mode mode = MODE_AUTO);

    template <typename T, typename = std::enable_if<std::is_same<T, Adapter>::value, T>>
    AdapterConfig (const std::initializer_list<T>& adapters, Mode mode = MODE_AUTO) :
        AdapterConfig(std::vector<Adapter>(adapters), mode) {}

    AdapterConfig(Mode mode = MODE_AUTO);

    AdapterConfig& add(const Adapter& adapter, float alpha);
    AdapterConfig& add(const Adapter& adapter);
    AdapterConfig& set_alpha(const Adapter& adapter, float alpha);
    float get_alpha(const Adapter& adapter) const;
    AdapterConfig& remove(const Adapter&);
    const std::vector<Adapter>& get_adapters() const { return adapters; }

    // Returns true if it is not a trivial config
    operator bool() const {
        return !adapters.empty();
    }

private:

    Mode mode;
    std::vector<Adapter> adapters;
    std::vector<float> alphas;
    //std::set<std::string> modules;  // additional modules that can be patched, from LoRA config "target_modules": ["q_proj", "v_proj"] etc.  // TODO: Implement this feature
    //ov::element::Type adapter_element_type = ov::element::dynamic; // optional element type for adapter tensors in case if multiple adapters have various types or they are not known in advance

};


class OPENVINO_GENAI_EXPORTS AdapterController {

    std::shared_ptr<AdapterControllerImpl> m_pimpl;
    friend AdapterControllerImpl;

public:

    AdapterController() = default;
    AdapterController(std::shared_ptr<ov::Model> model, const AdapterConfig& config, const std::string& prefix);

    // Call it every time when adapter config is changed; if adapter is configured as a static one, this call is not required
    void apply(ov::InferRequest& request, const AdapterConfig& config);

    // the next call of apply will set all adapter tensors regardless of config change, use this method if full state.reset is called for the controlled model
    void force_full_apply(bool full_apply = true);

    // Apply the same config that was used last time (in initialization or in previous call to apply).
    void apply(ov::InferRequest& request);

    operator bool() const {
        return bool(m_pimpl);
    }
};


}  // namespace genai
}  // namespace ov
