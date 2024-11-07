// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (5 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" [<LORA_SAFETENSORS> <ALPHA> ...] <MODE>");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well

    std::string mode = argv[argc-1];
    std::cout << "MODE: " << mode << "\n";

    std::map<std::string, ov::genai::AdapterConfig::Mode> modes = {
        {"NO", ov::genai::AdapterConfig::MODE_AUTO},  // doesn't matter
        {"EMPTY", ov::genai::AdapterConfig::MODE_DYNAMIC},
        {"DYNAMIC", ov::genai::AdapterConfig::MODE_DYNAMIC},
        {"STATIC", ov::genai::AdapterConfig::MODE_STATIC},
        {"FUSE", ov::genai::AdapterConfig::MODE_FUSE},
    };

    using namespace ov::genai;

    ov::genai::AdapterConfig adapter_config(modes.at(mode));
    size_t n_adapters = (argc - 4)/2;
    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for(size_t i = 0; i < n_adapters; ++i) {
        ov::genai::Adapter adapter(argv[3 + 2*i]);
        float alpha = std::atof(argv[3 + 2*i + 1]);
        adapter_config.add(adapter, alpha);
    }

    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;

    // LoRA adapters passed to the constructor will be activated by default in next generates
    auto t1 = high_resolution_clock::now();

    LLMPipeline pipe(models_path, device, mode == "NO" ? ov::AnyMap{} : ov::AnyMap{ov::genai::adapters(adapter_config)});    // register all required adapters here
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Initialization: " << ms_int.count() << "\n";
    ov::AnyMap conf = {
        max_new_tokens(20)
    };

    if(mode == "EMPTY") {
        conf.insert(ov::genai::adapters());
    }

    const size_t static_alpha_n = 3;

    for(size_t i = 0; i < static_alpha_n; ++i) {
        auto t1 = high_resolution_clock::now();
        std::cout << "Generated: " << pipe.generate(prompt, conf) << std::endl;
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Iteration " << i + 1 << ": " << ms_int.count() << "\n";
    }

    if(mode == "DYNAMIC") {
        for(size_t i = 0; i < 3; ++i) {
            auto t1 = high_resolution_clock::now();
            for(size_t i = 0; i < n_adapters; ++i) {
                auto adapter = adapter_config.get_adapters()[i];
                adapter_config.set_alpha(adapter, adapter_config.get_alpha(adapter) + 0.01);
            }
            conf[ov::genai::adapters.name()] = adapter_config;
            std::cout << "Generated: " << pipe.generate(prompt, conf) << std::endl;
            auto t2 = high_resolution_clock::now();
            auto ms_int = duration_cast<milliseconds>(t2 - t1);
            std::cout << "Iteration with changed alpha " << i + 1 << ": " << ms_int.count() << "\n";
        }
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
