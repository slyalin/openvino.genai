// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS_FILE> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";

    using namespace ov::genai;

    Adapter adapter(adapter_path);
    LLMPipeline pipe(model_path, device, adapters(adapter, AdapterConfig::MODE_DYNAMIC));    // register all required adapters here

    std::cout << "Generate with LoRA adapter and alpha set to 0.75:" << std::endl;
    std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter, 0.75)) << std::endl;

    std::cout << "\n-----------------------------";
    std::cout << "\nGenerate without LoRA adapter:" << std::endl;
    std::cout << pipe.generate(prompt, max_new_tokens(100), adapters()) << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
