// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    // Prepare Adapter object before creation of LLMPipeline
    ov::genai::Adapter adapter = ov::genai::Adapter(adapter_path);
    ov::genai::AdaptersConfig adapters_config(adapter, /*alpha = */ 1.0);

    // Pass AdapterConfig to LLMPipeline to be able to dynamically connect adapter in following generate calls
    ov::genai::LLMPipeline pipe(model_path, device, adapters_config);

    // Create generation config as usual or take it from an LLMPipeline
    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 100;
    // Note: If a GenerationConfig object is created from scratch and not given by `get_generation_config`
    // you need to set AdaptersConfig manually to it, the adapters won't be applied otherwise.

    std::cout << "*** Generation with LoRA adapter applied: ***\n";
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    std::cout << "*** Generation without LoRA adapter: ****\n";
    // Set alpha to 0 for a paticular adapter to disable it.
    config.adapters.set(adapter, 0.0);
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
