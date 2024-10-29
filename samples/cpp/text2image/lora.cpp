// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/pipeline.hpp"

#include <chrono>

#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 4 && (argc - 4) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...] <MODE>");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU, NPU can be used as well
    std::string mode = argv[argc-1];
    std::cout << "MODE: " << mode << "\n";

    std::map<std::string, ov::genai::AdapterConfig::Mode> modes = {
        {"NO", ov::genai::AdapterConfig::MODE_AUTO},  // doesn't matter
        {"EMPTY", ov::genai::AdapterConfig::MODE_DYNAMIC},
        {"DYNAMIC", ov::genai::AdapterConfig::MODE_DYNAMIC},
        {"STATIC", ov::genai::AdapterConfig::MODE_STATIC},
        {"FUSE", ov::genai::AdapterConfig::MODE_FUSE},
    };


    ov::genai::AdapterConfig adapter_config(modes.at(mode));
    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for(size_t i = 0; i < (argc - 4)/2; ++i) {
        ov::genai::Adapter adapter(argv[3 + 2*i]);
        float alpha = std::atof(argv[3 + 2*i + 1]);
        adapter_config.add(adapter, alpha);
    }

    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;

    // LoRA adapters passed to the constructor will be activated by default in next generates
    auto t1 = high_resolution_clock::now();
    ov::genai::Text2ImagePipeline pipe(models_path, device, mode == "NO" ? ov::AnyMap{} : ov::AnyMap{ov::genai::adapters(adapter_config)});
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Initialization: " << ms_int.count() << "\n";

    ov::AnyMap conf = {
        ov::genai::random_generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20)
    };

    if(mode == "EMPTY") {
        conf.insert(ov::genai::adapters());
    }

    for(size_t i = 0; i < 3; ++i) {
        auto t1 = high_resolution_clock::now();
        ov::Tensor image = pipe.generate(prompt, conf);
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Iteration " << i + 1 << ": " << ms_int.count() << "\n";
        imwrite("benchmark." + mode + "." + std::to_string(i+1) + ".bmp", image, true);
    }

    #if 0
    imwrite("lora.bmp", image, true);

    std::cout << "Generating image without LoRA adapters applied, resulting image will be in baseline.bmp\n";
    image = pipe.generate(prompt,
        ov::genai::adapters(),  // passing adapters in generate overrides adapters set in the constructor; adapters() means no adapters
        ov::genai::random_generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
        ov::genai::width(512),
        ov::genai::height(896),
        ov::genai::num_inference_steps(20));
    imwrite("baseline.bmp", image, true);
    #endif

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
