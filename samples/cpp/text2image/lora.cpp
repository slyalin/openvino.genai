// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

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
    size_t n_adapters = (argc - 4)/2;
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
    ov::genai::Text2ImagePipeline pipe(models_path, device, mode == "NO" ? ov::AnyMap{} : ov::AnyMap{ov::genai::adapters(adapter_config)});
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Initialization: " << ms_int.count() << "\n";

    ov::AnyMap conf = {
        ov::genai::generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
        ov::genai::width(512),
        ov::genai::height(896),
        ov::genai::num_inference_steps(20)
    };

    if(mode == "EMPTY") {
        conf.insert(ov::genai::adapters());
    }

    const size_t static_alpha_n = 3;

    for(size_t i = 0; i < static_alpha_n; ++i) {
        auto t1 = high_resolution_clock::now();
        ov::Tensor image = pipe.generate(prompt, conf);
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Iteration with the same alpha " << i + 1 << ": " << ms_int.count() << "\n";
        imwrite("benchmark." + mode + "." + std::to_string(i+1) + ".bmp", image, true);
    }

    if(mode == "DYNAMIC") {
        for(size_t i = 0; i < 3; ++i) {
            auto t1 = high_resolution_clock::now();
            for(size_t i = 0; i < n_adapters; ++i) {
                auto adapter = adapter_config.get_adapters()[i];
                adapter_config.set_alpha(adapter, adapter_config.get_alpha(adapter) + 0.01);
            }
            conf[ov::genai::adapters.name()] = adapter_config;
            ov::Tensor image = pipe.generate(prompt, conf);
            auto t2 = high_resolution_clock::now();
            auto ms_int = duration_cast<milliseconds>(t2 - t1);
            std::cout << "Iteration with changed alpha " << i + 1 << ": " << ms_int.count() << "\n";
            imwrite("benchmark." + mode + "." + std::to_string(static_alpha_n+i+1) + ".bmp", image, true);
        }
    }

    #if 0
    imwrite("lora.bmp", image, true);

    std::cout << "Generating image without LoRA adapters applied, resulting image will be in baseline.bmp\n";
    image = pipe.generate(prompt,
        ov::genai::adapters(),  // passing adapters in generate overrides adapters set in the constructor; adapters() means no adapters
        ov::genai::generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
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
