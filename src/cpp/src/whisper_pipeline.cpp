// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <variant>

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "whisper/whisper.hpp"
#include "whisper/whisper_config.hpp"
#include "whisper/whisper_feature_extractor.hpp"
#include "whisper/whisper_models.hpp"

namespace {
ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& models_path) {
    auto config_file_path = models_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::WhisperGenerationConfig((config_file_path).string());
    } else {
        return ov::genai::WhisperGenerationConfig{};
    }
}

ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}
}  // namespace


void add_attention_mask_input(std::shared_ptr<ov::Model> model) {
    using namespace ov::pass::pattern;
    using namespace ov::op;
    class AttentionMaskInput : public ov::pass::MatcherPass {
    public:

        OPENVINO_RTTI("AttentionMaskInput");

        AttentionMaskInput(std::shared_ptr<ov::Model> model) {
            auto range = wrap_type<v4::Range>();
            auto convert1 = wrap_type<v0::Convert>({range});
            auto greater = wrap_type<v1::Greater>({convert1, any_input()});
            auto convert2 = wrap_type<v0::Convert>({greater});

            register_matcher(
                std::make_shared<Matcher>(convert2, this->get_type_info().name), [model](Matcher& m) {
                    auto node = m.get_match_root();
                    auto attention_mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
                    attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                    model->add_parameters({attention_mask});
                    ov::replace_node(node, attention_mask);
                    return false;
                }
            );
        }
    };

    ov::pass::Manager pm;
    pm.register_pass<AttentionMaskInput>(model);
    pm.run_passes(model);
}


namespace ov {
namespace genai {

class WhisperPipeline::Impl {
private:
    ov::genai::WhisperConfig m_model_config;

public:
    ov::genai::WhisperGenerationConfig m_generation_config;
    ov::genai::WhisperInitializedModels m_models;
    ov::genai::WhisperFeatureExtractor m_feature_extractor;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0;

    Impl(const std::filesystem::path& models_path,
         const std::string& device,
         const ov::AnyMap& properties)
        : m_generation_config{from_config_json_if_exists(models_path)},
          m_tokenizer{models_path},
          m_feature_extractor{(models_path / "preprocessor_config.json")},
          m_model_config{(models_path / "config.json")} {
        ov::Core core = utils::singleton_core();
        auto [core_properties, compile_properties] = ov::genai::utils::split_core_complile_config(properties);
        core.set_property(core_properties);

        m_models.encoder = core.compile_model((models_path / "openvino_encoder_model.xml").string(), device, compile_properties)
                               .create_infer_request();
        m_models.decoder = core.compile_model((models_path / "openvino_decoder_model.xml").string(), device, compile_properties)
                               .create_infer_request();
        auto decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml");
        add_attention_mask_input(decoder_with_past_model);
        m_models.decoder_with_past =
            core.compile_model(decoder_with_past_model, device, compile_properties)
                .create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }
    }

    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config,
                                   StreamerVariant streamer) {
        auto start_time = std::chrono::steady_clock::now();
        WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        config.validate();

        std::shared_ptr<StreamerBase> streamer_ptr;
        if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
            streamer_ptr = nullptr;
        } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
            streamer_ptr = *streamer_obj;
        } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
            streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
        }

        auto generate_result = ov::genai::whisper_generate(config,
                                                           m_model_config,
                                                           raw_speech_input,
                                                           m_models,
                                                           m_feature_extractor,
                                                           streamer_ptr);
        auto decode_start_time = std::chrono::steady_clock::now();
        WhisperDecodedResults result{std::vector{m_tokenizer.decode(generate_result.output_tokens)}, std::vector{1.f}};
        generate_result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));

        result.perf_metrics = generate_result.perf_metrics;
        auto& segments = generate_result.segments;

        if (segments.has_value()) {
            std::vector<WhisperDecodedResultChunk> chunks;
            chunks.reserve((*segments).size());

            for (auto& segment : *segments) {
                decode_start_time = std::chrono::steady_clock::now();
                chunks.push_back(
                    WhisperDecodedResultChunk{segment.m_start, segment.m_end, m_tokenizer.decode(segment.m_tokens)});
                result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
                    PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
            }

            result.chunks = chunks;
        }


        m_models.decoder_with_past.reset_state();

        auto& metrics = result.perf_metrics;
        metrics.load_time = this->m_load_time_ms;
        auto stop_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(MicroSeconds(0.0f));
        metrics.evaluate_statistics(start_time);

        return result;
    }
};

}  // namespace genai
}  // namespace ov

ov::genai::WhisperPipeline::WhisperPipeline(const std::filesystem::path& models_path,
                                            const std::string& device,
                                            const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = std::make_unique<WhisperPipeline::Impl>(models_path, device, properties);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      OptionalWhisperGenerationConfig generation_config,
                                                                      StreamerVariant streamer) {
    return m_impl->generate(raw_speech_input, generation_config, streamer);
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_impl->generate(raw_speech_input, config, utils::get_streamer_from_map(config_map));
}

ov::genai::WhisperGenerationConfig ov::genai::WhisperPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::WhisperPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

void ov::genai::WhisperPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    int64_t default_eos_token_id = m_impl->m_generation_config.eos_token_id;
    m_impl->m_generation_config = config;
    // if eos_token_id was not provided in config forward from default config
    if (config.eos_token_id == -1)
        m_impl->m_generation_config.eos_token_id = default_eos_token_id;

    m_impl->m_generation_config.validate();
}

ov::genai::WhisperPipeline::~WhisperPipeline() = default;
