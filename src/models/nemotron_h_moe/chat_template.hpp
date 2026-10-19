#ifndef GUFO_MODELS_NEMOTRON_H_MOE_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::nemotron_h_moe {

/// Renders the ChatML framing the Bolt checkpoint consumes: one
/// `<|im_start|>{role}\n{content}\n` block per turn plus the assistant
/// opener. No turn closer is emitted; the vocabulary's stop spellings stay
/// out of the prompt and generation ends on the length limit. Returns
/// nullopt for the channels this runtime does not serve (tools, reasoning).
[[nodiscard]] std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt = true,
    std::span<const tokenization::ChatTool> tools = {});

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_CHAT_TEMPLATE_HPP_
