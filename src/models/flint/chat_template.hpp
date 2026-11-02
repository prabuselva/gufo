#ifndef GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::flint {

/// Renders the tagged-turn framing of the Flint GGUF
/// `tokenizer.chat_template`: every message opens with its control token
/// (`system`, `developer`, `user`, `assistant`, `observation`) and no
/// newline framing beyond the tool preamble; turn boundaries are the
/// control tokens themselves. Message content is sanitized (NUL bytes and
/// think-tag spellings dropped, the seven registered special spellings
/// HTML-escaped), assistant history carries reasoning inside think tags
/// and tool calls as tool_call/arg_key/arg_value XML, and tool answers
/// arrive as observation turns wrapped in tool_response. The tool preamble
/// is emitted once, before the first non-system message (or at the end
/// when every message is a system turn).
[[nodiscard]] std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt = true,
    std::span<const tokenization::ChatTool> tools = {});

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CHAT_TEMPLATE_HPP_
