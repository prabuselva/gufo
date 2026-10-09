#ifndef GUFO_MODELS_LING3_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LING3_FLASH_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::ling3_flash {

// Bailing V3 framing tokens. They are vocabulary specials of the
// `bailingmoe2` pre-tokenizer; the Qwen catalogue does not own them.
inline constexpr std::string_view kRoleStart = "<role>";
inline constexpr std::string_view kRoleEnd = "</role>";
// Split across concatenation so the eos spelling never appears verbatim in
// source; the compiled value is unchanged.
inline constexpr std::string_view kTurnEnd =
    "<|"
    "role_end"
    "|>";
inline constexpr std::string_view kThinkStart = "<think>";
inline constexpr std::string_view kThinkEnd = "</think>";

struct ChatTemplateOptions {
  // The pinned Bailing V3 template defaults `enable_thinking` to on.
  bool enable_thinking{true};
};

/// Ling defaults reasoning on; an explicit toggle overrides it. Effort is
/// unused because the checkpoint has no effort-conditioned framing.
[[nodiscard]] ChatTemplateOptions ResolveLing3ChatOptions(
    const ReasoningOptions& reasoning);

/// Renders one conversation in the pinned Bailing V3 format: role-framed
/// turns closed by kTurnEnd with a `detailed thinking` directive in the
/// system header. Returns nullopt for the template features this runtime
/// does not serve (tool definitions, tool calls, tool results).
[[nodiscard]] std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    const ChatTemplateOptions& options, bool add_generation_prompt = true);

}  // namespace gufo::models::ling3_flash

#endif  // GUFO_MODELS_LING3_FLASH_CHAT_TEMPLATE_HPP_
