#include "src/models/nemotron_h_moe/chat_template.hpp"

#include <string_view>

#include "src/models/nemotron_h_moe/tokenizer.hpp"

namespace gufo::models::nemotron_h_moe {

std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt, std::span<const tokenization::ChatTool> tools) {
  if (!tools.empty() || messages.empty()) {
    return std::nullopt;
  }
  std::string prompt;
  for (const tokenization::ChatMessage& message : messages) {
    if (message.role == tokenization::ChatRole::kTool ||
        !message.tool_calls.empty() || !message.thought.empty() ||
        !message.framing_suffix.empty() || !message.images.empty()) {
      return std::nullopt;
    }
    prompt.append(kImStart);
    prompt.append(tokenization::ToString(message.role));
    prompt.push_back('\n');
    prompt.append(message.content);
    prompt.push_back('\n');
  }
  if (add_generation_prompt) {
    prompt.append(kImStart);
    prompt.append("assistant\n");
  }
  return prompt;
}

}  // namespace gufo::models::nemotron_h_moe
