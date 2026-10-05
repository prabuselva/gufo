#ifndef GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
  std::string name;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete function object, including optional fields such as response.
  std::string definition_json;
};

struct ChatTemplateOptions {
  bool add_generation_prompt{true};
  bool enable_thinking{false};
  bool preserve_thinking{false};
};

/// Suffix opened for a new assistant turn, outside the stable conversation.
[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Renders the pinned Gemma-4 conversation format deterministically.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Rejects Gemma-4 artifacts whose embedded template is not the pinned
/// converter version implemented by the compiled formatter.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "gemma4-turn-compiled-v1";
}

[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
}

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_