#include "src/models/ling3_flash/chat_template.hpp"

#include <string>

#include "src/models/ling3_flash/tool_syntax.hpp"

namespace gufo::models::ling3_flash {
namespace {

constexpr std::string_view kDetailedThinkingPrefix = "detailed thinking ";

bool HasThinkingDirective(std::string_view content) {
  return content.find(kDetailedThinkingPrefix) != std::string_view::npos &&
         (content.find("detailed thinking on") != std::string_view::npos ||
          content.find("detailed thinking off") != std::string_view::npos);
}

void TrimLeadingNewlines(std::string& text) {
  const auto keep = text.find_first_not_of('\n');
  text.erase(0, keep == std::string::npos ? text.size() : keep);
}

void TrimTrailingNewlines(std::string& text) {
  const auto keep = text.find_last_not_of('\n');
  text.resize(keep == std::string::npos ? 0 : keep + 1);
}

}  // namespace

ChatTemplateOptions ResolveLing3ChatOptions(const ReasoningOptions& reasoning) {
  return {.enable_thinking = reasoning.enabled.value_or(true)};
}

std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    const ChatTemplateOptions& options, bool add_generation_prompt,
    std::span<const tokenization::ChatTool> tools) {
  const std::string_view thinking = options.enable_thinking ? "on" : "off";
  std::string out;

  // System header: the pinned template opens every conversation with the
  // SYSTEM role and guarantees a `detailed thinking` directive, appending it
  // to a system message that does not already carry one. With tools the
  // header carries the tools section after the system content and newline.
  const bool has_leading_system =
      !messages.empty() && messages[0].role == tokenization::ChatRole::kSystem;
  const bool system_has_directive =
      has_leading_system && HasThinkingDirective(messages[0].content);
  out.append(kRoleStart).append("SYSTEM").append(kRoleEnd);
  if (!tools.empty()) {
    if (has_leading_system) {
      out.append(messages[0].content).push_back('\n');
    }
    out.append(RenderToolsSection(tools));
    if (!system_has_directive) {
      out.append(kDetailedThinkingPrefix).append(thinking);
    }
  } else if (has_leading_system) {
    out.append(messages[0].content);
    if (!system_has_directive) {
      out.push_back('\n');
      out.append(kDetailedThinkingPrefix).append(thinking);
    }
  } else {
    out.append(kDetailedThinkingPrefix).append(thinking);
  }
  out.append(kTurnEnd);
  if (tools.empty() && has_leading_system) {
    // Without tools and with a leading system message, the template closes
    // the inner branch flush against an undashed `{% else %}`, so the newline
    // and indentation before that tag are rendered as header output.
    out.append("\n    ");
  }

  const std::size_t first_turn = has_leading_system ? 1 : 0;
  for (std::size_t i = first_turn; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (!message.images.empty() ||
        (!message.tool_calls.empty() &&
         message.role != tokenization::ChatRole::kAssistant)) {
      return std::nullopt;
    }
    switch (message.role) {
      case tokenization::ChatRole::kSystem:
      case tokenization::ChatRole::kDeveloper:
        out.append(kRoleStart)
            .append("SYSTEM")
            .append(kRoleEnd)
            .append(message.content)
            .append(kTurnEnd);
        break;
      case tokenization::ChatRole::kUser:
        out.append(kRoleStart)
            .append("HUMAN")
            .append(kRoleEnd)
            .append(message.content)
            .append(kTurnEnd);
        break;
      case tokenization::ChatRole::kAssistant: {
        std::string reasoning = message.thought;
        std::string content = message.content;
        if (!reasoning.empty()) {
          // The template writes the reasoning frame with the value stripped
          // of newlines on both sides.
          TrimLeadingNewlines(reasoning);
          TrimTrailingNewlines(reasoning);
          TrimLeadingNewlines(content);
        }
        if (reasoning.empty()) {
          const auto end = content.find(kThinkEnd);
          if (end != std::string::npos) {
            reasoning = content.substr(0, end);
            const auto start = reasoning.find(kThinkStart);
            if (start != std::string::npos) {
              reasoning.erase(0, start + kThinkStart.size());
            }
            content = content.substr(end + kThinkEnd.size());
            TrimLeadingNewlines(reasoning);
            TrimTrailingNewlines(reasoning);
            TrimLeadingNewlines(content);
          }
        }
        out.append(kRoleStart)
            .append("ASSISTANT")
            .append(kRoleEnd)
            .push_back('\n');
        if (!reasoning.empty()) {
          out.append(kThinkStart).append(reasoning).append(kThinkEnd);
        } else {
          out.append(kThinkStart).append(kThinkEnd);
        }
        out.append(content);
        for (std::size_t call = 0; call < message.tool_calls.size(); ++call) {
          if ((call == 0 && !content.empty()) || call > 0) {
            out.push_back('\n');
          }
          AppendToolCall(out, message.tool_calls[call]);
        }
        out.append(kTurnEnd);
        break;
      }
      case tokenization::ChatRole::kTool: {
        const bool run_continues =
            i > 0 && messages[i - 1].role == tokenization::ChatRole::kTool;
        if (!run_continues) {
          out.append(kRoleStart).append("OBSERVATION").append(kRoleEnd);
        }
        AppendToolResponse(out, message.content);
        if (i + 1 == messages.size() ||
            messages[i + 1].role != tokenization::ChatRole::kTool) {
          out.append(kTurnEnd);
        }
        break;
      }
    }
  }

  if (add_generation_prompt) {
    out.append(kRoleStart).append("ASSISTANT").append(kRoleEnd).push_back('\n');
    out.append(kThinkStart);
    if (!options.enable_thinking) {
      out.append(kThinkEnd);
    }
  }
  return out;
}

}  // namespace gufo::models::ling3_flash
