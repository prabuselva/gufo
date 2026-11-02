#include "src/models/flint/chat_template.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace gufo::models::flint {
namespace {

// The control-token delimiters, spelled with hex escapes so no source or
// build output ever emits a literal special-token sequence.
constexpr std::string_view kCtrlOpen = "\x3C\x7C";
constexpr std::string_view kCtrlClose = "\x7C\x3E";

[[nodiscard]] std::string Control(std::string_view name) {
  return std::string(kCtrlOpen) + std::string(name) + std::string(kCtrlClose);
}

void ReplaceAll(std::string* text, std::string_view from, std::string_view to) {
  if (from.empty()) {
    return;
  }
  std::size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

/// The `flint_text` macro: drop NUL bytes and think-tag spellings, then
/// HTML-escape the registered special-token spellings so untrusted content
/// cannot forge turn framing.
void Sanitize(std::string* text) {
  ReplaceAll(text, std::string(1, '\0'), "");
  ReplaceAll(text, "<think>", "");
  ReplaceAll(text, "</think>", "");
  for (std::string_view name :
       {"begin_of_text", "end_of_text", "system", "developer", "user",
        "assistant", "observation"}) {
    ReplaceAll(text, Control(name), "&lt;|" + std::string(name) + "|&gt;");
  }
}

void AppendJsonString(std::string* out, std::string_view text) {
  out->push_back('"');
  for (const char ch : text) {
    const auto code = static_cast<unsigned char>(ch);
    switch (ch) {
      case '"':
        out->append("\\\"");
        break;
      case '\\':
        out->append("\\\\");
        break;
      case '\n':
        out->append("\\n");
        break;
      case '\r':
        out->append("\\r");
        break;
      case '\t':
        out->append("\\t");
        break;
      default:
        if (code < 0x20) {
          std::ostringstream esc;
          esc << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(code);
          out->append(esc.str());
        } else {
          out->push_back(ch);
        }
    }
  }
  out->push_back('"');
}

/// One tool exactly like Jinja `tool.function | tojson(ensure_ascii=False)`:
/// compact object text, UTF-8 passthrough, template field order.
void AppendToolJson(std::string* out, const tokenization::ChatTool& tool) {
  out->append("{\"name\":");
  AppendJsonString(out, tool.name);
  out->append(",\"description\":");
  AppendJsonString(out, tool.description);
  out->append(",\"parameters\":");
  out->append(tool.parameters_json);
  out->push_back('}');
}

void AppendToolPreamble(std::string* out,
                        std::span<const tokenization::ChatTool> tools) {
  out->append(Control("system"));
  out->append(
      "\n# Tools\n\n"
      "You may call one or more functions to assist with the user "
      "query.\n\n"
      "You are provided with function signatures within <tools></tools>"
      " XML tags:\n<tools>\n");
  for (const auto& tool : tools) {
    AppendToolJson(out, tool);
    out->push_back('\n');
  }
  out->append(
      "</tools>\n\n"
      "For each function call, output the function name and arguments "
      "within the following XML format:\n"
      "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
      "<arg_value>{arg-value-1}</arg_value>...</tool_call>\n");
}

}  // namespace

std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt, std::span<const tokenization::ChatTool> tools) {
  std::string prompt;
  bool tools_pending = !tools.empty();
  for (const auto& message : messages) {
    if (tools_pending && message.role != tokenization::ChatRole::kSystem) {
      AppendToolPreamble(&prompt, tools);
      tools_pending = false;
    }
    switch (message.role) {
      case tokenization::ChatRole::kSystem: {
        std::string content = message.content;
        Sanitize(&content);
        prompt.append(Control("system"));
        prompt.append(content);
        break;
      }
      case tokenization::ChatRole::kDeveloper: {
        std::string content = message.content;
        Sanitize(&content);
        prompt.append(Control("developer"));
        prompt.append(content);
        break;
      }
      case tokenization::ChatRole::kUser: {
        std::string content = message.content;
        Sanitize(&content);
        prompt.append(Control("user"));
        prompt.append(content);
        break;
      }
      case tokenization::ChatRole::kAssistant: {
        prompt.append(Control("assistant"));
        if (!message.thought.empty()) {
          prompt.append("<think>");
          prompt.append(message.thought);
          prompt.append("</think>");
        }
        if (!message.content.empty()) {
          prompt.append(message.content);
        }
        for (const auto& call : message.tool_calls) {
          prompt.append("<tool_call>");
          prompt.append(call.name);
          for (const auto& argument : call.arguments) {
            prompt.append("<arg_key>");
            prompt.append(argument.name);
            prompt.append("</arg_key><arg_value>");
            // Strings pass through; non-strings arrive as their JSON text
            // (the server normalizes both), so the bytes go verbatim.
            prompt.append(argument.value);
            prompt.append("</arg_value>");
          }
          prompt.append("</tool_call>");
        }
        break;
      }
      case tokenization::ChatRole::kTool: {
        std::string content = message.content;
        Sanitize(&content);
        prompt.append(Control("observation"));
        prompt.append("<tool_response>");
        prompt.append(content);
        prompt.append("</tool_response>");
        break;
      }
    }
  }
  if (tools_pending) {
    AppendToolPreamble(&prompt, tools);
  }
  if (add_generation_prompt) {
    prompt.append(Control("assistant"));
  }
  return prompt;
}

}  // namespace gufo::models::flint
