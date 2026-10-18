#include "src/models/ling3_0/tool_syntax.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::models::ling3_0 {
namespace {

constexpr std::string_view kToolsBegin =
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>";

constexpr std::string_view kToolsEnd =
    "\n</tools>\n\nIf none of the functions can be used, point it out. If the "
    "given question lacks the parameters required by the function, also point "
    "it out.\nIf you need to use a function, for each function call, output "
    "the function name and arguments within the following XML format:\n"
    "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>"
    "{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>"
    "{arg-value-2}</arg_value>\n...\n</tool_call>\n";

void AppendCodepoint(std::string& out, std::uint32_t codepoint) {
  char buffer[8];
  (void)std::snprintf(buffer, sizeof(buffer), "\\u%04x", codepoint);
  out.append(buffer);
}

// Python's json.dumps + jinja's htmlsafe filter: control characters, every
// non-ASCII codepoint and <, >, &, ' all become lowercase \uXXXX escapes.
void AppendPythonJsonString(std::string& out, std::string_view value) {
  out.push_back('"');
  std::size_t i = 0;
  while (i < value.size()) {
    const auto byte = static_cast<unsigned char>(value[i]);
    std::uint32_t codepoint = byte;
    std::size_t width = 1;
    if (byte >= 0xF0 && i + 3 < value.size()) {
      codepoint = ((byte & 0x07U) << 18) |
                  ((static_cast<unsigned char>(value[i + 1]) & 0x3FU) << 12) |
                  ((static_cast<unsigned char>(value[i + 2]) & 0x3FU) << 6) |
                  (static_cast<unsigned char>(value[i + 3]) & 0x3FU);
      width = 4;
    } else if (byte >= 0xE0 && i + 2 < value.size()) {
      codepoint = ((byte & 0x0FU) << 12) |
                  ((static_cast<unsigned char>(value[i + 1]) & 0x3FU) << 6) |
                  (static_cast<unsigned char>(value[i + 2]) & 0x3FU);
      width = 3;
    } else if (byte >= 0xC0 && i + 1 < value.size()) {
      codepoint = ((byte & 0x1FU) << 6) |
                  (static_cast<unsigned char>(value[i + 1]) & 0x3FU);
      width = 2;
    }
    i += width;
    switch (codepoint) {
      case '"':
        out.append("\\\"");
        continue;
      case '\\':
        out.append("\\\\");
        continue;
      case '\b':
        out.append("\\b");
        continue;
      case '\f':
        out.append("\\f");
        continue;
      case '\n':
        out.append("\\n");
        continue;
      case '\r':
        out.append("\\r");
        continue;
      case '\t':
        out.append("\\t");
        continue;
      default:
        break;
    }
    if (width == 1 && codepoint >= 0x20 && codepoint != '<' &&
        codepoint != '>' && codepoint != '&' && codepoint != '\'') {
      out.push_back(static_cast<char>(codepoint));
    } else if (codepoint > 0xFFFF) {
      const std::uint32_t high = 0xD7C0U + (codepoint >> 10);
      const std::uint32_t low = 0xDC00U + (codepoint & 0x3FFU);
      AppendCodepoint(out, high);
      AppendCodepoint(out, low);
    } else {
      AppendCodepoint(out, codepoint);
    }
  }
  out.push_back('"');
}

// The template renders each definition through Jinja's `tojson`, which is
// Python json.dumps(sort_keys=True) with the htmlsafe escapes, ", " between
// items and ": " after keys. The definition parses to the complete tool
// object, so its keys and every nested member sort identically.
// NOLINTNEXTLINE(misc-no-recursion)
void AppendPythonJsonValue(std::string& out, const json::Value& value) {
  switch (value.type()) {
    case json::Value::Type::kNull:
      out.append("null");
      break;
    case json::Value::Type::kBool:
      out.append(value.as_bool() ? "true" : "false");
      break;
    case json::Value::Type::kNumber:
      out.append(value.dump());
      break;
    case json::Value::Type::kString:
      AppendPythonJsonString(out, value.str());
      break;
    case json::Value::Type::kArray: {
      out.push_back('[');
      const auto& items = value.items();
      for (std::size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
          out.append(", ");
        }
        AppendPythonJsonValue(out, items[i]);
      }
      out.push_back(']');
      break;
    }
    case json::Value::Type::kObject: {
      std::vector<json::Value::Member> members(value.members());
      std::stable_sort(members.begin(), members.end(),
                       [](const auto& left, const auto& right) {
                         return left.first < right.first;
                       });
      out.push_back('{');
      for (std::size_t i = 0; i < members.size(); ++i) {
        if (i != 0) {
          out.append(", ");
        }
        AppendPythonJsonString(out, members[i].first);
        out.append(": ");
        AppendPythonJsonValue(out, members[i].second);
      }
      out.push_back('}');
      break;
    }
  }
}

// The complete tool object for one request tool: the validated HTTP
// definition when present, otherwise the OpenAI wrapper built from the
// structured fields.
[[nodiscard]] json::Value ToolDefinition(const tokenization::ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    try {
      return json::parse(tool.definition_json);
    } catch (const std::exception&) {
      // A malformed envelope never reaches the prompt as raw text; fall
      // back to the structured fields like the request validator would.
    }
  }
  json::Value function = json::Value::object();
  function["name"] = tool.name;
  function["description"] = tool.description;
  json::Value parameters = json::Value::object();
  if (!tool.parameters_json.empty()) {
    try {
      parameters = json::parse(tool.parameters_json);
    } catch (const std::exception&) {
      parameters = json::Value::object();
    }
  }
  function["parameters"] = std::move(parameters);
  json::Value wrapper = json::Value::object();
  wrapper["type"] = "function";
  wrapper["function"] = std::move(function);
  return wrapper;
}

}  // namespace

std::string RenderToolsSection(std::span<const tokenization::ChatTool> tools) {
  std::string out;
  out.append(kToolsBegin);
  for (const auto& tool : tools) {
    out.push_back('\n');
    AppendPythonJsonValue(out, ToolDefinition(tool));
  }
  out.append(kToolsEnd);
  return out;
}

void AppendToolCall(std::string& out,
                    const tokenization::ChatMessage::ToolCall& call) {
  out.append(kToolCallStart).append(call.name);
  // The template writes the name flush, then a `{% set %}` block tag
  // rendered without a whitespace-control dash, so the newline and
  // indentation before that tag are part of every call frame.
  out.append("\n                ");
  for (const auto& argument : call.arguments) {
    out.append(kToolArgKeyStart)
        .append(argument.name)
        .append(kToolArgKeyEnd)
        .append("\n")
        .append(kToolArgValueStart);
    out.append(argument.value);
    out.append(kToolArgValueEnd);
  }
  out.append("\n").append(kToolCallEnd);
}

void AppendToolResponse(std::string& out, std::string_view content) {
  out.append("\n").append(kToolResponseStart).append("\n");
  out.append(content);
  out.append("\n").append(kToolResponseEnd);
}

}  // namespace gufo::models::ling3_0
