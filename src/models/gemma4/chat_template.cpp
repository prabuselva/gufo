#include "src/models/gemma4/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace gufo::models::gemma4 {
namespace {

using json::Value;

constexpr std::string_view kBos = "<bos>";
constexpr std::string_view kTurnOpen = "<|turn>";
constexpr std::string_view kTurnClose = "<turn|>\n";
constexpr std::string_view kChannelOpen = "<|channel>";
constexpr std::string_view kChannelClose = "<channel|>";
constexpr std::string_view kQuote = "<|\"|>";

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

std::string_view Trim(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && IsSpace(text[begin])) {
    ++begin;
  }
  while (end > begin && IsSpace(text[end - 1])) {
    --end;
  }
  return text.substr(begin, end - begin);
}

std::string Upper(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return out;
}

// Mirrors the template's strip_thinking macro: drop any <|channel>...<channel|>
// span embedded in visible model content, then trim.
std::string StripThinking(std::string_view text) {
  std::string result;
  std::size_t pos = 0;
  while (true) {
    const std::size_t close = text.find(kChannelClose, pos);
    std::string_view part = (close == std::string_view::npos)
                                ? text.substr(pos)
                                : text.substr(pos, close - pos);
    const std::size_t open = part.find(kChannelOpen);
    if (open != std::string_view::npos) {
      part = part.substr(0, open);
    }
    result.append(part);
    if (close == std::string_view::npos) {
      break;
    }
    pos = close + kChannelClose.size();
  }
  return std::string(Trim(result));
}

std::string NumberToString(double value) {
  if (value == std::floor(value) && std::fabs(value) < 1e15) {
    return std::to_string(static_cast<long long>(value));
  }
  std::string out = std::to_string(value);
  const std::size_t dot = out.find('.');
  if (dot != std::string::npos) {
    std::size_t last = out.size();
    while (last > dot + 1 && out[last - 1] == '0') {
      --last;
    }
    out.erase(last);
  }
  return out;
}

std::vector<std::pair<std::string, const Value*>> SortedMembers(
    const Value& object) {
  std::vector<std::pair<std::string, const Value*>> members;
  members.reserve(object.members().size());
  for (const auto& kv : object.members()) {
    members.emplace_back(kv.first, &kv.second);
  }
  std::sort(
      members.begin(), members.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return members;
}

std::string FormatArgument(const Value& argument, bool escape_keys);

// Emits a separating comma when a field was already written, otherwise marks
// the first field. Matches the template's `{%- if add_comma %},{%- else -%}`
// idiom that both separates fields and records presence.
void CommaOrMark(std::string& out, bool& add_comma) {
  if (add_comma) {
    out += ',';
  } else {
    add_comma = true;
  }
}

std::string FormatParameters(const Value& properties, bool filter_keys);

std::string FormatItems(const Value& items) {
  std::string out;
  bool found_first = false;
  for (const auto& [key, value] : SortedMembers(items)) {
    if (value->is_null()) {
      continue;
    }
    if (found_first) {
      out += ',';
    }
    found_first = true;
    if (key == "properties") {
      out += "properties:{";
      if (value->is_object()) {
        out += FormatParameters(*value, false);
      }
      out += "}";
    } else if (key == "required") {
      out += "required:[";
      bool first = true;
      for (const auto& item : value->items()) {
        if (!first) {
          out += ',';
        }
        first = false;
        out += kQuote;
        out += item.str();
        out += kQuote;
      }
      out += "]";
    } else if (key == "type") {
      out += "type:";
      if (value->is_string()) {
        out += FormatArgument(Value(Upper(value->str())), true);
      } else {
        Value uppered = Value::array();
        for (const auto& item : value->items()) {
          uppered.push_back(Value(Upper(item.str())));
        }
        out += FormatArgument(uppered, true);
      }
    } else {
      out += key;
      out += ':';
      out += FormatArgument(*value, true);
    }
  }
  return out;
}

std::string FormatParameters(const Value& properties, bool filter_keys) {
  static const std::vector<std::string> kStandard = {
      "description", "type", "properties", "required", "nullable"};
  std::string out;
  bool found_first = false;
  for (const auto& [key, value] : SortedMembers(properties)) {
    if (filter_keys &&
        std::find(kStandard.begin(), kStandard.end(), key) != kStandard.end()) {
      continue;
    }
    if (found_first) {
      out += ',';
    }
    found_first = true;
    out += key;
    out += ":{";
    bool add_comma = false;
    const std::string type_upper = Upper(value->member_str("type"));
    if (!value->member_str("description").empty()) {
      out += "description:";
      out += kQuote;
      out += value->member_str("description");
      out += kQuote;
      add_comma = true;
    }
    if (type_upper == "STRING") {
      const Value* enum_value = value->find("enum");
      if (enum_value != nullptr && !enum_value->empty()) {
        CommaOrMark(out, add_comma);
        out += "enum:";
        out += FormatArgument(*enum_value, true);
      }
    } else if (type_upper == "ARRAY") {
      const Value* items = value->find("items");
      if (items != nullptr && items->is_object() && !items->empty()) {
        CommaOrMark(out, add_comma);
        out += "items:{";
        out += FormatItems(*items);
        out += "}";
      }
    }
    if (value->find("nullable") != nullptr && value->as_bool(false)) {
      CommaOrMark(out, add_comma);
      out += "nullable:true";
    }
    if (type_upper == "OBJECT") {
      const Value* nested = value->find("properties");
      if (nested != nullptr && nested->is_object()) {
        CommaOrMark(out, add_comma);
        out += "properties:{";
        out += FormatParameters(*nested, false);
        out += "}";
      } else if (value->is_object()) {
        CommaOrMark(out, add_comma);
        out += "properties:{";
        out += FormatParameters(*value, true);
        out += "}";
      }
      const Value* required = value->find("required");
      if (required != nullptr && !required->empty()) {
        CommaOrMark(out, add_comma);
        out += "required:[";
        bool first = true;
        for (const auto& item : required->items()) {
          if (!first) {
            out += ',';
          }
          first = false;
          out += kQuote;
          out += item.str();
          out += kQuote;
        }
        out += "]";
      }
    }
    CommaOrMark(out, add_comma);
    out += "type:";
    out += kQuote;
    out += type_upper;
    out += kQuote;
    out += '}';
  }
  return out;
}

std::string FormatArgument(const Value& argument, bool escape_keys) {
  if (argument.is_null()) {
    return "null";
  }
  if (argument.is_string()) {
    std::string out(kQuote);
    out += argument.str();
    out += kQuote;
    return out;
  }
  if (argument.is_bool()) {
    return argument.as_bool() ? "true" : "false";
  }
  if (argument.is_object()) {
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : SortedMembers(argument)) {
      if (!first) {
        out += ',';
      }
      first = false;
      if (escape_keys) {
        out += kQuote;
        out += key;
        out += kQuote;
      } else {
        out += key;
      }
      out += ':';
      out += FormatArgument(*value, escape_keys);
    }
    out += '}';
    return out;
  }
  if (argument.is_array()) {
    std::string out = "[";
    bool first = true;
    for (const auto& item : argument.items()) {
      if (!first) {
        out += ',';
      }
      first = false;
      out += FormatArgument(item, escape_keys);
    }
    out += ']';
    return out;
  }
  return NumberToString(argument.as_double());
}

std::string FormatFunctionDeclaration(const Value& tool_data) {
  const Value* function = tool_data.find("function");
  if (function == nullptr) {
    return {};
  }
  std::string out = "declaration:";
  out += function->member_str("name");
  out += "{description:";
  out += kQuote;
  out += function->member_str("description");
  out += kQuote;
  const Value* params = function->find("parameters");
  if (params != nullptr && !params->empty()) {
    out += ",parameters:{";
    const Value* props = params->find("properties");
    if (props != nullptr && !props->empty()) {
      out += "properties:{";
      out += FormatParameters(*props, false);
      out += "},";
    }
    const Value* required = params->find("required");
    if (required != nullptr && !required->empty()) {
      out += "required:[";
      bool first = true;
      for (const auto& item : required->items()) {
        if (!first) {
          out += ',';
        }
        first = false;
        out += kQuote;
        out += item.str();
        out += kQuote;
      }
      out += "],";
    }
    const Value* type = params->find("type");
    if (type != nullptr && !type->is_null()) {
      out += "type:";
      out += kQuote;
      out += Upper(type->str());
      out += kQuote;
      out += '}';
    }
  }
  const Value* response = function->find("response");
  if (response != nullptr) {
    out += ",response:{";
    if (!response->member_str("description").empty()) {
      out += "description:";
      out += kQuote;
      out += response->member_str("description");
      out += kQuote;
      out += ',';
    }
    if (Upper(response->member_str("type")) == "OBJECT") {
      out += "type:";
      out += kQuote;
      out += "OBJECT";
      out += kQuote;
      out += '}';
    }
  }
  out += '}';
  return std::string(Trim(out));
}

// Builds the {function:{name,description,parameters[,response]}} object the
// template consumes, preferring the complete definition when available.
Value ToolFunctionValue(const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    try {
      Value parsed = json::parse(tool.definition_json);
      if (parsed.is_object() && parsed.contains("function")) {
        return parsed;
      }
      Value wrapper = Value::object();
      wrapper.append_member("function", std::move(parsed));
      return wrapper;
    } catch (const std::exception&) {
      // Fall through to the field-based construction below.
    }
  }
  Value function = Value::object();
  function.append_member("name", Value(tool.name));
  function.append_member("description", Value(tool.description));
  if (!tool.parameters_json.empty()) {
    try {
      function.append_member("parameters", json::parse(tool.parameters_json));
    } catch (const std::exception&) {
      function.append_member("parameters", Value::object());
    }
  }
  Value wrapper = Value::object();
  wrapper.append_member("function", std::move(function));
  return wrapper;
}

std::string FormatToolResponseBlock(std::string_view tool_name,
                                    const std::string& response) {
  std::string out = "<|tool_response>response:";
  out += tool_name;
  out += "{value:";
  out += kQuote;
  out += response;
  out += kQuote;
  out += "}<tool_response|>";
  return out;
}

std::string Render(std::span<const ChatMessage> messages,
                   std::span<const ChatTool> tools,
                   const ChatTemplateOptions& options) {
  std::string out(kBos);
  std::string prev_message_type;
  std::string prev_non_tool_role;

  const bool first_is_system =
      !messages.empty() &&
      (messages[0].role == "system" || messages[0].role == "developer");

  if (options.enable_thinking || !tools.empty() || first_is_system) {
    out += kTurnOpen;
    out += "system\n";
    if (options.enable_thinking) {
      out += "<|think|>\n";
      prev_message_type = "think";
    }
    std::size_t start = 0;
    if (first_is_system) {
      out += Trim(messages[0].content);
      start = 1;
    }
    for (const ChatTool& tool : tools) {
      out += "<|tool>";
      out += FormatFunctionDeclaration(ToolFunctionValue(tool));
      out += "<tool|>";
    }
    if (!tools.empty()) {
      prev_message_type = "tool";
    }
    out += kTurnClose;
    messages = messages.subspan(start);
  }

  std::size_t last_user_idx = static_cast<std::size_t>(-1);
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == "user") {
      last_user_idx = i;
    }
  }

  for (std::size_t index = 0; index < messages.size(); ++index) {
    const ChatMessage& message = messages[index];
    if (message.role == "tool") {
      continue;
    }
    prev_message_type.clear();
    const bool is_assistant = message.role == "assistant";
    const std::string role = is_assistant ? "model" : message.role;
    const bool continue_same_model_turn =
        role == "model" && prev_non_tool_role == "assistant";
    if (!continue_same_model_turn) {
      out += kTurnOpen;
      out += role;
      out += '\n';
    }

    const bool thinking_gate =
        (index != static_cast<std::size_t>(-1) && index > last_user_idx) ||
        (options.preserve_thinking && !message.tool_calls.empty());
    if (!message.reasoning_content.empty() && thinking_gate) {
      out += kChannelOpen;
      out += "thought\n";
      out += message.reasoning_content;
      out += '\n';
      out += kChannelClose;
    }

    bool tr_out = false;
    if (!message.tool_calls.empty()) {
      for (const auto& call : message.tool_calls) {
        out += "<|tool_call>call:";
        out += call.name;
        out += '{';
        bool first = true;
        for (const auto& argument : call.arguments) {
          if (!first) {
            out += ',';
          }
          first = false;
          out += argument.name;
          out += ':';
          if (argument.is_string) {
            out += kQuote;
            out += argument.value;
            out += kQuote;
          } else {
            out += argument.value;
          }
        }
        out += "}<tool_call|>";
      }
      prev_message_type = "tool_call";
    }

    for (std::size_t k = index + 1; k < messages.size(); ++k) {
      if (messages[k].role != "tool") {
        break;
      }
      const ChatMessage& follow = messages[k];
      std::string name = follow.name.empty() ? "unknown" : follow.name;
      for (const auto& call : message.tool_calls) {
        if (!call.id.empty() && call.id == follow.tool_call_id) {
          name = call.name;
          break;
        }
      }
      out += FormatToolResponseBlock(name, follow.content);
      tr_out = true;
      prev_message_type = "tool_response";
    }

    std::string content;
    if (role == "model") {
      content = StripThinking(message.content);
    } else {
      content = std::string(Trim(message.content));
    }
    out += content;
    const bool has_content = !Trim(content).empty();

    std::string_view next_role;
    bool found_next = false;
    for (std::size_t j = index + 1; j < messages.size(); ++j) {
      if (!found_next && messages[j].role != "tool") {
        next_role = messages[j].role;
        found_next = true;
      }
    }
    const bool continues_into_next = role == "model" &&
                                     next_role == "assistant" &&
                                     (message.tool_calls.empty() || tr_out);

    if (prev_message_type == "tool_call" && !tr_out) {
      out += "<|tool_response>";
    } else if (continues_into_next) {
      // Continue the same model turn without a closing token.
    } else if (!(tr_out && !has_content && !found_next)) {
      out += kTurnClose;
    }
    prev_non_tool_role = message.role;
  }

  if (options.add_generation_prompt) {
    if (prev_message_type != "tool_response" &&
        prev_message_type != "tool_call") {
      out += kTurnOpen;
      out += "model\n";
      if (!options.enable_thinking) {
        out += kChannelOpen;
        out += "thought\n";
        out += kChannelClose;
      }
    } else if (prev_message_type == "tool_response" &&
               options.enable_thinking) {
      out += kChannelOpen;
      out += "thought\n";
    }
  }
  return out;
}

}  // namespace

std::string_view GenerationPrompt(bool enable_thinking) {
  if (enable_thinking) {
    return "<|turn>model\n";
  }
  return "<|turn>model\n<|channel>thought\n<channel|>";
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       const ChatTemplateOptions& options) {
  return Render(messages, {}, options);
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  return Render(messages, tools, options);
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  const auto stored = reader.GetMetadataString("tokenizer.chat_template");
  if (!stored.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Gemma-4 artifact has no tokenizer.chat_template";
    }
    return false;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(stored->data());
  const std::string hash = crypto::Sha256Hex({bytes, stored->size()});
  if (hash != ArtifactTemplateSha256()) {
    if (error_msg != nullptr) {
      *error_msg = "unsupported Gemma-4 chat template hash " + hash;
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::gemma4