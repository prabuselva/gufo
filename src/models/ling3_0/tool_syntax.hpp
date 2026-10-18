#ifndef GUFO_MODELS_LING3_FLASH_TOOL_SYNTAX_HPP_
#define GUFO_MODELS_LING3_FLASH_TOOL_SYNTAX_HPP_

#include <span>
#include <string>
#include <string_view>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::ling3_0 {

// Delimiters of one native Ling 3.0 Flash tool call, exactly as the pinned
// Bailing V3 chat template renders it: the call opens with kToolCallStart
// followed immediately by the function name, then one kToolArgKey frame per
// argument name and an kToolArgValue frame on the following line, and closes
// with a newline and kToolCallEnd.
inline constexpr std::string_view kToolCallStart = "<tool_call>";
inline constexpr std::string_view kToolCallEnd = "</tool_call>";
inline constexpr std::string_view kToolArgKeyStart = "<arg_key>";
inline constexpr std::string_view kToolArgKeyEnd = "</arg_key>";
inline constexpr std::string_view kToolArgValueStart = "<arg_value>";
inline constexpr std::string_view kToolArgValueEnd = "</arg_value>";

// Frame of one tool-result turn: the content rides between the response
// delimiters, each on its own line, inside the OBSERVATION turn.
inline constexpr std::string_view kToolResponseStart = "<tool_response>";
inline constexpr std::string_view kToolResponseEnd = "</tool_response>";

/// The tools section the system header carries: the instruction block the
/// pinned template spells, with one tool definition JSON per line inside the
/// tools envelope. Definitions render as the template's `tojson` filter does:
/// sorted keys everywhere, ", "/": " separators and lowercase \uXXXX escapes
/// for non-ASCII, control characters and <, >, &, '.
[[nodiscard]] std::string RenderToolsSection(
    std::span<const tokenization::ChatTool> tools);

/// Appends one assistant tool call in the native rendering: the name opens
/// the call, then the newline and indentation the template leaves before its
/// undashed `{% set %}` block, then the arguments in order, and the call
/// closed by a newline and kToolCallEnd. The leading newline the template
/// inserts between text and the call, or between calls, is not part of this
/// frame.
void AppendToolCall(std::string& out,
                    const tokenization::ChatMessage::ToolCall& call);

/// Appends one tool-result frame: a newline, the response frame opening on
/// its own line, the content, a newline and the closing delimiter.
void AppendToolResponse(std::string& out, std::string_view content);

}  // namespace gufo::models::ling3_0

#endif  // GUFO_MODELS_LING3_FLASH_TOOL_SYNTAX_HPP_
