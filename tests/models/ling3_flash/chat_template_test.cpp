#include "src/models/ling3_flash/chat_template.hpp"

#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/chat_template.hpp"

namespace {

using gufo::models::ling3_flash::ChatTemplateOptions;
using gufo::models::ling3_flash::RenderChat;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using gufo::tokenization::ChatTool;

void ExpectRender(const std::string& label, const std::string& actual,
                  std::string_view expected) {
  if (actual == expected) {
    return;
  }
  std::size_t i = 0;
  while (i < actual.size() && i < expected.size() && actual[i] == expected[i]) {
    ++i;
  }
  std::cerr << "Assertion failed: " << label << " differs at byte " << i << '\n'
            << "  actual  ..." << actual.substr(i, 60) << "'\n'  expected..."
            << std::string(expected.substr(i, 60)) << "'\n";
  std::exit(1);
}

// Goldens are generated with Jinja2 from the pinned reference template
// (src/models/ling3_flash/reference/chat_template.jinja), the same artifact
// the checkpoint ships, rendering `tojson` as htmlsafe json.dumps with
// sort_keys=True. They pin every whitespace artifact of the official
// template, including the stray indentation left inside tag blocks.
#include "tests/models/ling3_flash/chat_template_goldens.inc"

std::vector<ChatTool> WeatherTools() {
  return {
      ChatTool{
          .name = "get_weather",
          .description = "Weather “plain” & <markup> for a city",
          .parameters_json =
              R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]})",
          .definition_json =
              R"({"type":"function","function":{"name":"get_weather","description":"Weather “plain” & <markup> for a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}})",
      },
  };
}

std::vector<ChatTool> AllTools() {
  auto tools = WeatherTools();
  tools.push_back(ChatTool{
      .name = "send_mail",
      .description = "Send it — café and 'quotes'",
      .parameters_json =
          R"({"type":"object","properties":{"body":{"type":"string"}}})",
      .definition_json =
          R"({"name":"send_mail","description":"Send it — café and 'quotes'","parameters":{"type":"object","properties":{"body":{"type":"string"}}}})",
  });
  return tools;
}

ChatMessage Message(ChatRole role, std::string content) {
  ChatMessage message;
  message.role = role;
  message.content = std::move(content);
  return message;
}

void TestThinkingAndFraming() {
  const ChatTemplateOptions on;
  const std::vector<ChatMessage> simple_messages = {
      Message(ChatRole::kSystem, "You are helpful."),
      Message(ChatRole::kUser, "Hi")};
  const auto simple = RenderChat(simple_messages, on);
  ExpectRender("simple", simple.value_or(""), kSimple);

  ChatTemplateOptions off;
  off.enable_thinking = false;
  const std::vector<ChatMessage> hi = {Message(ChatRole::kUser, "Hi")};
  const auto no_think = RenderChat(hi, off);
  ExpectRender("thinking off", no_think.value_or(""), kThinkingOff);

  const std::vector<ChatMessage> directive_messages = {
      Message(ChatRole::kSystem, "Be terse. detailed thinking off"),
      Message(ChatRole::kUser, "Hi")};
  const auto directive = RenderChat(directive_messages, on);
  ExpectRender("system directive", directive.value_or(""),
               kSystemCarriesDirective);
}

void TestToolsHeader() {
  const auto tools = WeatherTools();
  const std::vector<ChatMessage> history = {
      Message(ChatRole::kSystem, "You are helpful."),
      Message(ChatRole::kUser, "Weather in Paris?")};
  const auto rendered = RenderChat(history, ChatTemplateOptions{}, true, tools);
  ExpectRender("tools header", rendered.value_or(""), kToolsHeader);
}

void TestToolCallHistory() {
  const auto tools = AllTools();
  ChatMessage first_call = Message(ChatRole::kAssistant, "");
  first_call.thought = "Looking it up.";
  first_call.tool_calls.push_back({
      .id = "call_0",
      .name = "get_weather",
      .arguments =
          {
              {.name = "city", .value = "Paris"},
              {.name = "days", .value = "3", .is_string = false},
              {.name = "now", .value = "true", .is_string = false},
          },
  });
  ChatMessage second_call = Message(ChatRole::kAssistant, "Warm.");
  second_call.tool_calls.push_back({
      .id = "call_1",
      .name = "send_mail",
      .arguments =
          {
              {.name = "body", .value = "café — <b>ok</b>"},
              {.name = "urgent", .value = "false", .is_string = false},
          },
  });
  const std::vector<ChatMessage> history = {
      Message(ChatRole::kSystem, "You are helpful."),
      Message(ChatRole::kUser, "Weather in Paris?"),
      first_call,
      Message(ChatRole::kTool, R"({"temp": 21})"),
      second_call,
      Message(ChatRole::kTool, "sent & done"),
      Message(ChatRole::kUser, "thanks")};
  const auto rendered = RenderChat(history, ChatTemplateOptions{}, true, tools);
  ExpectRender("tool call history", rendered.value_or(""), kMultiTurn);
}

void TestConsecutiveToolTurns() {
  const auto tools = WeatherTools();
  ChatMessage call = Message(ChatRole::kAssistant, "");
  call.tool_calls.push_back(
      {.id = "call_0", .name = "get_weather", .arguments = {}});
  const std::vector<ChatMessage> history = {
      Message(ChatRole::kUser, "Weather?"), call, Message(ChatRole::kTool, "a"),
      Message(ChatRole::kTool, "b")};
  const auto rendered =
      RenderChat(history, ChatTemplateOptions{}, false, tools);
  ExpectRender("consecutive tool turns", rendered.value_or(""),
               kConsecutiveToolTurns);
}

}  // namespace

int main() {
  TestThinkingAndFraming();
  TestToolsHeader();
  TestToolCallHistory();
  TestConsecutiveToolTurns();
  std::cout << "All Ling 3.0 Flash chat template tests passed\n";
  return 0;
}
