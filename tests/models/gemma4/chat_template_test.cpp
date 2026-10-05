#include "src/models/gemma4/chat_template.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"

namespace {

using gufo::models::gemma4::ChatMessage;
using gufo::models::gemma4::ChatTemplateOptions;
using gufo::models::gemma4::ChatTool;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

std::string Sha(std::string_view value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(value.data());
  return gufo::crypto::Sha256Hex({bytes, value.size()});
}

ChatMessage Msg(std::string role, std::string content,
                std::string reasoning = {}) {
  ChatMessage message;
  message.role = std::move(role);
  message.content = std::move(content);
  message.reasoning_content = std::move(reasoning);
  return message;
}

void Check(std::string_view name, const std::string& rendered,
           std::string_view expected) {
  Expect(Sha(rendered) == expected, name);
}

ChatTool WeatherTool() {
  ChatTool tool;
  tool.name = "get_weather";
  tool.description = "Get weather";
  tool.parameters_json =
      R"({"type":"object","properties":{"city":{"type":"string","description":"City"}},"required":["city"]})";
  return tool;
}

void TestConversationRenderers() {
  const std::vector<ChatMessage> user{Msg("user", "Hello")};
  const std::vector<ChatMessage> sys_user{Msg("system", "Be concise."),
                                          Msg("user", "Hi")};
  const std::vector<ChatMessage> history{Msg("user", "One"),
                                         Msg("assistant", "Red", "think red"),
                                         Msg("user", "Two")};

  ChatTemplateOptions options;
  Check("user_only_no_think", RenderChat(user, options),
        "c46ffe61b7c7100a2e4fea2d1599aea099683738eb2e5312f97fc8c1e812a737");
  options.enable_thinking = true;
  Check("user_only_think", RenderChat(user, options),
        "5898358b457c78b0f8654923c0786e8fc923764dc8acf0dfa01a256a2d8a133c");

  options.enable_thinking = false;
  Check("sys_user_no_think", RenderChat(sys_user, options),
        "77107abe3a0c07dc6d4f34ecf2da695601c9db3d131338baec529a3a968eb702");
  options.enable_thinking = true;
  Check("sys_user_think", RenderChat(sys_user, options),
        "6845a6594abf6077094821a7cea3efef143ea20c397bf273943bb30cee4a3dd0");

  // Historical reasoning is dropped unless the turn also carries tool calls.
  Check("history_drop_think", RenderChat(history, options),
        "2c078df93a66356750943150021df0e9ff07f86f4a1a5d881411f862f07c39f3");
  options.preserve_thinking = true;
  Check("history_preserve_think", RenderChat(history, options),
        "2c078df93a66356750943150021df0e9ff07f86f4a1a5d881411f862f07c39f3");
}

void TestToolDeclarationsAndLoop() {
  const std::vector<ChatTool> tools{WeatherTool()};
  const std::vector<ChatMessage> sys_user{Msg("system", "Be concise."),
                                          Msg("user", "Hi")};
  ChatTemplateOptions options;
  Check("tools_no_think", RenderChat(sys_user, tools, options),
        "6dbeabf3c8ed9c225093da83ba3ccc54283585752f626a89fd700ed87e302aa3");
  options.enable_thinking = true;
  Check("tools_think", RenderChat(sys_user, tools, options),
        "55df5c7988d9a20a0b6c017b0b3c974475c631be3f175da24aa79f0260b70597");

  ChatMessage assistant = Msg("assistant", "");
  ChatMessage::ToolCall call;
  call.name = "get_weather";
  call.id = "c1";
  call.arguments.push_back({"city", "Rome", true});
  assistant.tool_calls.push_back(call);
  ChatMessage response = Msg("tool", "21C");
  response.tool_call_id = "c1";
  const std::vector<ChatMessage> loop{Msg("user", "Weather?"), assistant,
                                      response};
  Check("tool_loop", RenderChat(loop, tools, options),
        "98712c0e3fae49ead9b953fe92732c3d8c9d1a521bf4235b4ca3508615bd46fa");
}

void TestContinuation() {
  const std::vector<ChatMessage> messages{
      Msg("user", "Go"), Msg("assistant", "A"), Msg("assistant", "B")};
  ChatTemplateOptions options;
  options.enable_thinking = true;
  Check("continuation", RenderChat(messages, options),
        "20f929c8f9fc9e73499a0dfbb86640591da4816ff3e9af25936f8ae8a6eeb199");
}

void TestGenerationPrompt() {
  Expect(gufo::models::gemma4::GenerationPrompt(false) ==
             "<|turn>model\n<|channel>thought\n<channel|>",
         "Non-thinking generation prompt opens an empty thought channel");
  Expect(gufo::models::gemma4::GenerationPrompt(true) == "<|turn>model\n",
         "Thinking generation prompt leaves the channel open");
}

void TestArtifactTemplateHash() {
  const char* path = std::getenv("GUFO_GEMMA4_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "gemma4.chat_template: skipping artifact hash check "
                 "(GUFO_GEMMA4_GGUF unset)\n";
    return;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "Gemma-4 artifact opens");
  const auto stored = reader->GetMetadataString("tokenizer.chat_template");
  Expect(stored.has_value(), "artifact carries a chat template");
  Expect(Sha(*stored) == gufo::models::gemma4::ArtifactTemplateSha256(),
         "artifact template matches the pinned hash");
  Expect(gufo::models::gemma4::ValidateGgufTemplate(*reader, &error),
         "pinned artifact template is accepted");
}

}  // namespace

int main() {
  TestConversationRenderers();
  TestToolDeclarationsAndLoop();
  TestContinuation();
  TestGenerationPrompt();
  TestArtifactTemplateHash();
  std::cout << "Gemma-4 chat template tests passed\n";
  return 0;
}