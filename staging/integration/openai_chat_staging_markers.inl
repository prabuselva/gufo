// Staging model tool-call marker and closer sets for openai_chat.cpp,
// included inside its anonymous namespace ahead of ToolMarkers/ToolClosers.
// not compiled standalone.

#if defined(GUFO_STAGING_MODELS)
// Gemma 4 opens a call with the template's own marker token.
constexpr std::array<std::string_view, 1> kGemmaMarkers{
    models::gemma4::kToolCallStart};
#endif
// Ling 3.0 Flash writes the name directly after this opener, so the
// dialect needs its own set even though the opener text matches Qwen.
constexpr std::array<std::string_view, 1> kLingMarkers{"<tool_call>"};
// Ling 3.0 Flash frames each argument with its own closer; nothing else in
// its output is framing.
constexpr std::array<std::string_view, 3> kLingClosers{
    "</arg_value>",
    "</tool_call>",
    "</invoke>",
};
