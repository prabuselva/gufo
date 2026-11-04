// Gemma 4 dialect name helpers, included verbatim inside the anonymous
// namespace of src/core/json_constraint.cpp.
// Gemma 4 call arguments: `<|"|>`-delimited raw strings, bare keys.
constexpr std::string_view kGemmaQuote = "<|\"|>";

// A bare Gemma 4 key ends at its first ':' and may not hold structure bytes;
// its parser trims surrounding whitespace and reads `<|"|>` as a quoted key.
// Other names use the quoted form.
bool BareGemmaKey(std::string_view name) {
  constexpr std::string_view kSpace = " \t\r\n\f\v";
  return !name.empty() && name.find_first_of(":{}[],") == std::string::npos &&
         kSpace.find(name.front()) == std::string_view::npos &&
         kSpace.find(name.back()) == std::string_view::npos &&
         !name.starts_with(kGemmaQuote.substr(0, 1));
}
