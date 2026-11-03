// Gemma 4 dialect member bodies of JsonConstraintCompiler, defined
// out-of-line in src/core/json_constraint.cpp right after the class so the
// shared file only carries the four private declarations.
std::uint32_t JsonConstraintCompiler::GemmaString(
    const json::Value& predicate) {
  const auto raw =
      JsonSchemaLexeme::RawString(predicate, std::string(kGemmaQuote));
  auto text = Lexeme(raw);
  if (raw->Check("").complete)
    text = Optional(text);
  return Seq({Literal(kGemmaQuote), text, Literal(kGemmaQuote)});
}
// A dynamic bare key other than the declared names.
std::uint32_t JsonConstraintCompiler::GemmaKey(
    const std::set<std::string>& excluded) {
  std::string pattern = "^";
  if (!excluded.empty()) {
    pattern += "(?!(?:";
    bool separator = false;
    for (const auto& name : excluded) {
      if (separator)
        pattern += '|';
      separator = true;
      pattern += RegexLiteral(name);
    }
    pattern += ")$)";
  }
  pattern += "[^:{}\\[\\],<\\s](?:[^:{}\\[\\],]*[^:{}\\[\\],\\s])?$";
  auto predicate = json::Value::object();
  predicate["pattern"] = std::move(pattern);
  return Lexeme(JsonSchemaLexeme::RawString(predicate, ":"));
}
std::uint32_t JsonConstraintCompiler::KeyLiteral(const std::string& name) {
  if (!gemma_)
    return Literal(json::Value(name).dump());
  if (BareGemmaKey(name))
    return Literal(name);
  // The parser also reads `<|"|>`-quoted keys, as declarations render them.
  if (name.find(kGemmaQuote) != std::string_view::npos)
    Invalid("property name cannot be a Gemma 4 key");
  return Literal(std::string(kGemmaQuote) + name + std::string(kGemmaQuote));
}
// Gemma 4 renders object keys in sorted order at every depth.
std::vector<std::pair<std::string, const json::Value*>>
JsonConstraintCompiler::Members(const json::Value& object) const {
  std::vector<std::pair<std::string, const json::Value*>> members;
  for (const auto& [name, value] : object.members())
    members.emplace_back(name, &value);
  if (gemma_)
    std::ranges::sort(members, {}, &decltype(members)::value_type::first);
  return members;
}
