// Staging model tool-call parsers, included inside the openai_chat.cpp
// anonymous namespace; not compiled standalone.

bool ValidToolName(std::string_view name) {
  // The Chat Completions API admits [a-zA-Z0-9_-]; the same validity keeps a
  // Ling opener mentioned in prose from being parsed as a call.
  return !name.empty() && std::ranges::all_of(name, [](char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || byte == '_' || byte == '-';
  });
}

// Hold a possible closing delimiter until its next byte disambiguates it.
std::string_view BeforePartialDelimiter(std::string_view value,
                                        std::string_view delimiter) {
  for (auto size = std::min(value.size(), delimiter.size()); size > 0; --size)
    if (delimiter.starts_with(value.substr(value.size() - size)))
      return value.substr(0, value.size() - size);
  return value;
}

// Ling 3.0 Flash names its call directly after the opener and frames each
// argument as `<arg_key>key</arg_key><arg_value>value</arg_value>`, closed
// by `</tool_call>`. The structure mirrors ParseQwenCalls: declared names,
// schema-typed values and tentative streaming only for string arguments.
void ParseLingCalls(
    std::string_view text, std::span<const tokenization::ChatTool> tools,
    std::vector<ParsedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr,
    bool require_schema = false, bool partial = false) {
  constexpr std::string_view start = "<tool_call>";
  constexpr std::string_view end = "</tool_call>";
  constexpr std::string_view key_open = "<arg_key>";
  constexpr std::string_view key_close = "</arg_key>";
  constexpr std::string_view value_open = "<arg_value>";
  constexpr std::string_view value_close = "</arg_value>";
  QuoteTracker call_quotes;
  call_quotes.Reset(text);
  std::size_t cursor = 0;
  while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
    if (call_quotes.QuotedAt(cursor) ||
        (partial && call_quotes.UnclosedAt(cursor))) {
      cursor += start.size();
      continue;
    }
    const auto marker_begin = cursor;
    const auto begin = cursor + start.size();
    cursor = begin;  // A malformed call may be followed by a valid call.
    auto body = text.substr(begin);
    const auto consume = [&](std::string_view tag) {
      body = Trim(body);
      if (!body.starts_with(tag))
        return false;
      body.remove_prefix(tag.size());
      return true;
    };
    ParsedToolCall call;
    bool complete = false;
    bool tentative = false;
    // The name occupies the text between the opener and the first newline
    // (instruction framing) or tag (history framing, where the template
    // concatenates the name and `<arg_key>` without a newline).
    const auto line_end = body.find('\n');
    const auto tag_end = body.find('<');
    const auto name_end =
        tag_end != std::string_view::npos &&
                (line_end == std::string_view::npos || tag_end < line_end)
            ? tag_end
            : line_end;
    if (name_end == std::string_view::npos)
      continue;  // A streamed name stays invisible until its line ends.
    call.name = std::string(Trim(body.substr(0, name_end)));
    // A newline is framing; a `<` is the first tag's own character.
    body.remove_prefix(name_end + (body[name_end] == '\n' ? 1 : 0));
    if (!ValidToolName(call.name))
      continue;
    std::optional<json::Value> schema;
    for (const auto& tool : tools) {
      if (tool.name == call.name) {
        schema = TryParseJson(tool.parameters_json);
        break;
      }
    }
    tentative = partial && schema && !require_schema &&
                !call_quotes.UnclosedAt(marker_begin);
    const auto* object = schema ? ResolveToolSchema(*schema, *schema) : nullptr;
    const auto* properties = object ? object->find("properties") : nullptr;
    bool valid = true;
    while (consume(key_open)) {
      const auto key_end = body.find(key_close);
      if (key_end == std::string_view::npos) {
        valid = false;
        break;
      }
      // Preserve the schema's exact key, as in the Qwen parser: trimming a
      // declared key with surrounding spaces would corrupt the arguments.
      const std::string spelled(body.substr(0, key_end));
      const std::string name =
          properties && properties->is_object() && properties->contains(spelled)
              ? spelled
              : std::string(Trim(spelled));
      body.remove_prefix(key_end + key_close.size());
      if (name.empty() || !consume(value_open)) {
        valid = false;
        break;
      }
      auto close = body.find(value_close);
      if (close == std::string_view::npos) {
        const auto* property = properties ? properties->find(name) : nullptr;
        if (tentative && (!property || StringOnly(*schema, *property)))
          PartialStringArgument(call, name,
                                BeforePartialDelimiter(body, value_close));
        // Never recover a nested example from inside an unfinished value.
        cursor = text.size();
        valid = false;
        break;
      }
      const auto value = body.substr(0, close);
      const auto* property = properties ? properties->find(name) : nullptr;
      const bool string_allowed =
          !property ||
          SchemaAccepts(*schema, *property, json::Value(std::string(value)));
      // Prefer text if the schema permits it; parsing an ambiguous scalar
      // as JSON would silently change a caller's declared string type.
      std::optional<json::Value> typed;
      if (string_allowed && property) {
        typed = TryParseJson(Trim(value));
        if (!typed || typed->is_string() ||
            !SchemaAccepts(*schema, *property, *typed))
          typed.reset();
      }
      const bool is_string = string_allowed && !typed;
      std::string raw(is_string ? value : Trim(value));
      if (!is_string && !typed) {
        auto parsed = TryParseJson(raw);
        if (parsed && !SchemaAccepts(*schema, *property, *parsed))
          parsed.reset();
        if (!parsed) {
          auto converted = PythonLiteralsToJson(raw);
          auto alternative = TryParseJson(converted);
          if (alternative && SchemaAccepts(*schema, *property, *alternative)) {
            raw = std::move(converted);
            parsed = std::move(alternative);
          }
        }
        if (!parsed) {
          valid = false;
          break;
        }
      }
      // Models sometimes repeat an argument. An identical copy is harmless;
      // conflicting copies leave no safe choice.
      const auto previous = std::ranges::find(
          call.arguments, name, [](const auto& arg) { return arg.name; });
      if (previous == call.arguments.end()) {
        call.arguments.push_back(
            {.name = name, .value = std::move(raw), .is_string = is_string});
      } else if (previous->value != raw || previous->is_string != is_string) {
        valid = false;
        break;
      }
      body.remove_prefix(close + value_close.size());
    }
    complete = valid && consume(end);
    if (complete && !tools.empty() &&
        std::ranges::none_of(
            tools, [&](const auto& tool) { return tool.name == call.name; }))
      complete = false;
    if (complete && (require_schema || call_quotes.UnclosedAt(marker_begin))) {
      // An unfinished quotation is ambiguous. Keep only a declared call
      // whose complete arguments satisfy its schema.
      const auto tool =
          std::ranges::find(tools, call.name, &tokenization::ChatTool::name);
      if (tool == tools.end()) {
        complete = false;
      } else {
        try {
          const auto grammar = sampling::JsonConstraint::Compile(
              json::parse(tool->parameters_json), false);
          auto state = grammar->Start();
          for (const unsigned char byte : ArgumentsJson(call.arguments))
            state = grammar->Advance(state, byte);
          complete = grammar->Complete(state);
        } catch (const std::exception&) {
          complete = false;
        }
      }
    }
    if (complete || tentative) {
      call.id = RandomId("call_");
      if (!complete && !call.partial_arguments)
        call.partial_arguments = ArgumentsPrefix(call);
      calls->push_back(std::move(call));
      if (!complete)
        break;
      cursor = static_cast<std::size_t>(body.data() - text.data());
      // Argument bytes belong to the call, not to prose quotation state.
      call_quotes.Reset(text, cursor);
      if (spans)
        spans->emplace_back(marker_begin, cursor);
    }
  }
}

#if defined(GUFO_STAGING_MODELS)
// The declared tool a Gemma 4 call body addresses: `call:NAME{`.
[[nodiscard]] bool GemmaToolDeclared(
    std::string_view body, std::span<const tokenization::ChatTool> tools) {
  constexpr std::string_view prefix = "call:";
  if (!body.starts_with(prefix))
    return false;
  const auto rest = body.substr(prefix.size());
  const auto end = rest.find('{');
  if (end == std::string_view::npos)
    return false;
  const auto name = rest.substr(0, end);
  for (const auto& tool : tools)
    if (tool.name == name)
      return true;
  return false;
}

// Native Gemma 4 calls: `<|tool_call>call:NAME{...}<tool_call|>` blocks under
// the mandatory thought channel. Grammar guarantees the framing; this splits
// complete blocks from prose. An unclosed block is held: a streamed turn
// waits for its close, and a truncated one stays hidden markup.
ParsedGeneration ParseGemmaGeneration(
    std::string_view raw, std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required, QuoteTracker& quotes,
    ToolCloserSet closers) {
  ParsedGeneration parsed;
  const bool recognize =
      !tools.empty() && choice != ChatRequest::ToolChoice::kNone;
  std::string text;
  std::size_t cursor = 0;
  if (recognize) {
    quotes.Reset(raw);
    while (true) {
      const auto marker =
          EarliestMarker(raw, ToolMarkerSet(&models::gemma4::kToolCallStart, 1),
                         quotes, cursor);
      if (marker == std::string_view::npos)
        break;
      const auto begin = marker + models::gemma4::kToolCallStart.size();
      const auto close = raw.find(models::gemma4::kToolCallEnd, begin);
      if (close == std::string_view::npos)
        break;  // An unclosed call hides the rest of the turn.
      text.append(ContentBefore(raw, cursor, marker, closers, quotes, tools));
      const auto body = raw.substr(begin, close - begin);
      const auto call = models::gemma4::ParseToolCall(body);
      if (call && GemmaToolDeclared(body, tools)) {
        ParsedToolCall parsed_call;
        parsed_call.id = RandomId("call_");
        parsed_call.name = call->name;
        for (const auto& member : call->arguments.members()) {
          tokenization::ChatMessage::ToolArgument argument;
          argument.name = member.first;
          argument.is_string = member.second.is_string();
          argument.value =
              argument.is_string ? member.second.str() : member.second.dump();
          parsed_call.arguments.push_back(std::move(argument));
        }
        parsed.tool_calls.push_back(std::move(parsed_call));
      }
      cursor = close + models::gemma4::kToolCallEnd.size();
    }
  }
  text.append(ContentBefore(raw, cursor, raw.size(), closers, quotes, tools));
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired &&
      parsed.tool_calls.empty())
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not produce a declared tool call");
  parsed.text = std::move(text);
  parsed.hide_tool_markup = true;
  return parsed;
}
#endif  // GUFO_STAGING_MODELS
