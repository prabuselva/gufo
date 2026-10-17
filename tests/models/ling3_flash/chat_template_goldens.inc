// Generated from src/models/ling3_flash/reference/chat_template.jinja
inline constexpr std::string_view kSimple = R"X(<role>SYSTEM</role>You are helpful.
detailed thinking on<|role_end|>
    <role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>
<think>)X";
inline constexpr std::string_view kThinkingOff = R"X(<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>
<think></think>)X";
inline constexpr std::string_view kSystemCarriesDirective = R"X(<role>SYSTEM</role>Be terse. detailed thinking off<|role_end|>
    <role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>
<think>)X";
inline constexpr std::string_view kToolsHeader = R"X(<role>SYSTEM</role>You are helpful.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"description": "Weather \u201cplain\u201d \u0026 \u003cmarkup\u003e for a city", "name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"], "type": "object"}}, "type": "function"}
</tools>

If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
If you need to use a function, for each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call>
detailed thinking on<|role_end|><role>HUMAN</role>Weather in Paris?<|role_end|><role>ASSISTANT</role>
<think>)X";
inline constexpr std::string_view kMultiTurn = R"X(<role>SYSTEM</role>You are helpful.
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"description": "Weather \u201cplain\u201d \u0026 \u003cmarkup\u003e for a city", "name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"], "type": "object"}}, "type": "function"}
{"description": "Send it \u2014 caf\u00e9 and \u0027quotes\u0027", "name": "send_mail", "parameters": {"properties": {"body": {"type": "string"}}, "type": "object"}}
</tools>

If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
If you need to use a function, for each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call>
detailed thinking on<|role_end|><role>HUMAN</role>Weather in Paris?<|role_end|><role>ASSISTANT</role>
<think>Looking it up.</think><tool_call>get_weather
                <arg_key>city</arg_key>
<arg_value>Paris</arg_value><arg_key>days</arg_key>
<arg_value>3</arg_value><arg_key>now</arg_key>
<arg_value>true</arg_value>
</tool_call><|role_end|><role>OBSERVATION</role>
<tool_response>
{"temp": 21}
</tool_response><|role_end|><role>ASSISTANT</role>
<think></think>Warm.
<tool_call>send_mail
                <arg_key>body</arg_key>
<arg_value>café — <b>ok</b></arg_value><arg_key>urgent</arg_key>
<arg_value>false</arg_value>
</tool_call><|role_end|><role>OBSERVATION</role>
<tool_response>
sent & done
</tool_response><|role_end|><role>HUMAN</role>thanks<|role_end|><role>ASSISTANT</role>
<think>)X";
inline constexpr std::string_view kConsecutiveToolTurns = R"X(<role>SYSTEM</role># Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"description": "Weather \u201cplain\u201d \u0026 \u003cmarkup\u003e for a city", "name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"], "type": "object"}}, "type": "function"}
</tools>

If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
If you need to use a function, for each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call>
detailed thinking on<|role_end|><role>HUMAN</role>Weather?<|role_end|><role>ASSISTANT</role>
<think></think><tool_call>get_weather
                
</tool_call><|role_end|><role>OBSERVATION</role>
<tool_response>
a
</tool_response>
<tool_response>
b
</tool_response><|role_end|>)X";
