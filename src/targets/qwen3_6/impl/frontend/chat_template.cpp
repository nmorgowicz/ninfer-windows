#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr Sha256Digest kFroggericV22_5TemplateDigest{
    0xe5, 0x76, 0x84, 0xba, 0xe4, 0x15, 0x62, 0x11, 0xa5, 0x54, 0x73, 0xc5, 0xa6, 0x3b, 0xe9, 0x76,
    0xa4, 0x05, 0xa3, 0x7a, 0xb5, 0xbe, 0x5a, 0xe0, 0xe5, 0xab, 0xf1, 0xdf, 0x53, 0x49, 0xc4, 0xb2,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

void append_literal_span(std::vector<ByteSpan>& spans, ByteSpan span) {
    if (span.begin == span.end) { return; }
    if (!spans.empty() && spans.back().end == span.begin) {
        spans.back().end = span.end;
        return;
    }
    if (!spans.empty() && spans.back().end > span.begin) {
        throw std::logic_error("rendered literal byte spans overlap");
    }
    spans.push_back(span);
}

class RenderBuilder {
public:
    void append_template(std::string_view text) { fragment_.text += text; }

    void append_literal(std::string_view text) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        append_literal_span(fragment_.literal_spans, ByteSpan{begin, fragment_.text.size()});
    }

    void append_media_placeholder(std::string_view text, Modality modality,
                                  std::size_t item_index) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        fragment_.media_placeholders.push_back(MediaPlaceholderByteSpec{
            .bytes      = ByteSpan{begin, fragment_.text.size()},
            .modality   = modality,
            .item_index = item_index,
        });
    }

    void append(RenderedFragment fragment) {
        const std::size_t offset = fragment_.text.size();
        fragment_.text += fragment.text;
        for (const ByteSpan span : fragment.literal_spans) {
            append_literal_span(fragment_.literal_spans,
                                ByteSpan{offset + span.begin, offset + span.end});
        }
        for (MediaPlaceholderByteSpec placeholder : fragment.media_placeholders) {
            placeholder.bytes.begin += offset;
            placeholder.bytes.end += offset;
            fragment_.media_placeholders.push_back(placeholder);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return fragment_.text.size(); }

    [[nodiscard]] RenderedFragment release() && { return std::move(fragment_); }

private:
    RenderedFragment fragment_;
};

RenderedFragment literal_fragment(std::string text) {
    RenderBuilder builder;
    builder.append_literal(text);
    return std::move(builder).release();
}

std::pair<std::size_t, std::size_t> trim_ascii_whitespace_bounds(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return {begin, end};
}

RenderedFragment slice_fragment(const RenderedFragment& source, std::size_t begin,
                                std::size_t end) {
    if (begin > end || end > source.text.size()) {
        throw std::logic_error("rendered fragment slice is out of range");
    }
    RenderedFragment result;
    result.text = source.text.substr(begin, end - begin);
    for (const ByteSpan span : source.literal_spans) {
        const std::size_t clipped_begin = std::max(span.begin, begin);
        const std::size_t clipped_end   = std::min(span.end, end);
        if (clipped_begin < clipped_end) {
            append_literal_span(result.literal_spans,
                                ByteSpan{clipped_begin - begin, clipped_end - begin});
        }
    }
    for (MediaPlaceholderByteSpec placeholder : source.media_placeholders) {
        if (placeholder.bytes.end <= begin || placeholder.bytes.begin >= end) { continue; }
        if (placeholder.bytes.begin < begin || placeholder.bytes.end > end) {
            throw std::logic_error("rendered fragment slice intersects a media placeholder");
        }
        placeholder.bytes.begin -= begin;
        placeholder.bytes.end -= begin;
        result.media_placeholders.push_back(placeholder);
    }
    return result;
}

RenderedFragment trim_ascii_whitespace(const RenderedFragment& fragment) {
    const auto [begin, end] = trim_ascii_whitespace_bounds(fragment.text);
    return slice_fragment(fragment, begin, end);
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool starts_with_view(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    long trailing_tool_query = -1;
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role == ChatRole::Tool && trailing_tool_query < 0) { trailing_tool_query = i; }
        if (message.role != ChatRole::User) { continue; }
        const RenderedFragment content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content.text, "<tool_response>") &&
              ends_with(content.text, "</tool_response>"))) {
            return i;
        }
    }
    // A truncated imported history may begin at the tool-result turn. That result is the current
    // query even though the user message that initiated the omitted tool call is not visible.
    if (trailing_tool_query >= 0) { return trailing_tool_query; }
    throw std::invalid_argument("no user query found in chat messages");
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    RenderedFragment reasoning;
    RenderedFragment content;
};

ThinkParts derive_think_parts(const RenderedFragment& content) {
    ThinkParts parts;
    const std::size_t first_close = content.text.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::size_t before_end = first_close;
    while (before_end > 0 && content.text[before_end - 1] == '\n') { --before_end; }
    const std::size_t last_open = content.text.substr(0, before_end).rfind("<think>");
    std::size_t reasoning_begin =
        last_open == std::string::npos ? 0 : last_open + std::string("<think>").size();
    while (reasoning_begin < before_end && content.text[reasoning_begin] == '\n') {
        ++reasoning_begin;
    }
    parts.reasoning = slice_fragment(content, reasoning_begin, before_end);
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.text.rfind("</think>");
    std::size_t content_begin    = last_close + std::string("</think>").size();
    while (content_begin < content.text.size() && content.text[content_begin] == '\n') {
        ++content_begin;
    }
    parts.content = slice_fragment(content, content_begin, content.text.size());
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

RenderedFragment render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    RenderBuilder rendered;
    if (allow_empty_arguments && call.arguments_json.empty()) {
        rendered.append_template("<tool_call>\n<function=");
        rendered.append_literal(call.name);
        rendered.append_template(">\n</function>\n</tool_call>");
        return std::move(rendered).release();
    }
    OrderedJson args = OrderedJson::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    rendered.append_template("<tool_call>\n<function=");
    rendered.append_literal(call.name);
    rendered.append_template(">\n");
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered.append_template("<parameter=");
        rendered.append_literal(it.key());
        rendered.append_template(">\n");
        rendered.append_literal(parameter_text(it.value()));
        rendered.append_template("\n</parameter>\n");
    }
    rendered.append_template("</function>\n</tool_call>");
    return std::move(rendered).release();
}

struct RenderedToolsSystemBlock {
    RenderedFragment fragment;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
};

RenderedToolsSystemBlock render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                                   const RenderedFragment& leading_instruction,
                                                   std::string_view reasoning_instructions) {
    RenderedToolsSystemBlock out;
    RenderBuilder rendered;
    out.tool_boundaries.reserve(tool_jsons.size());
    rendered.append_template("<|im_start|>system\n");
    if (!reasoning_instructions.empty()) {
        rendered.append_template(reasoning_instructions);
        rendered.append_template("\n\n");
    }
    rendered.append_template("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const std::string& tool : tool_jsons) {
        rendered.append_template("\n");
        rendered.append_literal(tojson_text(OrderedJson::parse(tool)));
        out.tool_boundaries.push_back(rendered.size());
    }
    rendered.append_template("\n</tools>");
    rendered.append_template(kToolInstructions);
    if (!leading_instruction.text.empty()) {
        rendered.append_template("\n\n");
        out.instruction_begin = rendered.size();
        rendered.append(leading_instruction);
    }
    rendered.append_template("<|im_end|>\n");
    out.fragment = std::move(rendered).release();
    return out;
}

// --- froggeric-v22.5 renderer support -------------------------------------------------------

enum class FroggericEffort : std::uint8_t { Low, Medium, XHigh };

struct FroggericThinkState {
    bool thinking = true;
    FroggericEffort effort = FroggericEffort::Medium;
};

std::string ascii_lower(std::string_view text) {
    std::string result(text);
    for (char& ch : result) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
    return result;
}

// Global pre-scan of inline <|think_*|> control tags across all system/developer/user
// messages, mutating thinking/effort state based on the last matching tag found (in
// document order, per content part) anywhere in the message list.
FroggericThinkState scan_froggeric_think_tags(const std::vector<ChatMessage>& messages,
                                              FroggericThinkState initial) {
    FroggericThinkState state = initial;
    for (const ChatMessage& message : messages) {
        if (!is_instruction_role(message.role) && message.role != ChatRole::User) { continue; }
        for (const ChatPart& part : message.parts) {
            if (part.kind != ChatPartKind::Text) { continue; }
            const std::string& text = part.text;
            if (contains(text, "<|think_off|>")) {
                state.thinking = false;
            } else if (contains(text, "<|think_on|>")) {
                state.thinking = true;
            } else if (contains(text, "<|think_xhigh|>") || contains(text, "<|think_high|>") ||
                      contains(text, "<|think_ultracode|>") || contains(text, "<|think_extreme|>") ||
                      contains(text, "<|think_max|>")) {
                state.thinking = true;
                state.effort   = FroggericEffort::XHigh;
            } else if (contains(text, "<|think_low|>") || contains(text, "<|think_minimal|>")) {
                state.thinking = true;
                state.effort   = FroggericEffort::Low;
            } else if (contains(text, "<|think_medium|>")) {
                state.thinking = true;
                state.effort   = FroggericEffort::Medium;
            }
        }
    }
    return state;
}

constexpr std::array<std::string_view, 10> kFroggericThinkTagStripOrder = {
    "<|think_off|>",      "<|think_on|>",     "<|think_xhigh|>", "<|think_high|>",
    "<|think_ultracode|>", "<|think_extreme|>", "<|think_max|>",   "<|think_medium|>",
    "<|think_low|>",      "<|think_minimal|>",
};

RenderedFragment remove_all_literal(const RenderedFragment& source, std::string_view needle) {
    if (needle.empty()) { return source; }
    RenderBuilder builder;
    std::size_t pos = 0;
    while (true) {
        const std::size_t found = source.text.find(needle, pos);
        if (found == std::string::npos) {
            builder.append(slice_fragment(source, pos, source.text.size()));
            break;
        }
        builder.append(slice_fragment(source, pos, found));
        pos = found + needle.size();
    }
    return std::move(builder).release();
}

RenderedFragment strip_froggeric_think_tags(RenderedFragment content) {
    for (const std::string_view tag : kFroggericThinkTagStripOrder) {
        if (contains(content.text, tag)) {
            content = trim_ascii_whitespace(remove_all_literal(content, tag));
        }
    }
    return content;
}

constexpr std::string_view kFroggericToolInstructionsThinkingOn =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<think>\nBrief explanation of tool call\n</think>\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n"
    "- You can use the <think></think> block to plan your next tool call OR to synthesize data "
    "and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
    "thinking, with NO conversational text before it.\n"
    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO "
    "spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> "
    "block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user without "
    "any tool call.\n</IMPORTANT>";

constexpr std::string_view kFroggericToolInstructionsThinkingOff =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO "
    "conversational text before it.\n"
    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO "
    "spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> "
    "block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user without "
    "any tool call.\n</IMPORTANT>";

constexpr std::string_view kToolErrorWarningSuffix1 =
    "\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and "
    "retry with completely corrected arguments.";

std::string tool_error_warning_suffix_n(int failures) {
    return "\n\n⚠️ SYSTEM WARNING: " + std::to_string(failures) +
          " consecutive tool errors detected. Your previous approach is incorrect. You MUST use "
          "a fundamentally different approach or corrected arguments.";
}

bool froggeric_tool_response_is_failure(std::string_view content) {
    const std::string content_lower = ascii_lower(content);
    const std::string_view content_head =
        std::string_view(content_lower).substr(0, std::min<std::size_t>(120, content_lower.size()));

    const bool is_code_or_grep =
        contains(content_lower, "throw new ") || contains(content_lower, "throw error") ||
        contains(content_lower, "console.error") || contains(content_lower, "logger.error") ||
        contains(content_lower, "logging.error") || contains(content_head, "import ") ||
        contains(content_head, "def ") || contains(content_head, "function ");
    const bool exit_code_zero = contains(content_head, "exit code: 0") ||
                                contains(content_head, "process exited with code 0");
    const bool error_field_ok =
        contains(content_head, "\"error\": null") || contains(content_head, "\"error\":null") ||
        contains(content_head, "\"error\": false") || contains(content_head, "\"error\":false") ||
        contains(content_head, "\"error\": \"\"") || contains(content_head, "\"error\":\"\"");
    const bool strong_error =
        (contains(content_head, "\"error\":") && !error_field_ok) ||
        contains(content_head, "\"status\": \"error\"") ||
        contains(content_head, "\"status\":\"error\"") ||
        contains(content_head, "traceback (most recent call last):") ||
        contains(content_head, "command not found") || contains(content_head, "invalid syntax") ||
        contains(content_head, "fatal:") ||
        ((contains(content_head, "exit code: ") ||
          contains(content_head, "process exited with code")) &&
        !exit_code_zero) ||
        starts_with_view(content_head, "exception:") || starts_with_view(content_head, "failed to ");
    const bool weak_error = contains(content_head, "error:") || contains(content_head, "err!");
    const bool weak_suppressed = contains(content_head, "$ ") || contains(content_head, "took ") ||
                                 content.size() >= 600;
    return !is_code_or_grep && (strong_error || (weak_error && !weak_suppressed));
}

std::string truncate_with_marker(std::string_view text, std::size_t max_chars) {
    if (max_chars == 0 || text.size() <= max_chars) { return std::string(text); }
    return std::string(text.substr(0, max_chars)) + "\n[TRUNCATED - original length " +
          std::to_string(text.size()) + " chars]";
}

long froggeric_last_query_index(const std::vector<const ChatMessage*>& msgs) {
    const long last_idx = static_cast<long>(msgs.size()) - 1;
    bool multi_step_tool  = true;
    long last_query_index = last_idx;
    for (long i = last_idx; i >= 0; --i) {
        const ChatMessage& message = *msgs[static_cast<std::size_t>(i)];
        if (multi_step_tool && message.role == ChatRole::User) {
            const RenderedFragment content = trim_ascii_whitespace(message.rendered_content());
            if (!(starts_with(content.text, "<tool_response>") &&
                  ends_with(content.text, "</tool_response>"))) {
                multi_step_tool  = false;
                last_query_index = i;
            }
        }
    }
    if (multi_step_tool) { last_query_index = last_idx > 50 ? last_idx : 0; }
    return last_query_index;
}

RenderedToolsSystemBlock
render_froggeric_tools_system_block(const std::vector<std::string>& tool_jsons,
                                    const RenderedFragment& leading_instruction,
                                    std::string_view reasoning_instructions, bool thinking) {
    RenderedToolsSystemBlock out;
    RenderBuilder rendered;
    out.tool_boundaries.reserve(tool_jsons.size());
    rendered.append_template("<|im_start|>system\n");
    if (!reasoning_instructions.empty()) {
        rendered.append_template(reasoning_instructions);
        rendered.append_template("\n\n");
    }
    rendered.append_template("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const std::string& tool : tool_jsons) {
        rendered.append_template("\n");
        rendered.append_literal(tojson_text(OrderedJson::parse(tool)));
        out.tool_boundaries.push_back(rendered.size());
    }
    rendered.append_template("\n</tools>");
    rendered.append_template(thinking ? kFroggericToolInstructionsThinkingOn
                                      : kFroggericToolInstructionsThinkingOff);
    if (!leading_instruction.text.empty()) {
        rendered.append_template("\n\n");
        out.instruction_begin = rendered.size();
        rendered.append(leading_instruction);
    }
    rendered.append_template("<|im_end|>\n");
    out.fragment = std::move(rendered).release();
    return out;
}

RenderedFragment render_froggeric_tool_call(const ToolCall& call, std::size_t max_tool_arg_chars) {
    RenderBuilder rendered;
    rendered.append_template("<function=");
    rendered.append_literal(call.name);
    rendered.append_template(">\n");

    OrderedJson args;
    bool args_is_object = false;
    if (!call.arguments_json.empty()) {
        try {
            args            = OrderedJson::parse(call.arguments_json);
            args_is_object = args.is_object();
        } catch (const nlohmann::json::parse_error&) {
            args_is_object = false;
        }
    }

    if (args_is_object) {
        for (auto it = args.begin(); it != args.end(); ++it) {
            rendered.append_template("<parameter=");
            rendered.append_literal(it.key());
            rendered.append_template(">\n");
            rendered.append_literal(truncate_with_marker(parameter_text(it.value()), max_tool_arg_chars));
            rendered.append_template("\n</parameter>\n");
        }
    } else if (!call.arguments_json.empty()) {
        rendered.append_literal(truncate_with_marker(call.arguments_json, max_tool_arg_chars));
    }
    rendered.append_template("</function>\n</tool_call>");
    return std::move(rendered).release();
}

RenderedChat render_froggeric_v22_5(const std::vector<ChatMessage>& messages,
                                    const ChatRenderOptions& options) {
    const FroggericEffort initial_effort = [&] {
        switch (options.reasoning_effort.value_or(ReasoningEffort::Medium)) {
        case ReasoningEffort::Low:
            return FroggericEffort::Low;
        case ReasoningEffort::Medium:
            return FroggericEffort::Medium;
        case ReasoningEffort::XHigh:
            return FroggericEffort::XHigh;
        }
        return FroggericEffort::Medium;
    }();
    FroggericThinkState state =
        scan_froggeric_think_tags(messages, FroggericThinkState{options.enable_thinking, initial_effort});

    std::string_view reasoning_instructions;
    if (state.thinking) {
        if (state.effort == FroggericEffort::XHigh) {
            reasoning_instructions = kXHighReasoningInstructions;
        } else if (state.effort == FroggericEffort::Low) {
            reasoning_instructions = kLowReasoningInstructions;
        }
    }

    std::size_t head_count = 0;
    while (head_count < messages.size() && is_instruction_role(messages[head_count].role)) {
        ++head_count;
    }

    RenderBuilder leading_builder;
    bool leading_has_content = false;
    for (std::size_t i = 0; i < head_count; ++i) {
        validate_instruction_message(messages[i]);
        RenderedFragment part = trim_ascii_whitespace(messages[i].rendered_content());
        part                  = strip_froggeric_think_tags(std::move(part));
        if (part.text.empty()) { continue; }
        if (leading_has_content) { leading_builder.append_template("\n\n"); }
        leading_builder.append(std::move(part));
        leading_has_content = true;
    }
    RenderedFragment leading_instruction = std::move(leading_builder).release();

    std::vector<const ChatMessage*> tail_messages;
    tail_messages.reserve(messages.size() - head_count);
    for (std::size_t i = head_count; i < messages.size(); ++i) { tail_messages.push_back(&messages[i]); }

    RenderBuilder rendered;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        RenderedToolsSystemBlock preamble = render_froggeric_tools_system_block(
            options.tool_jsons, leading_instruction, reasoning_instructions, state.thinking);
        rendered.append(std::move(preamble.fragment));
        tool_boundaries   = std::move(preamble.tool_boundaries);
        instruction_begin = preamble.instruction_begin;
    } else if (leading_has_content) {
        rendered.append_template("<|im_start|>system\n");
        if (!reasoning_instructions.empty()) {
            rendered.append_template(reasoning_instructions);
            rendered.append_template("\n\n");
        }
        instruction_begin = rendered.size();
        rendered.append(leading_instruction);
        rendered.append_template("<|im_end|>\n");
    } else if (!reasoning_instructions.empty()) {
        rendered.append_template("<|im_start|>system\n");
        rendered.append_template(reasoning_instructions);
        rendered.append_template("<|im_end|>\n");
    }

    std::vector<std::optional<std::size_t>> message_boundaries(messages.size() + 1U);
    std::vector<std::optional<std::size_t>> cache_boundaries(options.cache_markers.size());
    if (head_count == 0) {
        message_boundaries[0] = rendered.size();
    } else {
        message_boundaries[head_count] = rendered.size();
    }

    const long last_query_index  = froggeric_last_query_index(tail_messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(true);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    const auto add_rewrite_execution_boundary = [&] {
        if (rewrite_execution_boundaries.empty() ||
            rewrite_execution_boundaries.back() != rendered.size()) {
            rewrite_execution_boundaries.push_back(rendered.size());
        }
    };

    int consecutive_failures = 0;
    ChatRole prev_role       = ChatRole::User;
    bool have_prev_role      = false;

    for (std::size_t i = 0; i < tail_messages.size(); ++i) {
        const ChatMessage& message = *tail_messages[i];
        const std::size_t original_index = head_count + i;
        const bool is_system             = is_instruction_role(message.role);
        if (is_system) { validate_instruction_message(message); }

        RenderedFragment content = trim_ascii_whitespace(message.rendered_content(options.add_vision_id));
        if (is_system || message.role == ChatRole::User) {
            content = strip_froggeric_think_tags(std::move(content));
        }

        if (is_system) {
            rendered.append_template("<|im_start|>system\n");
            rendered.append(std::move(content));
            rendered.append_template("<|im_end|>\n");
        } else if (message.role == ChatRole::User) {
            consecutive_failures = 0;
            rendered.append_template("<|im_start|>user\n");
            rendered.append(std::move(content));
            rendered.append_template("<|im_end|>\n");
        } else if (message.role == ChatRole::Assistant) {
            RenderedFragment reasoning;
            if (!message.reasoning_content.empty()) {
                std::string_view lead_end;
                if (starts_with(content.text, "<think>") && contains(content.text, "</think>")) {
                    lead_end = "</think>";
                } else if (starts_with(content.text, "<thinking>") &&
                          contains(content.text, "</thinking>")) {
                    lead_end = "</thinking>";
                } else if (starts_with(content.text, "</think>")) {
                    lead_end = "</think>";
                } else if (starts_with(content.text, "</thinking>")) {
                    lead_end = "</thinking>";
                }
                if (!lead_end.empty()) {
                    const std::size_t last = content.text.rfind(lead_end);
                    std::size_t begin      = last + lead_end.size();
                    while (begin < content.text.size() && content.text[begin] == '\n') { ++begin; }
                    content = slice_fragment(content, begin, content.text.size());
                }
                reasoning = literal_fragment(message.reasoning_content);
            } else {
                std::string_view think_end;
                if (starts_with(content.text, "</think>")) {
                    think_end = "</think>";
                } else if (starts_with(content.text, "</thinking>")) {
                    think_end = "</thinking>";
                } else if (contains(content.text, "\n</think>")) {
                    think_end = "\n</think>";
                } else if (contains(content.text, "\n</thinking>")) {
                    think_end = "\n</thinking>";
                } else if (contains(content.text, "\n</ think>")) {
                    think_end = "\n</ think>";
                } else if (contains(content.text, "\n</think >")) {
                    think_end = "\n</think >";
                } else if (starts_with(content.text, "<think>") && contains(content.text, "</think>")) {
                    think_end = "</think>";
                } else if (starts_with(content.text, "<thinking>") &&
                          contains(content.text, "</thinking>")) {
                    think_end = "</thinking>";
                }
                if (!think_end.empty()) {
                    const std::string_view think_start =
                        contains(think_end, "thinking") ? std::string_view("<thinking>")
                                                        : std::string_view("<think>");
                    const std::size_t first_close = content.text.find(think_end);
                    std::size_t before_end        = first_close;
                    while (before_end > 0 && content.text[before_end - 1] == '\n') { --before_end; }
                    RenderedFragment before = slice_fragment(content, 0, before_end);
                    if (contains(before.text, think_start)) {
                        const std::size_t last_open = before.text.rfind(think_start);
                        std::size_t reasoning_begin  = last_open + think_start.size();
                        while (reasoning_begin < before.text.size() &&
                              before.text[reasoning_begin] == '\n') {
                            ++reasoning_begin;
                        }
                        reasoning = slice_fragment(before, reasoning_begin, before.text.size());
                    } else {
                        reasoning = before;
                    }
                    const std::size_t last_close = content.text.rfind(think_end);
                    std::size_t content_begin     = last_close + think_end.size();
                    while (content_begin < content.text.size() && content.text[content_begin] == '\n') {
                        ++content_begin;
                    }
                    content = slice_fragment(content, content_begin, content.text.size());
                }
            }
            reasoning = trim_ascii_whitespace(reasoning);

            const bool keep_thinking =
                preserve_thinking || (static_cast<long>(i) > last_query_index);
            if (!preserve_thinking && !rewrite_checkpoint &&
                static_cast<long>(i) > last_query_index) {
                rewrite_checkpoint = RewriteCheckpointByteSpec{
                    .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
            }
            rendered.append_template("<|im_start|>assistant\n");
            add_rewrite_execution_boundary();
            if (keep_thinking) {
                rendered.append_template("<think>\n");
                add_rewrite_execution_boundary();
                rendered.append(std::move(reasoning));
                rendered.append_template(kCanonicalReasoningCloseSerialization);
                add_rewrite_execution_boundary();
            }
            const bool content_has_text = !trim_ascii_whitespace(content).text.empty();
            rendered.append(std::move(content));
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    rendered.append_template(content_has_text ? "\n\n<tool_call>\n"
                                                              : "<tool_call>\n");
                } else {
                    rendered.append_template("\n<tool_call>\n");
                }
                rendered.append(
                    render_froggeric_tool_call(message.tool_calls[call_index], options.max_tool_arg_chars));
            }
            rendered.append_template("<|im_end|>\n");
        } else if (message.role == ChatRole::Tool) {
            const bool is_failure = froggeric_tool_response_is_failure(content.text);
            consecutive_failures  = is_failure ? consecutive_failures + 1 : 0;
            if (!have_prev_role || prev_role != ChatRole::Tool) {
                rendered.append_template("<|im_start|>user");
            }
            if (options.max_tool_response_chars > 0 &&
                content.text.size() > options.max_tool_response_chars) {
                content = literal_fragment(
                    truncate_with_marker(content.text, options.max_tool_response_chars));
            }
            rendered.append_template("\n<tool_response>\n");
            rendered.append(std::move(content));
            if (consecutive_failures >= 2) {
                rendered.append_template(tool_error_warning_suffix_n(consecutive_failures));
            } else if (consecutive_failures == 1) {
                rendered.append_template(kToolErrorWarningSuffix1);
            }
            rendered.append_template("\n</tool_response>");
            const bool is_last  = i + 1 == tail_messages.size();
            const bool next_tool = !is_last && tail_messages[i + 1]->role == ChatRole::Tool;
            if (is_last || !next_tool) { rendered.append_template("<|im_end|>\n"); }
        } else {
            throw std::invalid_argument("unsupported chat role value");
        }

        prev_role      = message.role;
        have_prev_role = true;
        message_boundaries[original_index + 1U] = rendered.size();
    }

    if (options.add_generation_prompt) {
        const std::size_t generation_begin = rendered.size();
        if (preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
        } else if (!rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = generation_begin};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (state.thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
        } else {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append_template(kCanonicalReasoningCloseSerialization);
            add_rewrite_execution_boundary();
        }
    }

    for (std::size_t index = 0; index < options.cache_markers.size(); ++index) {
        const PromptCacheMarker marker = options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < message_boundaries.size()) {
                cache_boundaries[index] = message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (head_count > 0 && instruction_begin) {
                cache_boundaries[index] = instruction_begin;
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= tool_boundaries.size()) {
                cache_boundaries[index] = tool_boundaries[marker.after_tool_count - 1U];
            }
            break;
        }
    }

    RenderedFragment final = std::move(rendered).release();
    return RenderedChat{.text                         = std::move(final.text),
                        .literal_spans                = std::move(final.literal_spans),
                        .media_placeholders           = std::move(final.media_placeholders),
                        .rewrite_checkpoint           = rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(rewrite_execution_boundaries),
                        .message_boundaries           = std::move(message_boundaries),
                        .cache_boundaries             = std::move(cache_boundaries)};
}

// --- end froggeric-v22.5 renderer support ----------------------------------------------------

std::string_view resolve_reasoning_instructions(ChatTemplateSemantics semantics,
                                                const ChatRenderOptions& options) {
    if (semantics == ChatTemplateSemantics::ThinkingToggle) {
        if (options.reasoning_effort) {
            throw std::invalid_argument("loaded chat template does not support reasoning effort");
        }
        return {};
    }
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }

    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

RenderedFragment ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                               int* video_count, std::size_t* media_count,
                                               std::vector<std::size_t>* part_boundaries) const {
    int local_images        = 0;
    int local_videos        = 0;
    std::size_t local_media = 0;
    int& images             = image_count == nullptr ? local_images : *image_count;
    int& videos             = video_count == nullptr ? local_videos : *video_count;
    std::size_t& media      = media_count == nullptr ? local_media : *media_count;
    RenderBuilder out;
    if (part_boundaries != nullptr) {
        part_boundaries->clear();
        part_boundaries->reserve(parts.size());
    }
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out.append_literal(part.text);
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out.append_template("Picture " + std::to_string(images) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|image_pad|>", Modality::Image, media++);
            out.append_template("<|vision_end|>");
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out.append_template("Video " + std::to_string(videos) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|video_pad|>", Modality::Video, media++);
            out.append_template("<|vision_end|>");
            break;
        }
        if (part_boundaries != nullptr) { part_boundaries->push_back(out.size()); }
    }
    return std::move(out).release();
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    if (digest == kFroggericV22_5TemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::FroggericV22_5);
    }
    throw std::invalid_argument("unsupported frontend/chat_template.jinja (sha256 " +
                                sha256_hex(digest) + ")");
}

CompiledChatTemplate CompiledChatTemplate::resolve_with_semantics(std::string_view source,
                                                                   std::string_view semantics_mode) {
    if (semantics_mode.empty() || semantics_mode == "auto") { return resolve(source); }
    if (semantics_mode == "froggeric") { return CompiledChatTemplate(ChatTemplateSemantics::FroggericV22_5); }
    if (semantics_mode == "thinking-toggle") {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (semantics_mode == "reasoning-effort") {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    if (semantics_mode == "generic") { return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle); }
    throw std::invalid_argument("unknown --chat-template-semantics mode: " +
                                std::string(semantics_mode));
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort ||
        semantics_ == ChatTemplateSemantics::FroggericV22_5) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }
    if (semantics_ == ChatTemplateSemantics::FroggericV22_5) {
        return render_froggeric_v22_5(messages, options);
    }

    const bool continue_final_assistant =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (continue_final_assistant) {
        const ChatMessage& final = messages.back();
        if (final.role != ChatRole::Assistant || final.parts.empty() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (final.has_media()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (options.enable_thinking) {
            throw std::invalid_argument("assistant continuation cannot start in thinking mode");
        }
    }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        resolve_reasoning_instructions(semantics_, options);

    std::size_t message_begin = 0;
    RenderedFragment leading_instruction_raw;
    RenderedFragment leading_instruction;
    std::size_t leading_trim_begin = 0;
    std::size_t leading_trim_end   = 0;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_instruction_raw = messages[0].rendered_content();
        std::tie(leading_trim_begin, leading_trim_end) =
            trim_ascii_whitespace_bounds(leading_instruction_raw.text);
        leading_instruction =
            slice_fragment(leading_instruction_raw, leading_trim_begin, leading_trim_end);
        message_begin = 1;
    }

    RenderBuilder rendered;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        RenderedToolsSystemBlock preamble = render_tools_system_block(
            options.tool_jsons, leading_instruction, reasoning_instructions);
        rendered.append(std::move(preamble.fragment));
        tool_boundaries   = std::move(preamble.tool_boundaries);
        instruction_begin = preamble.instruction_begin;
    } else if (message_begin == 1) {
        if (!effort_template || !leading_instruction.text.empty() ||
            !reasoning_instructions.empty()) {
            rendered.append_template("<|im_start|>system\n");
            if (!reasoning_instructions.empty()) {
                rendered.append_template(reasoning_instructions);
                if (!leading_instruction.text.empty()) { rendered.append_template("\n\n"); }
            }
            if (!leading_instruction.text.empty()) { instruction_begin = rendered.size(); }
            rendered.append(leading_instruction);
            rendered.append_template("<|im_end|>\n");
        }
    } else if (!reasoning_instructions.empty()) {
        rendered.append_template("<|im_start|>system\n");
        rendered.append_template(reasoning_instructions);
        rendered.append_template("<|im_end|>\n");
    }

    std::vector<std::optional<std::size_t>> message_boundaries(messages.size() + 1U);
    std::vector<std::optional<std::size_t>> cache_boundaries(options.cache_markers.size());
    if (message_begin == 0) {
        message_boundaries[0] = rendered.size();
    } else {
        // The first instruction message is folded into the system preamble by this template.
        message_boundaries[1] = rendered.size();
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    const auto add_rewrite_execution_boundary = [&] {
        if (rewrite_execution_boundaries.empty() ||
            rewrite_execution_boundaries.back() != rendered.size()) {
            rewrite_execution_boundaries.push_back(rendered.size());
        }
    };

    int image_count         = 0;
    int video_count         = 0;
    std::size_t media_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        std::vector<std::size_t> raw_part_boundaries;
        const RenderedFragment raw_content = message.rendered_content(
            options.add_vision_id, &image_count, &video_count, &media_count, &raw_part_boundaries);
        const auto [content_trim_begin, content_trim_end] =
            trim_ascii_whitespace_bounds(raw_content.text);
        const RenderedFragment content =
            slice_fragment(raw_content, content_trim_begin, content_trim_end);
        const auto resolve_part_boundaries = [&](std::size_t content_begin) {
            for (std::size_t marker_index = 0; marker_index < options.cache_markers.size();
                 ++marker_index) {
                const PromptCacheMarker& marker = options.cache_markers[marker_index];
                if (marker.location != PromptCacheMarkerLocation::MessagePartBoundary ||
                    marker.after_message_count != i + 1U || marker.after_message_part_count == 0 ||
                    marker.after_message_part_count > raw_part_boundaries.size()) {
                    continue;
                }
                const std::size_t raw = raw_part_boundaries[marker.after_message_part_count - 1U];
                const std::size_t clamped = std::clamp(raw, content_trim_begin, content_trim_end);
                cache_boundaries[marker_index] = content_begin + clamped - content_trim_begin;
            }
        };
        if (is_instruction_role(message.role)) {
            rendered.append_template("<|im_start|>system\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered.append_template("<|im_start|>user\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i == 0 || messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered.append_template("<|im_start|>user"); }
            rendered.append_template("\n<tool_response>\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("\n</tool_response>");
            if (closes_group) { rendered.append_template("<|im_end|>\n"); }
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        if (continue_final_assistant && i + 1U == messages.size()) {
            const std::size_t generation_begin = rendered.size();
            rewrite_checkpoint                 = RewriteCheckpointByteSpec{
                                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
            rendered.append_template("<|im_start|>assistant\n");
            add_rewrite_execution_boundary();
            rendered.append(content);
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        RenderedFragment reasoning;
        RenderedFragment body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = literal_fragment(message.reasoning_content);
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            // Closing the current turn may rewrite everything beginning with this assistant
            // segment. Keep the stable history before the opener recoverable; retaining the
            // deterministic opener itself is not worth losing the whole prefix when a caller
            // branches with a new user message instead.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (keep_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append(reasoning);
            rendered.append_template(kCanonicalReasoningCloseSerialization);
            add_rewrite_execution_boundary();
        }
        rendered.append(body);
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).text.empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered.append_template("\n\n"); }
                } else {
                    rendered.append_template("\n");
                }
                rendered.append(render_tool_call(message.tool_calls[call_index], effort_template));
            }
        }
        rendered.append_template("<|im_end|>\n");
        message_boundaries[i + 1U] = rendered.size();
    }

    if (!continue_final_assistant && options.add_generation_prompt) {
        // The generation suffix is replaceable as a unit. An immediate successor may replay the
        // response, close the turn, or branch by appending a different user message directly to
        // the input history. The rolling private checkpoint must therefore precede the assistant
        // opener; placing it after the deterministic prologue makes the complete history
        // unrecoverable for the branch case merely to save a handful of prompt tokens.
        const std::size_t generation_begin = rendered.size();
        if (preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
        } else if (!rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = generation_begin};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (options.enable_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
        } else {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append_template(kCanonicalReasoningCloseSerialization);
            add_rewrite_execution_boundary();
        }
    }
    for (std::size_t index = 0; index < options.cache_markers.size(); ++index) {
        const PromptCacheMarker marker = options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < message_boundaries.size()) {
                cache_boundaries[index] = message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            // Resolved while the containing message's content offset is known. Assistant
            // reasoning/tool-call interiors deliberately remain advisory and unresolved.
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (message_begin == 1 && instruction_begin &&
                marker.leading_instruction_bytes <= leading_instruction_raw.text.size()) {
                const std::size_t clamped = std::clamp<std::size_t>(
                    marker.leading_instruction_bytes, leading_trim_begin, leading_trim_end);
                cache_boundaries[index] = *instruction_begin + clamped - leading_trim_begin;
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= tool_boundaries.size()) {
                cache_boundaries[index] = tool_boundaries[marker.after_tool_count - 1U];
            }
            break;
        }
    }
    RenderedFragment final = std::move(rendered).release();
    return RenderedChat{.text                         = std::move(final.text),
                        .literal_spans                = std::move(final.literal_spans),
                        .media_placeholders           = std::move(final.media_placeholders),
                        .rewrite_checkpoint           = rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(rewrite_execution_boundaries),
                        .message_boundaries           = std::move(message_boundaries),
                        .cache_boundaries             = std::move(cache_boundaries)};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
