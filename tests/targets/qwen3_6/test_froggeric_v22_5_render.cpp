// Render checks for the froggeric v22.5 chat-template semantics, hand-coded against the
// authoritative froggeric_qwen-chat_template-v22.5.jinja spec (not gzenz's older v22 port).

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <ninfer/types.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

namespace {

std::string read_fixture(const char* path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string source = ss.str();
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

int check(bool ok, const char* label) {
    if (!ok) { std::cerr << "FAIL: " << label << "\n"; }
    return ok ? 0 : 1;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::ChatMessage tool_result(std::string id, std::string content) {
    fi::ChatMessage message = chat_message(ninfer::ChatRole::Tool, std::move(content));
    message.tool_call_id    = std::move(id);
    return message;
}

fi::ChatMessage assistant_with_calls(std::string content, std::vector<fi::ToolCall> calls) {
    fi::ChatMessage message = chat_message(ninfer::ChatRole::Assistant, std::move(content));
    message.tool_calls      = std::move(calls);
    return message;
}

const std::string kToolJson =
    R"({"type":"function","function":{"name":"bash","description":"run","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}})";

const fi::CompiledChatTemplate& froggeric_template() {
    static const fi::CompiledChatTemplate value = fi::CompiledChatTemplate::resolve(read_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/froggeric_v22_5_chat_template.jinja"));
    return value;
}

} // namespace

int main() {
    int failures = 0;

    // 0. digest dispatch resolves to the froggeric semantics and advertises its full
    //    reasoning-effort capability set (low/medium/xhigh, default xhigh).
    {
        const ninfer::PromptCapabilities caps = froggeric_template().capabilities();
        failures += check(caps.enable_thinking && caps.reasoning_effort.low &&
                              caps.reasoning_effort.medium && caps.reasoning_effort.xhigh &&
                              caps.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
                          "froggeric template did not advertise its complete capability set");
    }

    // 1. tools system block: thinking-on variant carries the froggeric-specific instructions.
    {
        fi::ChatRenderOptions options;
        options.tool_jsons = {kToolJson};
        const std::string out =
            froggeric_template().render({chat_message(ninfer::ChatRole::User, "hi")}, options).text;
        failures += check(out.find("<think>\nBrief explanation of tool call\n</think>\n"
                                   "<tool_call>\n<function=example_function_name>") !=
                              std::string::npos,
                          "thinking-on tool instructions missing");
        failures += check(out.find("IMMEDIATELY after thinking") != std::string::npos,
                          "thinking-on IMPORTANT block missing");
        failures += check(out.find("Do NOT nest <tool_call> blocks") != std::string::npos,
                          "no-nest rule missing");
    }

    // 2. tools system block: thinking-off variant drops the think-first example and reminder.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "<|think_off|> hi"));
        fi::ChatRenderOptions options;
        options.tool_jsons = {kToolJson};
        const std::string out = froggeric_template().render(msgs, options).text;
        failures += check(out.find("Brief explanation of tool call") == std::string::npos,
                          "thinking-off tool instructions still carried the think-first example");
        failures += check(out.find("IMMEDIATELY, with NO") != std::string::npos,
                          "thinking-off IMPORTANT block missing");
        failures += check(out.find("<|think_off|>") == std::string::npos,
                          "control tag leaked into rendered output");
        failures +=
            check(out.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos,
                  "think_off tag did not disable the generation prologue's thinking block");
    }

    // 3. global think-tag pre-scan: a tag on an earlier message still governs a later
    //    system-block emission (last-match-wins across the whole message list).
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "<|think_xhigh|> plan carefully"));
        msgs.push_back(chat_message(ninfer::ChatRole::Assistant, "ok"));
        msgs.push_back(chat_message(ninfer::ChatRole::User, "now answer"));
        const std::string out = froggeric_template().render(msgs).text;
        failures += check(out.find("Reasoning effort is set to xhigh") != std::string::npos,
                          "later message did not inherit the earlier think_xhigh tag");
        failures += check(out.find("<|think_xhigh|>") == std::string::npos,
                          "think_xhigh tag leaked into rendered user content");
    }

    // 4. consecutive tool error warnings via the froggeric failure classifier.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "build it"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "1", .name = "bash", .arguments_json = "{\"command\":\"git clone x\"}"}}));
        msgs.push_back(tool_result("1", "fatal: could not read Username"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "2", .name = "bash", .arguments_json = "{\"command\":\"git clone x\"}"}}));
        msgs.push_back(tool_result("2", "fatal: could not read Username"));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        const std::string out = froggeric_template().render(msgs, options).text;
        failures += check(out.find("SYSTEM WARNING: The previous tool call returned an error") !=
                              std::string::npos,
                          "first-error warning missing");
        failures += check(out.find("SYSTEM WARNING: 2 consecutive tool errors detected") !=
                              std::string::npos,
                          "second-error warning missing");
    }

    // 5. a successful-looking tool response (source code payload) is not misclassified as an
    //    error even though it contains an "error" keyword deep in a log line.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "read the file"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "1", .name = "bash", .arguments_json = "{\"command\":\"cat x.py\"}"}}));
        msgs.push_back(tool_result("1", "def handler():\n    logger.error('boom')\n"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "2", .name = "bash", .arguments_json = "{\"command\":\"cat x.py\"}"}}));
        msgs.push_back(tool_result("2", "def handler():\n    logger.error('boom')\n"));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        const std::string out = froggeric_template().render(msgs, options).text;
        failures += check(out.find("SYSTEM WARNING") == std::string::npos,
                          "code payload containing 'error' was misclassified as a tool failure");
    }

    // 6. tool response truncation via max_tool_response_chars.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "q"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "1", .name = "bash", .arguments_json = "{}"}}));
        msgs.push_back(tool_result("1", std::string(50, 'x')));
        fi::ChatRenderOptions options;
        options.add_generation_prompt   = false;
        options.max_tool_response_chars = 10;
        const std::string out           = froggeric_template().render(msgs, options).text;
        failures += check(out.find(std::string(10, 'x') + "\n[TRUNCATED - original length 50 "
                                                           "chars]") != std::string::npos,
                          "tool response was not truncated with the expected marker");
    }

    // 7. tool call argument truncation via max_tool_arg_chars.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "q"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "1",
                              .name = "bash",
                              .arguments_json = R"({"command":")" + std::string(50, 'y') + "\"}"}}));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        options.max_tool_arg_chars    = 10;
        const std::string out         = froggeric_template().render(msgs, options).text;
        failures += check(out.find(std::string(10, 'y') + "\n[TRUNCATED - original length 50 "
                                                           "chars]") != std::string::npos,
                          "tool call argument was not truncated with the expected marker");
    }

    // 8. string (non-object) tool-call arguments are rendered raw rather than throwing.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "q"));
        msgs.push_back(assistant_with_calls(
            "", {fi::ToolCall{.id = "1", .name = "bash", .arguments_json = "not json at all"}}));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        const std::string out         = froggeric_template().render(msgs, options).text;
        failures += check(out.find("<function=bash>\nnot json at all</function>") != std::string::npos,
                          "raw string arguments not rendered verbatim");
    }

    // 9. multiple tool calls in one turn are blank-line separated.
    {
        std::vector<fi::ChatMessage> msgs;
        msgs.push_back(chat_message(ninfer::ChatRole::User, "q"));
        msgs.push_back(assistant_with_calls("", {fi::ToolCall{.id = "1", .name = "a", .arguments_json = "{}"},
                                                 fi::ToolCall{.id = "2", .name = "b", .arguments_json = "{}"}}));
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        const std::string out         = froggeric_template().render(msgs, options).text;
        failures += check(out.find("</tool_call>\n<tool_call>") != std::string::npos,
                          "second call did not immediately follow the first");
    }

    // 10. explicit reasoning_content field: a leading <think> block embedded in content is
    //     still stripped (the fixed bug from the initial port).
    {
        fi::ChatMessage assistant = chat_message(ninfer::ChatRole::Assistant,
                                                 "<think>\nstale inline reasoning\n</think>\nfinal answer");
        assistant.reasoning_content = "authoritative reasoning";
        std::vector<fi::ChatMessage> msgs{chat_message(ninfer::ChatRole::User, "q"), assistant};
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        options.preserve_thinking     = true;
        const std::string out         = froggeric_template().render(msgs, options).text;
        failures += check(out.find("authoritative reasoning") != std::string::npos,
                          "explicit reasoning_content field was not used");
        failures += check(out.find("stale inline reasoning") == std::string::npos,
                          "leading inline <think> block was not stripped when reasoning_content "
                          "was explicit");
        failures += check(out.find("<|im_start|>assistant\n<think>\nauthoritative reasoning"
                                   "\n</think>\n\nfinal answer") != std::string::npos,
                          "final answer body was not preserved after stripping the stale think block");
    }

    // 11. embedded-tag reasoning derivation path (no explicit reasoning_content field).
    {
        fi::ChatMessage assistant =
            chat_message(ninfer::ChatRole::Assistant, "<think>\nderived reasoning\n</think>\nfinal");
        std::vector<fi::ChatMessage> msgs{chat_message(ninfer::ChatRole::User, "q"), assistant};
        fi::ChatRenderOptions options;
        options.add_generation_prompt = false;
        options.preserve_thinking     = true;
        const std::string out         = froggeric_template().render(msgs, options).text;
        failures += check(out.find("<|im_start|>assistant\n<think>\nderived reasoning\n</think>\n\n"
                                   "final") != std::string::npos,
                          "embedded-tag reasoning was not derived from content");
    }

    // 12. generation prompt with thinking disabled emits the empty <think></think> prologue.
    {
        fi::ChatRenderOptions options;
        options.enable_thinking = false;
        const std::string out =
            froggeric_template().render({chat_message(ninfer::ChatRole::User, "hi")}, options).text;
        failures += check(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                          "thinking-off generation prompt did not emit the empty think block");
    }

    if (failures == 0) { std::cout << "ALL FROGGERIC V22.5 RENDER CHECKS PASSED\n"; }
    return failures == 0 ? 0 : 1;
}
