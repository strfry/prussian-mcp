#!/usr/bin/env node
/**
 * Test ReAct-style tool calling for models without native function calling.
 * Tools are described in the prompt, model outputs <tool_call>...</tool_call>,
 * we parse and execute, then feed results back.
 */

import { EventSource } from 'eventsource';
globalThis.EventSource = EventSource;

const { MCPClient } = await import('../ui/mcp-client.js');

const MCP_URL  = process.argv[2] || 'http://localhost:8000';
const LLM_URL  = process.argv[3] || 'http://localhost:8001/v3';
const MESSAGE  = process.argv[4] || 'Wie sagt man "Haus" auf Preußisch?';
const MODEL    = process.env.OPENAI_MODEL || 'eurollm-22b-instruct-int8';

async function fetchSystemPrompt(mcp, language = 'de') {
    const result = await mcp.getPrompt('chat', { language });
    // MCP returns messages with role "user"/"assistant", content as TextContent
    const msg = result.messages?.[0];
    if (!msg) throw new Error('No messages in prompt');
    // content can be a string or {type: "text", text: "..."}
    const content = msg.content;
    if (typeof content === 'string') return content;
    if (content?.type === 'text') return content.text;
    if (content?.text) return content.text;
    throw new Error('Could not extract prompt text');
}

async function streamChat(messages) {
    const response = await fetch(`${LLM_URL}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: MODEL,
            messages,
            temperature: 0.7,
            max_tokens: 500,
            stream: true,
            // NOTE: no 'tools' parameter - using ReAct prompting instead
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';
    let aborted = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            if (!dataStr) continue;
            try {
                const chunk = JSON.parse(dataStr);
                const delta = chunk.choices?.[0]?.delta?.content;
                if (delta) {
                    content += delta;
                    process.stdout.write(delta);

                    // Cut off after </tool_call> - model tends to hallucinate after it
                    const endTag = content.indexOf('</tool_call>');
                    if (endTag !== -1) {
                        content = content.slice(0, endTag + '</tool_call>'.length);
                        aborted = true;
                    }
                }
            } catch {}
        }
        if (aborted) {
            reader.cancel();
            break;
        }
    }
    return content;
}

function parseToolCalls(text) {
    const calls = [];
    
    const tagRegex = /(?:```\s*)?<tool_call>\s*([\s\S]*?)\s*<\/tool_call>(?:\s*```)?/g;
    let match;
    while ((match = tagRegex.exec(text)) !== null) {
        const content = match[1].trim();
        const parsed = parseFlexibleFormat(content);
        if (parsed) {
            calls.push(parsed);
        } else {
            console.error('  [parse error]', content);
        }
    }
    
    // Fallback: find bare JSON tool call without <tool_call> tags
    if (calls.length === 0) {
        const bareJsonMatch = text.match(/\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}/);
        if (bareJsonMatch) {
            try {
                calls.push({ name: bareJsonMatch[1], arguments: JSON.parse(bareJsonMatch[2]) });
            } catch {}
        }
    }
    
    return calls;
}

function parseFlexibleFormat(text) {
    text = text.trim();
    
    try {
        const parsed = JSON.parse(text);
        if (parsed.name && parsed.arguments) {
            return parsed;
        }
    } catch {}
    
    const nameMatch = text.match(/"name"\s*:\s*"([^"]+)"/);
    const argsMatch = text.match(/"arguments"\s*:\s*(\{[^}]+\}|\[[^\]]+\])/);
    if (nameMatch && argsMatch) {
        return { name: nameMatch[1], arguments: JSON.parse(argsMatch[1]) };
    }
    
    const simpleMatch = text.match(/^(\w+)\s*\(\s*([^)]+)\s*\)$/);
    if (simpleMatch) {
        const funcName = simpleMatch[1];
        const argsStr = simpleMatch[2];
        const args = {};
        const kvPattern = /(\w+)\s*=\s*(?:"([^"]*)"|(\d+))/g;
        let kv;
        while ((kv = kvPattern.exec(argsStr)) !== null) {
            args[kv[1]] = kv[3] !== undefined ? parseInt(kv[3]) : kv[2];
        }
        return { name: funcName, arguments: args };
    }
    
    return null;
}

async function main() {
    console.log(`--- ReAct Tool-Calling Test ---`);
    console.log(`MCP:   ${MCP_URL}`);
    console.log(`LLM:   ${LLM_URL} (${MODEL})`);
    console.log(`Message: "${MESSAGE}"\n`);

    // Connect MCP for tool execution and prompt, LLM calls go directly to OVMS
    const mcp = new MCPClient(MCP_URL);
    await mcp.connect();

    // Fetch system prompt from MCP (includes tool descriptions and ReAct format)
    const systemPrompt = await fetchSystemPrompt(mcp, 'de');
    console.log(`System prompt (${systemPrompt.length} chars):\n${systemPrompt.slice(0, 200)}...\n`);

    const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: MESSAGE }
    ];

    const MAX_TURNS = 10;
    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        console.log(`\n--- Turn ${turn} ---\n`);

        const response = await streamChat(messages);
        messages.push({ role: 'assistant', content: response });

        // Check for tool calls
        const toolCalls = parseToolCalls(response);
        if (toolCalls.length === 0) {
            console.log('\n\n[No tool calls - final answer]');
            break;
        }

        // Execute tool calls
        let toolResults = '';
        for (const tc of toolCalls) {
            console.log(`\n\n  [executing] ${tc.name}(${JSON.stringify(tc.arguments)})`);
            try {
                const result = await mcp.callTool(tc.name, tc.arguments);
                const resultStr = JSON.stringify(result, null, 2);
                console.log(`  [result] ${resultStr.slice(0, 200)}...`);
                toolResults += `<tool_result>\n${resultStr}\n</tool_result>\n`;
            } catch (err) {
                toolResults += `<tool_result>\nError: ${err.message}\n</tool_result>\n`;
            }
        }

        // Feed results back
        messages.push({ role: 'user', content: toolResults });
    }

    mcp.disconnect();
    console.log('\nDone.');
    process.exit(0);
}

main().catch(err => { console.error('FATAL:', err); process.exit(1); });
