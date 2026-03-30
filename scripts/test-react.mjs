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
const MODEL    = process.env.OPENAI_MODEL || 'eurollm-22b-instruct-int4';

function compactToolResults(results) {
    const lines = [];
    for (const { name, args, result } of results) {
        if (name === 'lookup_prussian_word') {
            const r = result;
            if (r.word) {
                let line = `${args.word} → ${r.word} [${r.de}] (${r.en})`;
                if (r.gender) line += ` {${r.gender}}`;
                if (r.matched_form) line += ` matched: ${r.matched_form}`;
                if (r.matched_paths) line += ` (${r.matched_paths.join(', ')})`;
                lines.push(line);
            } else if (Array.isArray(r)) {
                for (const entry of r.slice(0, 5)) {
                    lines.push(`${args.word} → ${entry.word} [${entry.de}] (${entry.en})`);
                }
            } else {
                lines.push(`${args.word} → not found`);
            }
        } else if (name === 'search_dictionary') {
            const entries = Array.isArray(result) ? result : [];
            lines.push(`search "${args.query}":`);
            for (const e of entries.slice(0, 5)) {
                lines.push(`  ${e.word} [${e.de}] (${e.en})`);
            }
        } else if (name === 'get_word_forms') {
            lines.push(`forms of ${args.lemma}: ${JSON.stringify(result)}`);
        }
    }
    return lines.join('\n');
}

async function fetchSystemPrompt(mcp, name = 'chat', language = 'de') {
    const result = await mcp.getPrompt(name, { language });
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
    
    // Fallback: find all bare JSON tool calls without <tool_call> tags
    if (calls.length === 0) {
        const bareJsonRegex = /\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}/g;
        let bareMatch;
        while ((bareMatch = bareJsonRegex.exec(text)) !== null) {
            try {
                calls.push({ name: bareMatch[1], arguments: JSON.parse(bareMatch[2]) });
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

/**
 * Run a ReAct loop: send messages to LLM, parse tool calls, execute via MCP, repeat.
 * Returns collected tool results.
 */
async function reactLoop(mcp, messages, label, maxTurns = 10) {
    const collected = [];
    const seen = new Set();
    let dupStreak = 0;

    for (let turn = 1; turn <= maxTurns; turn++) {
        console.log(`\n--- ${label} turn ${turn} ---\n`);

        const response = await streamChat(messages);
        messages.push({ role: 'assistant', content: response });

        const toolCalls = parseToolCalls(response);
        if (toolCalls.length === 0) break;

        let newResults = false;
        for (const tc of toolCalls) {
            const key = `${tc.name}:${JSON.stringify(tc.arguments)}`;
            if (seen.has(key)) {
                console.log(`\n  [skip duplicate] ${tc.name}(${JSON.stringify(tc.arguments)})`);
                continue;
            }
            seen.add(key);
            newResults = true;
            console.log(`\n\n  [executing] ${tc.name}(${JSON.stringify(tc.arguments)})`);
            try {
                const result = await mcp.callTool(tc.name, tc.arguments);
                collected.push({ name: tc.name, args: tc.arguments, result });
                const resultStr = JSON.stringify(result, null, 2);
                console.log(`  [result] ${resultStr.slice(0, 200)}...`);
            } catch (err) {
                collected.push({ name: tc.name, args: tc.arguments, result: `Error: ${err.message}` });
            }
        }

        dupStreak = newResults ? 0 : dupStreak + 1;
        if (dupStreak >= 2) {
            console.log('\n  [stuck — ending loop]');
            break;
        }

        // Re-compact all results so far as context for the next turn
        messages.push({ role: 'user', content: `<context>\n${compactToolResults(collected)}\n</context>` });
    }
    return collected;
}

async function main() {
    console.log(`--- ReAct Tool-Calling Test ---`);
    console.log(`MCP:   ${MCP_URL}`);
    console.log(`LLM:   ${LLM_URL} (${MODEL})`);
    console.log(`Message: "${MESSAGE}"\n`);

    const mcp = new MCPClient(MCP_URL);
    await mcp.connect();

    // Phase 1: Understand user input
    const chatPrompt = await fetchSystemPrompt(mcp, 'chat', 'de');
    console.log(`Chat prompt (${chatPrompt.length} chars)\n`);

    const phase1Messages = [
        { role: 'system', content: chatPrompt },
        { role: 'user', content: MESSAGE }
    ];
    const phase1Results = await reactLoop(mcp, phase1Messages, 'Understand');

    // Phase 2: Plan — single LLM call to get a German response
    const planPrompt = await fetchSystemPrompt(mcp, 'plan', 'de');
    const compacted1 = compactToolResults(phase1Results);
    console.log(`\n=== Phase 2: Plan ===\n`);

    // Translate input to user language from lookup results
    const translated = phase1Results
        .filter(r => r.name === 'lookup_prussian_word')
        .map(r => {
            const res = r.result;
            if (Array.isArray(res)) return res[0]?.de || r.args.word;
            return res.de || r.args.word;
        });

    const planMessages = [
        { role: 'system', content: planPrompt },
        { role: 'user', content: translated.join(' ') }
    ];
    const germanResponse = await streamChat(planMessages);
    console.log(`\n\n[German response: "${germanResponse.trim()}"]\n`);

    // Phase 3: Search — extract content words and look them up
    console.log(`\n=== Phase 3: Search ===\n`);
    const stopwords = new Set(['der','die','das','ein','eine','und','oder','ist','sind','war','hat','ich','du','er','sie','es','wir','ihr','mein','dein','sein','nicht','auch','an','in','auf','mit','von','zu','für','als','nach','bei','aus','um','über','vor','zum','zur','den','dem','des','einen','einem','einer','dass','wenn','aber','so','wie','noch','schon','sehr','nur','doch','ja','nein','kein','keine','wird','kann','will','muss','soll','darf','sich','man','was','wer','wo','hier','da','nun','dann','mal','mehr','uns','euch']);
    const words = germanResponse
        .replace(/[.,!?;:"""()[\]{}]/g, ' ')
        .split(/\s+/)
        .map(w => w.toLowerCase())
        .filter(w => w.length > 1 && !stopwords.has(w));
    const uniqueWords = [...new Set(words)];
    console.log(`Content words: ${uniqueWords.join(', ')}\n`);

    const searchResults = [];
    for (const word of uniqueWords) {
        console.log(`  [search] "${word}"`);
        try {
            const result = await mcp.callTool('search_dictionary', { query: word, top_k: 3 });
            searchResults.push({ name: 'search_dictionary', args: { query: word }, result });
            const entries = Array.isArray(result) ? result : [];
            for (const e of entries.slice(0, 3)) {
                console.log(`    → ${e.word} [${e.de}]`);
            }
        } catch (err) {
            console.log(`    → Error: ${err.message}`);
        }
    }

    // Phase 4: Final formulation from all compacted results
    const finalPrompt = await fetchSystemPrompt(mcp, 'final', 'de');
    const allResults = [...phase1Results, ...searchResults];
    const compactedAll = compactToolResults(allResults);
    console.log(`\n=== Final answer (${allResults.length} total lookups) ===\n`);
    console.log(`Context:\n${compactedAll}\n`);
    console.log(`Plan: ${germanResponse.trim()}\n`);

    const finalMessages = [
        { role: 'system', content: finalPrompt },
        { role: 'user', content: `${MESSAGE}\n\nGeplante Antwort: ${germanResponse.trim()}\n\n<context>\n${compactedAll}\n</context>` }
    ];
    await streamChat(finalMessages);

    mcp.disconnect();
    console.log('\nDone.');
    process.exit(0);
}

main().catch(err => { console.error('FATAL:', err); process.exit(1); });
