/**
 * ReAct Engine - Streaming interface for ReAct-style tool calling
 * Shows RAG steps explicitly in the UI
 */

class ReactEngine {
    /**
     * @param {MCPClient} mcpClient - The MCP client for tool execution
     * @param {string} serverUrl - MCP server URL (e.g. http://localhost:8001)
     * @param {string} model - Model name
     */
    constructor(mcpClient, serverUrl, model) {
        this.mcpClient = mcpClient;
        this.serverUrl = serverUrl;
        this.model = model;
        this.maxIterations = 10;
    }

    /**
     * Send a message and get streaming response with visible RAG steps
     * @param {string} message - User message
     * @param {string} language - Response language
     * @param {Array} history - Conversation history
     * @param {Array|null} grammar - Grammar sources to inject
     * @param {object} callbacks - Streaming callbacks
     * @returns {Promise<object>} - Parsed response
     */
    async sendMessage(message, language = 'de', history = [], grammar = null, callbacks = {}) {
        const {
            onTurnStart = () => {},
            onTurnEnd = () => {},
            onToolCall = () => {},
            onToolResult = () => {},
            onContentDelta = () => {},
            onDone = () => {}
        } = callbacks;

        const messages = [
            ...history,
            { role: 'user', content: message }
        ];

        const debugInfo = {
            query: message,
            turns: [],
            toolCalls: [],
            results: [],
            usedWords: []
        };

        let fullContent = '';
        let iteration = 0;

        const tools = this.mcpClient?.getToolsOpenAIFormat?.() || undefined;

        while (iteration < this.maxIterations) {
            iteration++;
            onTurnStart({ turn: iteration, message: messages[messages.length - 1] });

            const streamEvents = await this._streamCompletion(messages, language, grammar, tools);

            let toolCalls = [];

            for (const event of streamEvents) {
                switch (event.type) {
                    case 'content_delta':
                        fullContent += event.data.content;
                        onContentDelta(event.data.content, fullContent);
                        break;
                    case 'tool_call_start':
                        // New tool call started (accumulated in _streamCompletion)
                        break;
                    case 'tool_call_delta':
                        // Arguments accumulated in _streamCompletion
                        break;
                    case 'tool_call_end':
                        if (event.data) {
                            toolCalls.push(event.data);
                        }
                        break;
                    case 'done':
                        break;
                }
            }

            if (fullContent.trim()) {
                messages.push({ role: 'assistant', content: fullContent });
            }

            debugInfo.turns.push({
                turn: iteration,
                content: fullContent,
                toolCalls: [...toolCalls]
            });

            onTurnEnd({ turn: iteration, content: fullContent, toolCalls });

            if (toolCalls.length === 0) {
                break;
            }

            // Execute tool calls
            for (const tc of toolCalls) {
                onToolCall({ name: tc.name, arguments: tc.arguments, id: tc.id });

                try {
                    const result = await this.mcpClient.callTool(tc.name, tc.arguments);
                    
                    debugInfo.toolCalls.push({
                        name: tc.name,
                        input: tc.arguments,
                        result: result
                    });

                    if (Array.isArray(result)) {
                        debugInfo.results.push(...result);
                        if (tc.name === 'lookup_prussian_word' || tc.name === 'search_dictionary') {
                            result.forEach(r => {
                                if (r.word) debugInfo.usedWords.push(r.word);
                            });
                        }
                    }

                    onToolResult({ name: tc.name, result, id: tc.id });

                    const resultStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
                    messages.push({
                        role: 'tool',
                        tool_call_id: tc.id,
                        content: resultStr
                    });
                } catch (err) {
                    messages.push({
                        role: 'tool',
                        tool_call_id: tc.id,
                        content: JSON.stringify({ error: err.message })
                    });
                }
            }

            fullContent = '';
        }

        onDone();

        const { prussian, translation } = this._parseResponse(fullContent, language);

        return {
            prussian,
            translation,
            usedWords: debugInfo.usedWords,
            debugInfo
        };
    }

    async _streamCompletion(messages, language = 'de', grammar = null, tools = undefined) {
        const body = {
            model: this.model,
            messages,
            temperature: 0.7,
            max_tokens: 500,
            stream: true,
            language,
            tool_choice: "auto"
        };
        if (grammar?.length) body.grammar = grammar;
        if (tools) body.tools = tools;

        const response = await fetch(`${this.serverUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const events = [];
        let buffer = '';
        const toolCallBuffer = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                for (const idx in toolCallBuffer) {
                    const tc = toolCallBuffer[idx];
                    try { tc.arguments = JSON.parse(tc.arguments); } catch {}
                    events.push({ type: 'tool_call_end', data: tc });
                }
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') {
                    for (const idx in toolCallBuffer) {
                        const tc = toolCallBuffer[idx];
                        try { tc.arguments = JSON.parse(tc.arguments); } catch {}
                        events.push({ type: 'tool_call_end', data: tc });
                    }
                    events.push({ type: 'done', data: {} });
                    break;
                }
                if (!dataStr) continue;

                try {
                    const chunk = JSON.parse(dataStr);
                    const delta = chunk.choices?.[0]?.delta;

                    if (delta?.content) {
                        events.push({ type: 'content_delta', data: { content: delta.content } });
                    }

                    if (delta?.tool_calls) {
                        for (const tc of delta.tool_calls) {
                            if (tc.id || tc.name) {
                                const existing = toolCallBuffer[tc.index];
                                if (existing) {
                                    try { existing.arguments = JSON.parse(existing.arguments); } catch {}
                                    events.push({ type: 'tool_call_end', data: existing });
                                }
                                toolCallBuffer[tc.index] = {
                                    index: tc.index,
                                    id: tc.id || '',
                                    name: tc.name || '',
                                    type: tc.type || 'function',
                                    arguments: tc.function?.arguments || ''
                                };
                                if (tc.name) {
                                    events.push({
                                        type: 'tool_call_start',
                                        data: { index: tc.index, name: tc.name }
                                    });
                                }
                            } else if (tc.function?.arguments && toolCallBuffer[tc.index]) {
                                toolCallBuffer[tc.index].arguments += tc.function.arguments;
                                events.push({
                                    type: 'tool_call_delta',
                                    data: { index: tc.index, arguments: tc.function.arguments }
                                });
                            }
                        }
                    }

                    const finish = chunk.choices?.[0]?.finish_reason;
                    if (finish === 'tool_calls') {
                        for (const idx in toolCallBuffer) {
                            const tc = toolCallBuffer[idx];
                            try { tc.arguments = JSON.parse(tc.arguments); } catch {}
                            events.push({ type: 'tool_call_end', data: tc });
                        }
                    }
                } catch {}
            }
        }

        return events;
    }

    _parseResponse(text, language) {
        const langCode = language === 'lt' ? 'LT' : 'DE';
        const pattern = new RegExp(`\\[${langCode}:\\s*(.+?)\\]`, 's');
        const match = pattern.exec(text);

        if (match) {
            const translation = match[1].trim();
            const prussian = text.slice(0, match.index).trim();
            return { prussian, translation };
        }

        return { prussian: text.trim(), translation: '' };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ReactEngine };
}
