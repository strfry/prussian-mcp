/**
 * Chat Engine - Browser-side orchestrator for LLM conversation loop
 * Ports logic from prussian_engine/chat.py to JavaScript
 * Manages tool-calling loop, streaming, and response parsing
 */

class ChatEngine {
    constructor(mcpClient) {
        this.mcpClient = mcpClient;
        this.maxIterations = 10;
        this.baseUrl = mcpClient.baseUrl;
    }

    /**
     * Send a message and get streaming response
     * @param {string} message - User message
     * @param {string} language - Response language ('de' or 'lt')
     * @param {Array} history - Conversation history
     * @param {object} callbacks - Streaming callbacks
     * @returns {Promise<object>} - Parsed response
     */
    async sendMessage(message, language = 'de', history = [], callbacks = {}) {
        const {
            onContentDelta = () => {},
            onReasoningDelta = () => {},
            onToolCall = () => {},
            onToolResult = () => {},
            onDone = () => {}
        } = callbacks;

        const messages = this._buildMessages(message, language, history);
        const debugInfo = {
            query: message,
            toolCalls: [],
            results: [],
            usedWords: [],
            reasoning: []
        };

        let fullContent = '';
        let fullReasoning = '';
        let allUsedWords = new Set();
        let iteration = 0;

        while (iteration < this.maxIterations) {
            iteration++;
            console.log(`[ChatEngine] Turn ${iteration}: Calling LLM`);

            const tools = this.mcpClient.getToolsOpenAIFormat();
            const streamEvents = await this._streamCompletion(messages, tools, language);

            let finishReason = null;
            let toolCalls = [];

            for (const event of streamEvents) {
                switch (event.type) {
                    case 'content_delta':
                        fullContent += event.data.content;
                        onContentDelta(event.data.content, fullContent);
                        break;
                    case 'reasoning_delta':
                        fullReasoning += event.data.content;
                        onReasoningDelta(event.data.content, fullReasoning);
                        break;
                    case 'tool_call_delta':
                        if (!toolCalls[event.data.index]) {
                            toolCalls[event.data.index] = event.data.tool_call;
                        } else {
                            const tc = toolCalls[event.data.index];
                            if (event.data.tool_call.function) {
                                tc.function.arguments += event.data.tool_call.function.arguments || '';
                            }
                        }
                        break;
                    case 'done':
                        finishReason = event.data.finish_reason;
                        break;
                    case 'error':
                        console.error('[ChatEngine] Stream error:', event.data.error);
                        throw new Error(event.data.error);
                }
            }

            onDone();

            const assistantMsg = {
                role: 'assistant',
                content: fullContent
            };

            if (fullReasoning) {
                assistantMsg.reasoning_content = fullReasoning;
                debugInfo.reasoning.push({ turn: iteration, reasoning: fullReasoning });
            }

            if (toolCalls.length > 0) {
                assistantMsg.tool_calls = toolCalls.map((tc, idx) => ({
                    id: `call_${iteration}_${idx}`,
                    type: 'function',
                    function: tc.function
                }));
            }

            messages.push(assistantMsg);

            if (!toolCalls.length) {
                console.log('[ChatEngine] No tool calls, final response');
                break;
            }

            console.log(`[ChatEngine] ${toolCalls.length} tool call(s)`);

            for (const tc of toolCalls) {
                const toolName = tc.function.name;
                const args = JSON.parse(tc.function.arguments || '{}');
                const callId = `call_${iteration}_${toolCalls.indexOf(tc)}`;

                onToolCall({ name: toolName, arguments: args, id: callId });

                try {
                    const result = await this.mcpClient.callTool(toolName, args);

                    debugInfo.toolCalls.push({
                        name: toolName,
                        input: args,
                        result: result
                    });

                    if (Array.isArray(result)) {
                        debugInfo.results.push(...result);
                    }

                    if (toolName === 'get_word_forms' && result.lemma) {
                        allUsedWords.add(result.lemma);
                    } else if (toolName === 'lookup_prussian_word' && Array.isArray(result)) {
                        result.forEach(r => { if (r.word) allUsedWords.add(r.word); });
                    }

                    onToolResult({ name: toolName, result, id: callId });

                    messages.push({
                        role: 'tool',
                        tool_call_id: callId,
                        content: JSON.stringify(result)
                    });
                } catch (err) {
                    console.error(`[ChatEngine] Tool ${toolName} failed:`, err);
                    messages.push({
                        role: 'tool',
                        tool_call_id: callId,
                        content: JSON.stringify({ error: err.message })
                    });
                }
            }
        }

        const { prussian, translation } = this._parseResponse(fullContent, language);

        return {
            prussian,
            translation,
            usedWords: Array.from(allUsedWords),
            debugInfo: {
                ...debugInfo,
                usedWords: Array.from(allUsedWords)
            }
        };
    }

    /**
     * Build messages array for LLM
     */
    _buildMessages(message, language, history) {
        return [...history, { role: 'user', content: message }];
    }

    /**
     * Stream completion from LLM proxy (custom SSE format)
     */
    async _streamCompletion(messages, tools, language) {
        const response = await fetch(`${this.baseUrl}/api/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages,
                tools,
                temperature: 0.7,
                max_tokens: 2000,
                language
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const events = [];
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            let currentEvent = null;
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const dataStr = line.slice(6).trim();
                    if (!dataStr) continue;

                    try {
                        const data = JSON.parse(dataStr);
                        events.push({ type: currentEvent, data });
                    } catch (e) {
                        console.error('[ChatEngine] Failed to parse SSE data:', dataStr);
                    }
                    currentEvent = null;
                }
            }
        }

        return events.filter(e => e.data !== null);
    }

    /**
     * Parse LLM response to extract Prussian text and translation
     */
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
    module.exports = { ChatEngine };
}
