/**
 * ReAct Engine - Streaming interface for ReAct-style tool calling
 * Shows RAG steps explicitly in the UI
 */

class ReactEngine {
    /**
     * @param {MCPClient} mcpClient - The MCP client for tool execution
     * @param {string} llmUrl - URL for LLM API
     * @param {string} model - Model name
     */
    constructor(mcpClient, llmUrl, model) {
        this.mcpClient = mcpClient;
        this.llmUrl = llmUrl;
        this.model = model;
        this.maxIterations = 10;
    }

    /**
     * Send a message and get streaming response with visible RAG steps
     * @param {string} message - User message
     * @param {string} language - Response language
     * @param {Array} history - Conversation history
     * @param {object} callbacks - Streaming callbacks
     * @returns {Promise<object>} - Parsed response
     */
    async sendMessage(message, language = 'de', history = [], callbacks = {}) {
        const {
            onTurnStart = () => {},
            onTurnEnd = () => {},
            onToolCall = () => {},
            onToolResult = () => {},
            onContentDelta = () => {},
            onDone = () => {}
        } = callbacks;

        const instructions = this._buildSystemPrompt();
        const messages = [
            { role: 'system', content: instructions },
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

        while (iteration < this.maxIterations) {
            iteration++;
            onTurnStart({ turn: iteration, message: messages[messages.length - 1] });

            const streamEvents = await this._streamCompletion(messages);

            let toolCalls = [];
            let currentToolCall = null;

            for (const event of streamEvents) {
                switch (event.type) {
                    case 'content_delta':
                        fullContent += event.data.content;
                        onContentDelta(event.data.content, fullContent);
                        break;
                    case 'tool_call_start':
                        currentToolCall = {
                            index: event.data.index,
                            name: event.data.name,
                            arguments: ''
                        };
                        break;
                    case 'tool_call_delta':
                        if (currentToolCall) {
                            currentToolCall.arguments += event.data.arguments || '';
                        }
                        break;
                    case 'tool_call_end':
                        if (currentToolCall) {
                            try {
                                currentToolCall.arguments = JSON.parse(currentToolCall.arguments);
                            } catch {}
                            toolCalls.push(currentToolCall);
                            currentToolCall = null;
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

    _buildSystemPrompt() {
        return `You have access to tools to look up Prussian words:

- search_dictionary(query: string, top_k: integer = 10): Semantic search
- lookup_prussian_word(word: string): Look up Prussian words
- get_word_forms(lemma: string): Get word forms

STRICT RULES:
1. lookup_prussian_word: For Prussian words ONLY (ā, ē, ī, ō, ū or endings -un, -wei, -si, -an, -is, -ans).
2. search_dictionary: For German/English/Lithuanian/etc. ONE concept at a time.
3. NEVER break words apart. "minintun" = ONE word.
4. OUTPUT ONLY ONE TOOL CALL. Wait for result.
5. If word not found, move to the next unique word.
6. MUST look up ALL words before summarizing.

EXACT output format - ONLY ONE tool call:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>`;
    }

    async _streamCompletion(messages) {
        const response = await fetch(`${this.llmUrl}/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: this.model,
                messages,
                temperature: 0.7,
                max_tokens: 500,
                stream: true
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

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') {
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
                            if (tc.name) {
                                events.push({
                                    type: 'tool_call_start',
                                    data: { index: tc.index, name: tc.name }
                                });
                            }
                            if (tc.function?.arguments) {
                                events.push({
                                    type: 'tool_call_delta',
                                    data: { index: tc.index, arguments: tc.function.arguments }
                                });
                            }
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
