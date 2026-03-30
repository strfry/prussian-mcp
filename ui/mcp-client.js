/**
 * MCP Client - JavaScript implementation of MCP protocol over SSE
 * Connects to MCP server via SSE endpoint and handles JSON-RPC 2.0 protocol
 */

class MCPClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.eventSource = null;
        this.messagesEndpoint = '/messages';  // Will be updated by server
        this.requestId = 0;
        this.pendingRequests = new Map();
        this.handlers = new Map();
        this.initialized = false;
        this.serverInfo = null;
        this.tools = [];
        this._messageHandler = this._handleMessage.bind(this);
        this._sseHandler = this._handleSSE.bind(this);
    }

    /**
     * Connect to MCP server via SSE and perform handshake
     * @returns {Promise<{serverInfo, tools}>}
     */
    async connect() {
        return new Promise((resolve, reject) => {
            const sseUrl = `${this.baseUrl}/sse`;
            console.log(`[MCP] Connecting to ${sseUrl}...`);

            this.eventSource = new EventSource(sseUrl);

            this.eventSource.onopen = () => {
                console.log('[MCP] SSE connection opened');
            };

            this.eventSource.onerror = (error) => {
                console.error('[MCP] SSE error:', error);
                if (!this.initialized) {
                    reject(new Error('Failed to connect to MCP server'));
                }
            };

            this.eventSource.addEventListener('message', this._messageHandler);
            this.eventSource.addEventListener('sse', this._sseHandler);

            // Wait for endpoint event (with session_id) before sending any requests
            this.eventSource.addEventListener('endpoint', (event) => {
                this.messagesEndpoint = event.data.trim();
                console.log(`[MCP] Endpoint received: ${this.messagesEndpoint}`);
                // Now that we have the endpoint with session_id, initialize
                this._sendInitialize();
            });

            const timeout = setTimeout(() => {
                if (!this.initialized) {
                    this.disconnect();
                    reject(new Error('MCP initialization timeout'));
                }
            }, 30000);

            this._waitForInit = () => {
                clearTimeout(timeout);
                resolve({ serverInfo: this.serverInfo, tools: this.tools });
            };
        });
    }

    /**
     * Disconnect from MCP server
     */
    disconnect() {
        if (this.eventSource) {
            this.eventSource.removeEventListener('message', this._messageHandler);
            this.eventSource.removeEventListener('sse', this._sseHandler);
            this.eventSource.close();
            this.eventSource = null;
        }
        this.initialized = false;
        this.pendingRequests.clear();
    }

    /**
     * Send JSON-RPC request and return promise
     */
    async _sendRequest(method, params = {}) {
        const id = ++this.requestId;
        const request = { jsonrpc: '2.0', id, method, params };

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });

            const postData = JSON.stringify(request);
            // Use the endpoint provided by server, fallback to /messages
            const endpoint = this.messagesEndpoint.startsWith('http')
                ? this.messagesEndpoint
                : `${this.baseUrl}${this.messagesEndpoint}`;

            fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: postData
            }).then(async (response) => {
                if (!response.ok) {
                    this.pendingRequests.delete(id);
                    reject(new Error(`HTTP ${response.status}: ${await response.text()}`));
                }
            }).catch((err) => {
                this.pendingRequests.delete(id);
                reject(err);
            });
        });
    }

    /**
     * Send JSON-RPC notification (no id, no response expected)
     */
    async _sendNotification(method, params = {}) {
        const notification = { jsonrpc: '2.0', method, params };
        const endpoint = this.messagesEndpoint.startsWith('http')
            ? this.messagesEndpoint
            : `${this.baseUrl}${this.messagesEndpoint}`;

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(notification)
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
    }

    /**
     * Send initialize request
     */
    async _sendInitialize() {
        try {
            const result = await this._sendRequest('initialize', {
                protocolVersion: '2024-11-05',
                capabilities: {
                    tools: {},
                    sampling: {}
                },
                clientInfo: {
                    name: 'prussian-chatbot',
                    version: '1.0.0'
                }
            });

            this.serverInfo = result.serverInfo;
            this.tools = result.tools || [];
            this.initialized = true;
            console.log(`[MCP] Initialized with server: ${this.serverInfo?.name}`);
            console.log(`[MCP] Available tools:`, this.tools.map(t => t.name).join(', '));

            await this._sendNotification('notifications/initialized', {});

            // Fetch tool list (initialize response may not include them)
            if (this.tools.length === 0) {
                const toolsResult = await this._sendRequest('tools/list', {});
                this.tools = toolsResult.tools || [];
                console.log(`[MCP] Fetched tools:`, this.tools.map(t => t.name).join(', '));
            }

            this._waitForInit();
        } catch (err) {
            console.error('[MCP] Initialize failed:', err);
            throw err;
        }
    }

    /**
     * Handle incoming SSE messages from server
     */
    _handleSSE(event) {
        try {
            const data = JSON.parse(event.data);
            this._processServerMessage(data);
        } catch (err) {
            console.error('[MCP] Failed to parse SSE message:', err);
        }
    }

    /**
     * Handle incoming message events
     */
    _handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this._processServerMessage(data);
        } catch (err) {
            console.error('[MCP] Failed to parse message:', err);
        }
    }

    /**
     * Process messages from MCP server
     */
    _processServerMessage(message) {
        if (message.id && this.pendingRequests.has(message.id)) {
            const { resolve, reject } = this.pendingRequests.get(message.id);
            this.pendingRequests.delete(message.id);

            if (message.error) {
                reject(new Error(message.error.message || JSON.stringify(message.error)));
            } else {
                resolve(message.result);
            }
        } else if (message.method && message.params) {
            this._handleNotification(message.method, message.params);
        }
    }

    /**
     * Handle notification messages from server
     */
    _handleNotification(method, params) {
        if (method === 'tools/list') {
            this.tools = params.tools || [];
            console.log('[MCP] Tools updated:', this.tools.map(t => t.name).join(', '));
        }

        const handler = this.handlers.get(method);
        if (handler) {
            handler(params);
        }
    }

    /**
     * Register handler for a notification method
     */
    on(method, handler) {
        this.handlers.set(method, handler);
    }

    /**
     * Call a tool on the MCP server
     * @param {string} name - Tool name
     * @param {object} args - Tool arguments
     * @returns {Promise<any>} - Tool result
     */
    async callTool(name, args = {}) {
        if (!this.initialized) {
            throw new Error('MCP client not initialized');
        }

        console.log(`[MCP] Calling tool: ${name}(${JSON.stringify(args)})`);

        try {
            const result = await this._sendRequest('tools/call', {
                name,
                arguments: args
            });

            // Parse MCP content format: extract JSON from text content blocks
            if (result?.content && Array.isArray(result.content)) {
                const parsed = result.content
                    .filter(c => c.type === 'text')
                    .map(c => {
                        try { return JSON.parse(c.text); }
                        catch { return c.text; }
                    });
                // Return single object or array depending on result count
                const unwrapped = parsed.length === 1 ? parsed[0] : parsed;
                console.log(`[MCP] Tool ${name} result:`, unwrapped);
                return unwrapped;
            }

            console.log(`[MCP] Tool ${name} result:`, result);
            return result;
        } catch (err) {
            console.error(`[MCP] Tool ${name} failed:`, err);
            throw err;
        }
    }

    /**
     * List available prompts
     * @returns {Promise<Array>} - List of prompt definitions
     */
    async listPrompts() {
        const result = await this._sendRequest('prompts/list', {});
        return result.prompts || [];
    }

    /**
     * Get a prompt by name with arguments
     * @param {string} name - Prompt name
     * @param {object} args - Prompt arguments
     * @returns {Promise<{description: string, messages: Array}>}
     */
    async getPrompt(name, args = {}) {
        return await this._sendRequest('prompts/get', { name, arguments: args });
    }

    /**
     * List available tools
     */
    getTools() {
        return this.tools;
    }

    /**
     * Get tool definition by name
     */
    getTool(name) {
        return this.tools.find(t => t.name === name);
    }

    /**
     * Get tool definitions in OpenAI function calling format
     */
    getToolsOpenAIFormat() {
        return this.tools.map(tool => ({
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description || '',
                parameters: tool.inputSchema || { type: 'object', properties: {} }
            }
        }));
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MCPClient };
}
