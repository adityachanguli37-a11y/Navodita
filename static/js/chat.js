document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const messagesContainer = document.getElementById("messages-container");
    const typingIndicator = document.getElementById("typing-indicator");
    const clearBtn = document.getElementById("clear-btn");
    const chipsContainer = document.getElementById("chips-container");
    const sendBtn = document.getElementById("send-btn");

    // Client-side session and history tracking
    let conversationId = generateUUID();
    let conversationHistory = [];
    let isRequestInProgress = false;

    function generateUUID() {
        if (typeof crypto !== "undefined" && crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return "conv_" + Math.random().toString(36).substring(2, 15);
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Source display names & icons
    const SOURCE_FORMATTERS = {
        "openai": { label: "OpenAI Intelligence", icon: "✨" },
        "knowledge_base": { label: "Knowledge Base", icon: "📚" },
        "intent_model": { label: "Intent Model", icon: "⚙️" },
        "calculator": { label: "Calculator", icon: "🧮" },
        "ollama": { label: "Local LLM (Ollama)", icon: "🧠" },
        "fallback": { label: "Safe Fallback", icon: "ℹ️" }
    };

    function appendUserMessage(text) {
        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper user-wrapper";
        wrapper.innerHTML = `
            <div class="message-avatar user-avatar">YOU</div>
            <div class="message-body">
                <div class="message-sender-name">You</div>
                <div class="message-bubble user-bubble">
                    ${escapeHTML(text)}
                </div>
            </div>
        `;
        messagesContainer.appendChild(wrapper);
        scrollToBottom();
    }

    function appendBotMessage(data) {
        const answerText = data.answer || data.response || "No response received.";
        const sourceKey = data.source || "fallback";
        const sourceInfo = SOURCE_FORMATTERS[sourceKey] || { label: sourceKey, icon: "🤖" };
        const confPercent = Math.round((data.confidence || (sourceKey === "openai" ? 0.98 : 0.85)) * 100);

        let dotClass = "high";
        if (confPercent < 70) dotClass = "low";
        else if (confPercent < 85) dotClass = "medium";

        const formattedHTML = formatMarkdown(answerText);

        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper bot-wrapper";
        wrapper.innerHTML = `
            <div class="message-avatar bot-avatar">NX</div>
            <div class="message-body">
                <div class="message-sender-name">NexaAI</div>
                <div class="message-bubble bot-bubble">
                    ${formattedHTML}
                </div>
                <div class="message-metadata">
                    <span class="meta-tag source-tag"><span class="meta-icon">${sourceInfo.icon}</span> Source: ${escapeHTML(sourceInfo.label)}</span>
                    <span class="meta-separator">•</span>
                    <span class="meta-tag conf-tag"><span class="conf-dot ${dotClass}"></span> Confidence: ${confPercent}%</span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(wrapper);
        scrollToBottom();
    }

    // Markdown and Code Formatter
    function formatMarkdown(raw) {
        if (!raw) return "";

        // Handle markdown code blocks ```lang ... ```
        const codeBlockRegex = /```([a-zA-Z0-9_\-]*)\n([\s\S]*?)```/g;
        let processed = raw.replace(codeBlockRegex, (match, lang, code) => {
            return `<div class="code-block-container"><div class="code-header">${escapeHTML(lang || "code")}</div><pre><code>${escapeHTML(code.trim())}</code></pre></div>`;
        });

        // Handle inline code `code`
        processed = processed.replace(/`([^`]+)`/g, (match, code) => `<code>${escapeHTML(code)}</code>`);

        // Handle bold **text**
        processed = processed.replace(/\*\*([^*]+)\*\*/g, (match, b) => `<strong>${b}</strong>`);

        // Handle paragraphs and line breaks
        const blocks = processed.split("\n\n");
        return blocks.map(block => {
            if (block.startsWith("<div class=\"code-block-container\"")) {
                return block;
            }
            const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
            if (lines.length === 0) return "";
            // If bullet list
            if (lines.every(l => l.startsWith("- ") || l.startsWith("* "))) {
                const items = lines.map(l => `<li>${l.substring(2)}</li>`).join("");
                return `<ul>${items}</ul>`;
            }
            return `<p>${lines.join("<br>")}</p>`;
        }).join("");
    }

    function escapeHTML(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    async function sendMessage(text) {
        const trimmed = text.trim();
        if (!trimmed || isRequestInProgress) return;

        isRequestInProgress = true;
        if (sendBtn) sendBtn.disabled = true;

        appendUserMessage(trimmed);

        // Capture prior conversation history
        const priorHistory = conversationHistory.slice();
        conversationHistory.push({ role: "user", content: trimmed });

        typingIndicator.classList.remove("hidden");
        scrollToBottom();

        try {
            const resp = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: trimmed,
                    conversation_id: conversationId,
                    history: priorHistory
                })
            });

            const data = await resp.json();
            typingIndicator.classList.add("hidden");

            appendBotMessage(data);
            const botAnswer = data.answer || data.response || "";
            if (botAnswer) {
                conversationHistory.push({ role: "assistant", content: botAnswer });
            }

        } catch (err) {
            typingIndicator.classList.add("hidden");
            appendBotMessage({
                answer: "A communication error occurred while connecting to the AI service. Please verify the server is running.",
                source: "fallback",
                confidence: 0.0,
                success: false
            });
        } finally {
            isRequestInProgress = false;
            if (sendBtn) sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // Form submit
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value;
        userInput.value = "";
        sendMessage(text);
    });

    // Enter key
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Suggestion chips
    chipsContainer.addEventListener("click", (e) => {
        const chip = e.target.closest(".suggestion-chip");
        if (chip) {
            const query = chip.getAttribute("data-query");
            if (query) {
                sendMessage(query);
            }
        }
    });

    // Clear Chat
    clearBtn.addEventListener("click", () => {
        conversationId = generateUUID();
        conversationHistory = [];
        messagesContainer.innerHTML = `
            <div class="message-wrapper bot-wrapper">
                <div class="message-avatar bot-avatar">NX</div>
                <div class="message-body">
                    <div class="message-sender-name">NexaAI</div>
                    <div class="message-bubble bot-bubble">
                        <p>Conversation history cleared. Ready for fresh questions across any topic!</p>
                    </div>
                    <div class="message-metadata">
                        <span class="meta-tag source-tag"><span class="meta-icon">⚙️</span> Source: Intent Model</span>
                        <span class="meta-separator">•</span>
                        <span class="meta-tag conf-tag"><span class="conf-dot high"></span> Confidence: 100%</span>
                    </div>
                </div>
            </div>
        `;
        scrollToBottom();
    });

    userInput.focus();
});
