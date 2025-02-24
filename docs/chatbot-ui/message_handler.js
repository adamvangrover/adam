// message_handler.js

// Message Handling Module
const messageHandler = {
    sendMessage(message, sender) {
        const chatMessage = document.createElement('div');
        chatMessage.classList.add('chat-message', sender);

        // Handle different message types (text, charts, tables, etc.)
        if (typeof message === 'object') {
            // Example: Handle chart data
            if (message.type === 'chart') {
                const chartCanvas = document.createElement('canvas');
                chatMessage.appendChild(chartCanvas);
                //... (use a charting library to draw the chart)
            }
            //... (handle other message types)
        } else {
            // Sanitize output before displaying
            const sanitizedMessage = this.sanitizeOutput(message);
            chatMessage.textContent = sanitizedMessage;
        }

        chatWindow.appendChild(chatMessage);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    },

    sanitizeOutput(message) {
        //... (implement output sanitization to prevent XSS vulnerabilities)
        // Example: Escape HTML special characters
        return message.replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
    },

    //... (add more message handling functions)
};
