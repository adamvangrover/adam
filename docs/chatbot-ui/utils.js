// utils.js

// Helper function to display messages
function displayMessage(message, sender) {
    const chatMessage = document.createElement('div');
    chatMessage.classList.add('chat-message', sender);
    chatMessage.textContent = message;
    chatWindow.appendChild(chatMessage);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
}
