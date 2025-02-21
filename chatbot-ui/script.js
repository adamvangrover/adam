// script.js

// 1. UI Initialization
const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// 2. Menu and Navigation (Simplified for GitHub Demo)
const menu = document.createElement('div');
menu.id = 'menu';
menu.innerHTML = `
  <button onclick="showMarketSentiment()">Market Sentiment</button>
  <button onclick="showMacroeconomicAnalysis()">Macroeconomic Analysis</button>
  <button onclick="showGeopoliticalRisks()">Geopolitical Risks</button>
  <button onclick="showIndustryAnalysis()">Industry Analysis</button>
`;
document.body.insertBefore(menu, chatWindow);

// 3. Message Handling
sendButton.addEventListener('click', () => {
  const userMessage = userInput.value;
  displayMessage(userMessage, 'user');
  userInput.value = ''; // Clear the input field

  // Simulate response based on user input
  if (userMessage.toLowerCase().includes('market sentiment')) {
    showMarketSentiment();
  } else if (userMessage.toLowerCase().includes('macroeconomic')) {
    showMacroeconomicAnalysis();
  } else if (userMessage.toLowerCase().includes('geopolitical')) {
    showGeopoliticalRisks();
  } else if (userMessage.toLowerCase().includes('industry analysis')) {
    showIndustryAnalysis();
  } else {
    displayMessage("I'm still learning. Try asking about market sentiment, macroeconomic analysis, geopolitical risks, or industry analysis.", 'bot');
  }
});

// 4. Simulated Analysis
function showMarketSentiment() {
  const sentiment = ["bullish", "bearish", "neutral"][Math.floor(Math.random() * 3)];
  displayMessage(`The current market sentiment is ${sentiment}.`, 'bot');
}

function showMacroeconomicAnalysis() {
  const gdpGrowth = (Math.random() * 5).toFixed(2);
  const inflation = (Math.random() * 3).toFixed(2);
  displayMessage(`Here's a quick macroeconomic snapshot:
    GDP Growth: ${gdpGrowth}%
    Inflation: ${inflation}%`, 'bot');
}

function showGeopoliticalRisks() {
  const risks = ["Trade tensions", "Political instability", "Supply chain disruptions"][Math.floor(Math.random() * 3)];
  displayMessage(`Key geopolitical risks to watch out for include: ${risks}.`, 'bot');
}

function showIndustryAnalysis() {
  const industries = ["Technology", "Healthcare", "Energy", "Financials"];
  const industry = industries[Math.floor(Math.random() * industries.length)];
  displayMessage(`The ${industry} sector is showing strong growth potential.`, 'bot');
}

// Helper function to display messages
function displayMessage(message, sender) {
  const chatMessage = document.createElement('div');
  chatMessage.classList.add('chat-message', sender);
  chatMessage.textContent = message;
  chatWindow.appendChild(chatMessage);
  chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
}
