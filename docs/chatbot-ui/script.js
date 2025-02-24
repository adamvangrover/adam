// 1. Core Modules

// Import message handling functions from message_handler.js
import { sendMessage, sanitizeOutput } from './message_handler.js';

// Import API communication functions from api_communicator.js
import { sendMessageToAPI } from './api_communicator.js';

// Import UI update functions from ui_updater.js
import {
    updateChatWindow,
    updateKnowledgeGraphVisualization,
    displayMarkdownContent,
    toggleAdvancedMode
} from './ui_updater.js';

// Import analysis modules from analysis_modules.js
import {
    generateMarketSentimentAnalysis,
    generateMacroeconomicAnalysis,
    generateGeopoliticalRiskAnalysis,
    generateIndustryAnalysis,
    generateFundamentalAnalysis,
    generateTechnicalAnalysis,
    generatePortfolioOptimization
} from './analysis_modules.js';

// Import UI components from ui_components.js
import {
    chatWindow,
    userInput,
    sendButton,
    knowledgeGraphVisualization,
    markdownViewer,
    advancedModeButton
} from './ui_components.js';

// Import menu functions from menu_functions.js
import {
    showMainMenu,
    handleMenuSelection,
    showMarketAnalysisMenu,
    showInvestmentResearchMenu,
    showPortfolioManagementMenu
} from './menu_functions.js';

// Import utility functions from utils.js
import { displayMessage } from './utils.js';

// 2. Event Handlers

// 4. Initialization and Event Handling
sendButton.addEventListener('click', () => {
    const userMessage = userInput.value;
    displayMessage(userMessage, 'user');
    userInput.value = '';

    // Check if it's the initial interaction
    if (isFirstInteraction) {
        isFirstInteraction = false;
        showMainMenu();
    } else {
        // Handle menu selection or other user input
        if (currentConversation.length > 0) {
            const response = handleREADMEresponse(userMessage);
            displayMessage(response, 'bot');
        } else {
            // Check if the user message matches any button functionality
            if (userMessage.toLowerCase().includes('market sentiment')) {
                showMarketSentiment();
            } else if (userMessage.toLowerCase().includes('macroeconomic')) {
                showMacroeconomicAnalysis();
            } else if (userMessage.toLowerCase().includes('geopolitical')) {
                showGeopoliticalRisks();
            } else if (userMessage.toLowerCase().includes('industry analysis')) {
                showIndustryAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('fundamental analysis')) {
                showFundamentalAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('technical analysis')) {
                showTechnicalAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('portfolio optimization')) {
                showPortfolioOptimization();
            } else {
                // If no button functionality is matched, send the message to the API
                apiCommunicator.sendMessage(userMessage, (response) => {
                    uiUpdater.updateChatWindow(response, 'bot');
                });
            }
        }
    }
});

//... (Other event handlers)

// 3. Helper Functions

// Function to display README content
function getREADMEContent() {
    // Initialize conversation state
    let currentREADMEQuestion = 0;
    const readmeQuestions = [
        "Would you like a quick overview of what I can do? (yes/no)",
        "What would you like to know more about? (market analysis, investment research, risk management, personalized insights)",
        "Would you like to dive deeper into any specific feature? (yes/no)",
        "Perhaps you're curious about the technology behind me? (architecture, agents, data sources)",
        "Are you interested in exploring the code and contributing to my development? (repository, deployment, contributing)",
        "Is there anything else I can help you with? (yes/no)"
    ];

    // Function to handle user response and provide next question
    function handleREADMEresponse(userResponse) {
        const response = userResponse.toLowerCase();
        let message = "";

        switch (currentREADMEQuestion) {
            case 0:
                if (response.includes('yes')) {
                    message = `I'm Adam v17.0, an AI-powered system designed to provide sophisticated investors with actionable insights and personalized investment recommendations. My capabilities include market analysis, investment research, risk management, and personalized insights. I'm continuously learning and growing!`;
                } else if (response.includes('no')) {
                    message = "No problem. Feel free to ask me anything about the Adam v17.0 project or its features.";
                } else {
                    message = "I didn't understand your response. Please answer with 'yes' or 'no'.";
                    currentREADMEQuestion--;
                }
                break;
            case 1:
                if (response.includes('market analysis')) {
                    message = "I can provide market sentiment analysis, macroeconomic analysis, and geopolitical risk assessment. Would you like to know more about any of these? (sentiment, macro, geopolitical)";
                } else if (response.includes('investment research')) {
                    message = "I can analyze different industries, perform fundamental analysis on companies, and provide technical analysis tools. What would you like to explore further? (industry, fundamental, technical)";
                } else if (response.includes('risk management')) {
                    message = "I can assess various types of investment risk, provide risk mitigation strategies, and simulate market conditions using a World Simulation Model. Would you like to know more about any of these? (risk assessment, mitigation, simulation)";
                } else if (response.includes('personalized insights')) {
                    message = "I can offer personalized investment recommendations and generate customized newsletters based on your risk tolerance and investment goals. Would you like to know more about either of these? (recommendations, newsletters)";
                } else {
                    message = "I'm not sure I understand. Please ask about one of the following: market analysis, investment research, risk management, or personalized insights.";
                    currentREADMEQuestion--;
                }
                break;
            case 2:
                if (response.includes('yes')) {
                    message = "Great! Which feature would you like to dive deeper into? (market sentiment, macroeconomic analysis, geopolitical risk assessment, industry analysis, fundamental analysis, technical analysis, risk assessment, risk mitigation, simulation, recommendations, newsletters)";
                } else if (response.includes('no')) {
                    message = "No problem. Perhaps you're curious about the technology behind me? (architecture, agents, data sources)";
                } else {
                    message = "I didn't understand your response. Please answer with 'yes' or 'no'.";
                    currentREADMEQuestion--;
                }
                break;
            case 3:
                if (response.includes('architecture')) {
                    message = "I'm built on a modular, agent-based architecture, with specialized agents for different tasks. Would you like to know more about the agents, data sources, or analysis modules? (agents, data sources, analysis modules)";
                } else if (response.includes('agents')) {
                    message = "I have various agents, including a Market Sentiment Agent, Macroeconomic Analysis Agent, Geopolitical Risk Agent, and more. Would you like to know more about any specific agent? (yes/no)";
                } else if (response.includes('data sources')) {
                    message = "I gather data from various sources, including financial news APIs, social media, government statistics, and market data providers. Would you like to know more about any specific data source? (yes/no)";
                } else {
                    message = "I'm not sure I understand. Please ask about one of the following: architecture, agents, or data sources.";
                    currentREADMEQuestion--;
                }
                break;
            case 4:
                if (response.includes('repository')) {
                    message = "You can find the full repository and its detailed README here: [https://github.com/adamvangrover/adam](https://github.com/adamvangrover/adam)";
                } else if (response.includes('deployment')) {
                    message = "I can be deployed in various ways, including direct deployment, virtual environment, Docker container, or cloud platforms. Would you like to know more about any of these? (direct, virtual, docker, cloud)";
                } else if (response.includes('contributing')) {
                    message = "Contributions are welcome! You can contribute by reporting bugs, suggesting enhancements, or submitting code changes. See the [CONTRIBUTING.md](https://github.com/adamvangrover/adam/blob/main/CONTRIBUTING.md) file for more details.";
                } else {
                    message = "I'm not sure I understand. Please ask about one of the following: repository, deployment, or contributing.";
                    currentREADMEQuestion--;
                }
                break;
            case 5:
                if (response.includes('yes')) {
                    message = "Great! What else would you like to know about? (market analysis, investment research, risk management, personalized insights, architecture, agents, data sources, repository, deployment, contributing)";
                    currentREADMEQuestion = 1; // Go back to exploring features
                } else if (response.includes('no')) {
                    message = "Thanks for exploring the Adam v17.0 chatbot demo! Feel free to reach out if you have any further questions.";
                } else {
                    message = "I didn't understand your response. Please answer with 'yes' or 'no'.";
                    currentREADMEQuestion--;
                }
                break;
        }

        // Update conversation state and prompt the next question
        currentREADMEQuestion++;
        if (currentREADMEQuestion < readmeQuestions.length) {
            message += " " + readmeQuestions[currentREADMEQuestion];
        }

        return message;
    }

    // Start the README conversation
    return `Welcome to the Adam v17.0 chatbot demo! This chatbot provides a glimpse into the capabilities of the full Adam v17.0 system, which is designed to be a comprehensive financial analysis tool for sophisticated investors.

${readmeQuestions[currentREADMEQuestion]}
`;
}

function generateGenericResponse(message) {
    const responses = [
        "Interesting...",
        "I see...",
        "Tell me more.",
        "That's insightful.",
        "I'm learning something new every day.",
    ];
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];

    // Add some personality and humor
    if (message.toLowerCase().includes('joke')) {
        return "Why don't scientists trust atoms? Because they make up everything!";
    } else if (message.toLowerCase().includes('weather')) {
        const weatherConditions = ["sunny", "cloudy", "rainy", "snowy"][Math.floor(Math.random() * 4)];
        return `The weather in New York is currently ${weatherConditions}.`;
    } else {
        return randomResponse;
    }
}

// 3. UI Components
const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const knowledgeGraphVisualization = document.getElementById('knowledge-graph-visualization');
const markdownViewer = document.getElementById('markdown-viewer');
const advancedModeButton = document.getElementById('advanced-mode-button');
let isAdvancedMode = false;  // Flag to track advanced mode
let isFirstInteraction = true; // Flag to track the initial interaction
let currentConversation =; // Initialize currentConversation as an empty array

// Function to toggle advanced mode
function toggleAdvancedMode() {
    isAdvancedMode =!isAdvancedMode;
    // Update UI based on advanced mode status
    if (isAdvancedMode) {
        // Show advanced features and configurations
        //... (Implementation for showing advanced features)
        advancedModeButton.textContent = "Basic Mode";
    } else {
        // Hide advanced features and configurations
        //... (Implementation for hiding advanced features)
        advancedModeButton.textContent = "Advanced Mode";
    }
}

// Function to display the main menu
function showMainMenu() {
    const menuOptions = [
        "Adam v17.0 Overview",
        "Market Analysis",
        "Investment Research",
        "Portfolio Management",
        "README and Documentation"
    ];
    let menuMessage = "Here's what I can do. What would you like to explore?\n\n";
    for (let i = 0; i < menuOptions.length; i++) {
        menuMessage += `${i + 1}. ${menuOptions[i]}\n`;
    }
    displayMessage(menuMessage, 'bot');
}

// Function to handle menu selection
function handleMenuSelection(userResponse) {
    const response = userResponse.toLowerCase();
    if (response.includes('1') || response.includes('overview')) {
        displayMessage("Adam v17.0 is a sophisticated AI for financial market analysis and personalized insights. It's designed to help investors like you make informed decisions.", 'bot');
    } else if (response.includes('2') || response.includes('market analysis')) {
        showMarketAnalysisMenu();
    } else if (response.includes('3') || response.includes('investment research')) {
        showInvestmentResearchMenu();
    } else if (response.includes('4') || response.includes('portfolio management')) {
        showPortfolioManagementMenu();
    } else if (response.includes('5') || response.includes('readme') || response.includes('documentation')) {
        currentConversation.push(true); // Start the README conversation
        displayMessage(getREADMEContent(), 'bot');
    } else {
        displayMessage("I'm sorry, I didn't understand your request. Please try again.", 'bot');
    }
}

function showMarketAnalysisMenu() {
    const marketAnalysisOptions = [
        "Market Sentiment Analysis",
        "Macroeconomic Analysis",
        "Geopolitical Risk Assessment"
    ];
    let menuMessage = "What kind of market analysis are you interested in?\n\n";
    for (let i = 0; i < marketAnalysisOptions.length; i++) {
        menuMessage += `${i + 1}. ${marketAnalysisOptions[i]}\n`;
    }
    displayMessage(menuMessage, 'bot');
}

function showInvestmentResearchMenu() {
    const researchOptions = [
        "Industry Analysis",
        "Fundamental Analysis",
        "Technical Analysis"
    ];
    let menuMessage = "Investment research, exciting! What would you like to explore?\n\n";
    for (let i = 0; i < researchOptions.length; i++) {
        menuMessage += `${i + 1}. ${researchOptions[i]}\n`;
    }
    displayMessage(menuMessage, 'bot');
}

function showPortfolioManagementMenu() {
    const portfolioOptions = [
        "Portfolio Optimization",
        "Risk Assessment"
    ];
    let menuMessage = "Ah, portfolio management. A wise choice. What would you like to do?\n\n";
    for (let i = 0; i < portfolioOptions.length; i++) {
        menuMessage += `${i + 1}. ${portfolioOptions[i]}\n`;
    }
    displayMessage(menuMessage, 'bot');
}

//... (Additional menu functions)

// 4. Initialization and Event Handling
sendButton.addEventListener('click', () => {
    const userMessage = userInput.value;
    displayMessage(userMessage, 'user');
    userInput.value = '';

    // Check if it's the initial interaction
    if (isFirstInteraction) {
        isFirstInteraction = false;
        showMainMenu();
    } else {
        // Handle menu selection or other user input
        if (currentConversation.length > 0) {
            const response = handleREADMEresponse(userMessage);
            displayMessage(response, 'bot');
        } else {
            // Check if the user message matches any button functionality
            if (userMessage.toLowerCase().includes('market sentiment')) {
                showMarketSentiment();
            } else if (userMessage.toLowerCase().includes('macroeconomic')) {
                showMacroeconomicAnalysis();
            } else if (userMessage.toLowerCase().includes('geopolitical')) {
                showGeopoliticalRisks();
            } else if (userMessage.toLowerCase().includes('industry analysis')) {
                showIndustryAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('fundamental analysis')) {
                showFundamentalAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('technical analysis')) {
                showTechnicalAnalysis(userMessage);
            } else if (userMessage.toLowerCase().includes('portfolio optimization')) {
                showPortfolioOptimization();
            } else {
                // If no button functionality is matched, send the message to the API
                apiCommunicator.sendMessage(userMessage, (response) => {
                    uiUpdater.updateChatWindow(response, 'bot');
                });
            }
        }
    }
});

// Helper function to display messages
function displayMessage(message, sender) {
    const chatMessage = document.createElement('div');
    chatMessage.classList.add('chat-message', sender);
    chatMessage.textContent = message;
    chatWindow.appendChild(chatMessage);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
}

// 5. Message Processing and Response Generation
//... (Implementation for processing messages and generating responses)
// This could involve calling the API, processing the response, and displaying it in the chat window.

// 6. Dynamic Content Rendering and UI Updates
//... (Implementation for rendering dynamic content and updating the UI)
// This could involve updating the knowledge graph visualization, displaying markdown content,
// or showing/hiding advanced features based on user interactions.

// 7. Error Handling and Logging
//... (Implement error handling and logging)
// This could involve catching errors from the API or other modules and displaying appropriate messages to the user.
