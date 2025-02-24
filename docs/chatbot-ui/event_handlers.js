// event_handlers.js

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
