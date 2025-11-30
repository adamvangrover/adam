// menu_functions.js

// Function to display the main menu
function showMainMenu() {
    const menuOptions = [
        "Adam v21.0 Overview",
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
        displayMessage("Adam v15.4 is a sophisticated AI for financial market analysis and personalized insights. It's designed to help investors like you make informed decisions.", 'bot');
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
