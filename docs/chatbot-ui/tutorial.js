function startInteractiveTutorial() {
  introJs().setOptions({
    steps: [
      {
        intro: "Welcome to the Adam v18.0 interactive tutorial! Let's explore its key features and functionalities."
      },
      {
        element: '#landing-page-chatbot',
        intro: "This is the landing page chatbot, your guide to Adam v18.0. It provides a concise overview of the system, setup instructions, user guide, and other essential information."
      },
      {
        element: '.chatbot-options button:first-child',
        intro: "Click this button to watch a video overview of Adam v18.0's capabilities (not available in this demo, but the full system will include a video tutorial). <span class='tutorial-highlight'>Highlighted elements</span> like this will guide your attention throughout the tutorial."
      },
      {
        element: '.chatbot-options button:nth-child(2)',
        intro: "This button starts the interactive tutorial, guiding you through the system's features step-by-step, just like you're experiencing now."
      },
      {
        element: '.chatbot-options button:nth-child(3)',
        intro: "Click this button to take a guided tour of the chatbot UI (not available in this demo, but the full system will have an interactive tour)."
      },
      {
        element: '.chatbot-options button:last-child',
        intro: "This button takes you to the advanced analysis mode, where you can interact with the full Adam v18.0 system, powered by a sophisticated AI engine."
      },
      {
        element: '#functional-chatbot',
        intro: "This is the functional chatbot, where you can interact with the AI engine, access the knowledge graph, run analysis modules, and generate reports. This is where the magic happens!"
      },
      {
        element: '#chat-window',
        intro: "This is the chat window where you can send messages to Adam and receive its responses. Think of it as your direct line to the AI brain."
      },
      {
        element: '#input-area',
        intro: "This is where you craft your messages and questions. Type your queries here and click 'Send' to get Adam's insights."
      },
      {
        element: '#user-input',
        intro: "This is the text box where you type your messages. Be clear and concise, and Adam will do its best to understand and respond."
      },
      {
        element: '#send-button',
        intro: "Click this button to send your message to Adam. It's like hitting the 'Enter' key on your keyboard, but with a bit more flair."
      },
      {
        element: '#menu',
        intro: "These menu options are your gateway to Adam's powerful functionalities. They provide access to various analysis modules, knowledge graph exploration, and system documentation."
      },
      {
        element: '#menu button:first-child',
        intro: "Click this button to analyze market sentiment. Adam will gauge investor sentiment from news articles, social media, and financial forums."
      },
      {
        element: '#menu button:nth-child(2)',
        intro: "This button initiates macroeconomic analysis. Adam will delve into economic indicators, forecasts, and their impact on financial markets."
      },
      {
        element: '#menu button:nth-child(3)',
        intro: "Click this button to assess geopolitical risks. Adam will analyze global events and political developments that could impact your investments."
      },
      {
        element: '#menu button:nth-child(4)',
        intro: "This button opens the door to industry analysis. Adam will provide insights into specific sectors, including trends, company performance, and competitive landscapes."
      },
      {
        element: '#menu button:nth-child(5)',
        intro: "Click this button to perform fundamental analysis. Adam will scrutinize financial statements, key metrics, and valuation models to uncover hidden value and risks."
      },
      {
        element: '#menu button:nth-child(6)',
        intro: "This button unleashes the power of technical analysis. Adam will analyze price charts, technical indicators, and patterns to identify trading signals and opportunities."
      },
      {
        element: '#menu button:nth-child(7)',
        intro: "Click this button to optimize your portfolio. Adam will use sophisticated algorithms to maximize returns and minimize risks based on your preferences."
      },
      {
        element: '#menu button:nth-child(8)',
        intro: "This button grants you access to the knowledge graph. Explore the interconnected web of financial concepts, entities, and events that power Adam's intelligence."
      },
      {
        element: '#menu button:nth-child(9)',
        intro: "Click this button to view the README file, which provides a comprehensive overview of the Adam v18.0 system and its features."
      },
      {
        element: '#menu button:nth-child(10)',
        intro: "This button opens the User Guide, your comprehensive guide to navigating and utilizing Adam's functionalities."
      },
      {
        element: '#menu button:nth-child(11)',
        intro: "Click this button to access the API Documentation, which provides detailed information on how to integrate Adam v18.0 with other systems and applications."
      },
      {
        element: '#content-area',
        intro: "This is where the fruits of Adam's labor are displayed. It showcases the results of your analysis, visualizations, reports, and other valuable insights."
      },
      {
        element: '#markdown-viewer',
        intro: "This section presents the output in a clear and readable format, making it easy to understand and interpret Adam's findings."
      },
      {
        element: '#knowledge-graph-visualization',
        intro: "This section will visualize the knowledge graph, allowing you to explore the interconnectedness of financial concepts and entities (not available in this demo)."
      },
      {
        element: '#react-root',
        intro: "This section can house interactive React components for data visualization and analysis, providing dynamic and engaging insights (not available in this demo)."
      },
      {
        element: '#readme-links',
        intro: "These links offer quick access to important resources, including the README files, user guide, and API documentation. They're your go-to guides for understanding and utilizing Adam v18.0 effectively."
      },
      {
        intro: "Congratulations! You've completed the interactive tutorial. Now you're well-equipped to explore the power of Adam v18.0. Feel free to experiment, ask questions, and unlock the insights that will empower your investment journey. <br><br> <span id='quiz'></span>"
      }
    ]
  }).start().oncomplete(() => {
    // Add a quiz after the tutorial
    const quizContainer = document.getElementById('quiz');
    quizContainer.innerHTML = `
      <p>Quick Quiz:</p>
      <p>Which of these is NOT a core component of the Adam v18.0 system?</p>
      <button onclick="checkAnswer('agent_orchestrator')">Agent Orchestrator</button>
      <button onclick="checkAnswer('knowledge_graph')">Knowledge Graph</button>
      <button onclick="checkAnswer('landing_page_chatbot')">Landing Page Chatbot</button>
      <button onclick="checkAnswer('data_manager')">Data Manager</button>
    `;
  });
}

function checkAnswer(answer) {
  const incorrectAnswer = 'landing_page_chatbot'; 
  if (answer === incorrectAnswer) {
    alert("Correct! The Landing Page Chatbot is part of the UI demo, not the core Adam v18.0 system.");
  } else {
    alert("Not quite. Try reviewing the tutorial and the README again.");
  }
}
