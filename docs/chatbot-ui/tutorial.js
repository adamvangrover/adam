function startInteractiveTutorial() {
  introJs().setOptions({
    steps: [
      {
        intro: "Welcome to the Adam v18.0 interactive tutorial! Let's explore its key features and functionalities."
      },
      {
        element: '#landing-page-chatbot',
        intro: "This is the landing page chatbot, your guide to Adam v18.0. It provides an overview of the system, setup instructions, user guide, and more."
      },
      {
        element: '#functional-chatbot',
        intro: "This is the functional chatbot, where you can interact with the AI engine, access the knowledge graph, run analysis modules, and generate reports."
      },
      {
        element: '#chat-window',
        intro: "This is the chat window where you can interact with Adam and receive responses."
      },
      {
        element: '#input-area',
        intro: "Type your messages and questions here, then click 'Send' to communicate with Adam."
      },
      {
        element: '#menu',
        intro: "These menu options provide access to various analysis modules, knowledge graph exploration, and system documentation."
      },
      {
        element: '#content-area',
        intro: "This area displays the results of your analysis, visualizations, reports, and other information retrieved from Adam."
      },
      {
        element: '#readme-links',
        intro: "These links provide access to the README files, user guide, and API documentation."
      }
    ]
  }).start();
}
