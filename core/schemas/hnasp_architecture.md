graph TD
    UserInput[User Input] --> Meta[Meta Layer: Security & Config]
    Meta --> Context[Context Stream: History]
    
    Context --> Logic[Logic Layer: Fact Extraction]
    Context --> Persona[Persona Layer: BayesACT]
    
    Logic -- "Update Facts" --> Logic
    Persona -- "Calculate Deflection" --> PersonaDynamics
    
    Logic --> Synthesis[Response Synthesis]
    PersonaDynamics --> Synthesis
    Context --> Synthesis
    
    Synthesis --> Output[Agent Output]
