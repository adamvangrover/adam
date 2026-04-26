## Phase 2: THE REFACTOR (Architecture & Modularity)
**Objective**: Reorganize the code to adhere to modern best practices. Improve modularity and readability. Abstract repetitive logic into reusable utility functions or classes, and apply structural design patterns that make it robust and scalable.

### Instructions:
1.  **Extract Repetitive Logic**: Identify code blocks that are repeated multiple times (DRY principle). Extract them into standalone utility functions, helper methods, or separate classes.
2.  **Improve Cohesion**: Ensure that each class or function has a single, well-defined responsibility (Single Responsibility Principle). If a module does too many things, split it up.
3.  **Enhance Modularity**: Organize the code into logical groupings (e.g., separating data access, business logic, and presentation/API layers). Use clear interfaces or abstract base classes if appropriate to define boundaries.
4.  **Apply Design Patterns**: Look for opportunities to apply established structural design patterns (e.g., Factory, Strategy, Observer, Decorator) if they genuinely simplify the design and improve scalability without over-engineering.
5.  **Rename for Clarity**: Rename variables, functions, and classes to be highly descriptive of their purpose and behavior. Favor clarity over brevity.
6.  **Verify Tests**: Update existing tests to reflect the new structure. Ensure all tests pass.