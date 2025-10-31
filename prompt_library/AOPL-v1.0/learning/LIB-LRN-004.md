# LIB-LRN-004: Personalized Learning Plan Generator

*   **ID:** `LIB-LRN-004`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To create a structured, actionable, and personalized learning plan for a complex topic, tailored to a user's specific goals, existing knowledge, and preferred learning style.
*   **When to Use:** When you are starting to learn a new, complex subject and want a roadmap that goes beyond just "reading a book." This prompt helps create a curriculum for self-study.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Topic]`: The new subject you want to learn (e.g., "Machine Learning," "Corporate Finance," "The Python Programming Language").
    *   `[Current_Knowledge_Level]`: Your current level of understanding (e.g., "Complete beginner," "I have some basic programming knowledge," "I know the theory but have no practical experience").
    *   `[Learning_Goal]`: What you want to be able to *do* with the knowledge (e.g., "Build a predictive model," "Analyze a company's financial statements," "Create a web application").
    *   `[Preferred_Learning_Style]`: How you learn best (e.g., "Reading books and articles," "Watching video tutorials," "Hands-on projects," "A mix of theory and practice").
    *   `[Time_Commitment]`: How much time you can dedicate per week (e.g., "5 hours per week for 3 months").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `LearningCoachAgent` or `EducationAgent`.
    *   **Onboarding Tool:** This is an excellent tool for onboarding new team members or for existing members who want to upskill in a new area.

---

### **Example Usage**

```
[Topic]: "Advanced Financial Modeling"
[Current_Knowledge_Level]: "I have a good understanding of basic accounting and Excel, but I've never built a full three-statement financial model."
[Learning_Goal]: "To be able to build a detailed, robust three-statement financial model for a public company from scratch."
[Preferred_Learning_Style]: "I learn best by doing, so I'd prefer a project-based approach with some recommended readings."
[Time_Commitment]: "10 hours per week for the next 8 weeks."
```

---

## **Full Prompt Template**

```markdown
# ROLE: Expert Curriculum Designer & Learning Coach

# CONTEXT:
You are an expert in pedagogy and curriculum design. Your task is to create a personalized, actionable, and structured learning plan for a user who wants to master a new, complex topic. The plan must be tailored to their specific needs and goals.

# LEARNER PROFILE:
*   **Topic to Learn:** `[Topic]`
*   **Current Knowledge Level:** `[Current_Knowledge_Level]`
*   **Ultimate Learning Goal:** `[Learning_Goal]`
*   **Preferred Learning Style:** `[Preferred_Learning_Style]`
*   **Time Commitment:** `[Time_Commitment]`

# TASK:
Generate a comprehensive, week-by-week learning plan based on the learner's profile.

---
## **Personalized Learning Plan: Mastering [Topic]**

### **1. Foundational Concepts (Weeks 1-2)**
*(What are the absolute, must-know fundamentals? This section should front-load the most critical theoretical knowledge.)*
*   **Key Topics:**
    *   Topic 1.1
    *   Topic 1.2
*   **Learning Resources:**
    *   **Reading:** [Suggest specific books, articles, or documentation]
    *   **Videos:** [Suggest specific online courses or video series]
*   **Key Outcome for this Phase:** "By the end of this phase, you should be able to explain [core concept] in your own words."

### **2. Practical Application & Core Skills (Weeks 3-5)**
*(This section should focus on hands-on application. How does the user start *doing* the thing they want to learn?)*
*   **Key Skills to Develop:**
    *   Skill 2.1
    *   Skill 2.2
*   **Project:**
    *   **Project Goal:** [Define a small, achievable project. e.g., "Build a simple discounted cash flow (DCF) model." ]
    *   **Steps:** [Break the project down into manageable steps]
*   **Key Outcome for this Phase:** "By the end of this phase, you will have built your first [project artifact]."

### **3. Advanced Concepts & Integration (Weeks 6-7)**
*(Introduce more complex topics and connect them back to the fundamentals.)*
*   **Key Topics:**
    *   Topic 3.1
    *   Topic 3.2
*   **Learning Resources:**
    *   **Reading:** [Suggest more advanced materials]
*   **Project Enhancement:** "Now, enhance your project by integrating [advanced concept]."

### **4. Capstone Project & Solidification (Week 8)**
*(A final project that ties everything together and proves mastery of the learning goal.)*
*   **Capstone Project Goal:** "[Define a project that directly aligns with the user's ultimate learning goal.]"
*   **Next Steps:** "Once you have completed this learning plan, your next logical step would be to [suggest a more advanced topic or application]."

### **Measurement of Success:**
*   You will know you have successfully mastered this topic when you can confidently [re-state the learning goal].

---
```
