import { PromptObject } from '../types/promptAlpha';
import { calculatePromptAlpha } from './alphaCalculator';

const TOPICS = [
  "Quantum Physics", "Market Analysis", "Code Refactoring", "Creative Writing",
  "Legal Contract", "Medical Diagnosis", "Cybersecurity Audit", "DeFi Protocol",
  "Neural Architecture", "Supply Chain Optimization", "Climate Modeling"
];

const ROLES = [
  "Senior Architect", "Quant Analyst", "Legal Counsel", "Security Researcher",
  "Prompt Engineer", "AI Ethicist", "Rust Developer", "Data Scientist"
];

const TEMPLATES = [
  "Act as a {{ROLE}}. Your task is to {{TASK}}. constraints: {{CONSTRAINTS}}.",
  "System: You are an expert in {{TOPIC}}.\nUser: Explain {{TASK}} step-by-step.",
  "Generate a {{TOPIC}} report in JSON format. Fields: {{FIELDS}}.",
  "Analyze the following text for {{TOPIC}} bias:\n[Context]\n{{TASK}}",
  "Write a python script to {{TASK}}. Ensure high performance and type safety.",
  "Construct a knowledge graph for {{TOPIC}} focusing on key entities.",
  "Simulate a debate between a {{ROLE}} and a regulator regarding {{TOPIC}}."
];

function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateRandomString(length: number): string {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export const TASKS = [
  "Optimizing weights for layer 4",
  "Backpropagating error gradients",
  "Fetching live market data",
  "Sanitizing user input",
  "Computing Alpha coefficients",
  "Rebalancing portfolio",
  "Validating JSON schema",
  "Encrypting payload",
  "Compressing context window",
  "Pruning low-confidence nodes",
  "Synthesizing narrative",
  "Running Monte Carlo simulation"
];

export function generateAgentTask(): string {
  return randomChoice(TASKS);
}

export function generateSyntheticPrompt(): PromptObject {
  const topic = randomChoice(TOPICS);
  const role = randomChoice(ROLES);
  const baseTemplate = randomChoice(TEMPLATES);

  // Fill template partially to create variety
  let content = baseTemplate
    .replace('{{ROLE}}', role)
    .replace('{{TOPIC}}', topic)
    .replace('{{TASK}}', `analyze ${generateRandomString(10)}`)
    .replace('{{CONSTRAINTS}}', 'no hallucinations, return JSON')
    .replace('{{FIELDS}}', 'id, score, reason, confidence');

  // Add some random noise/length to vary the score
  if (Math.random() > 0.5) {
    content += `\n\nContext:\n${generateRandomString(Math.floor(Math.random() * 500))}`;
  }

  // Sometimes add structural keywords explicitly to boost score
  if (Math.random() > 0.7) {
    content = "System: " + content + "\nStep-by-step reasoning:\n1. ";
  }

  const metrics = calculatePromptAlpha(content);

  return {
    id: `synth-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
    title: `SYNTH: ${topic} - ${role}`,
    content: content,
    source: 'Simulation',
    timestamp: Date.now(),
    author: 'Ghost_in_Shell',
    alphaScore: metrics.score,
    metrics: {
      length: metrics.lengthScore,
      variableDensity: metrics.variableScore,
      structuralKeywords: metrics.structureScore,
    },
    tags: ['simulation', topic.toLowerCase().replace(' ', '-'), 'synthetic'],
    isFavorite: false,
  };
}
