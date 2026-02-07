export interface AlphaMetrics {
  score: number;
  lengthScore: number;
  variableScore: number;
  structureScore: number;
  raw: {
    length: number;
    variableCount: number;
    keywordCount: number;
  };
}

const STRUCTURAL_KEYWORDS = [
  /system:/i,
  /user:/i,
  /assistant:/i,
  /step-by-step/i,
  /think step/i,
  /return format/i,
  /json/i,
  /constraint/i,
  /context:/i,
  /example:/i
];

export function calculatePromptAlpha(promptText: string): AlphaMetrics {
  if (!promptText) {
    return {
      score: 0,
      lengthScore: 0,
      variableScore: 0,
      structureScore: 0,
      raw: { length: 0, variableCount: 0, keywordCount: 0 }
    };
  }

  // 1. Length Score (Logarithmic scale to reward length but with diminishing returns)
  // Cap at 2000 chars for max score contribution
  const length = promptText.length;
  const lengthScore = Math.min(100, (Math.log(length + 1) / Math.log(2000)) * 40); // Weight: ~30-40% of total

  // 2. Variable Density (Counting {{placeholders}})
  // Regex for {{...}} or { ... }
  const variableMatches = promptText.match(/\{\{.*?\}\}|\{.*?\}/g) || [];
  const variableCount = variableMatches.length;
  // 5 points per variable, capped at 30
  const variableScore = Math.min(30, variableCount * 5);

  // 3. Structural Keywords
  let keywordCount = 0;
  STRUCTURAL_KEYWORDS.forEach(regex => {
    if (regex.test(promptText)) {
      keywordCount++;
    }
  });
  // 5 points per distinct keyword, capped at 40
  const structureScore = Math.min(40, keywordCount * 5);

  // Total Score Calculation
  let totalScore = lengthScore + variableScore + structureScore;

  // Normalize to 0-100 (it can go slightly over with the logic above, so cap it)
  totalScore = Math.min(100, Math.max(0, totalScore));

  return {
    score: Math.round(totalScore),
    lengthScore: Math.round(lengthScore),
    variableScore: Math.round(variableScore),
    structureScore: Math.round(structureScore),
    raw: {
      length,
      variableCount,
      keywordCount
    }
  };
}
