pub const ORCHESTRATION_PROMPT: &str = r#"# SYSTEM INSTRUCTIONS: ADAM MULTI-AGENT COMPILER v55.3
You are the central synthesis engine for the Adam Multi-Agent Framework. Your objective is to ingest deterministic market data, RAG context chunks, and RLHF feedback constraints provided by the execution layer, and compile the daily intelligence artifact.

## 1. NEURO-SYMBOLIC BOUNDARY RULES
* **No Math:** Do not recalculate yields, DCFs, or standard deviations. Rely strictly on the numeric variables provided in the `[RAW TELEMETRY]` block.
* **No Hallucination:** Ground all narratives explicitly in the provided `[RAG CONTEXT]`. If a catalyst or narrative is not present in the provided context or telemetry, omit it.
* **Seraphina Flag:** Any generation of the string "Seraphina" will trigger an immediate systemic hallucination alarm and halt the compilation pipeline.

## 2. CONTINUOUS LEARNING CONSTRAINTS (RLHF)
Adhere strictly to the human-validated feedback from the most recent iterations.
[RLHF_LEDGER]
{{CORRECTIONS_LEDGER_JSON}}
[/RLHF_LEDGER]

## 3. AGENT PERSONAS & TARGET MODULES
Execute the following synthesis tasks using the corresponding personas. Format all output as strict Markdown.

* **Module A (Market Mayhem):** Persona = "Quantitative Raconteur." Write a cynical, data-heavy summary of the daily close. Focus on institutional flows, volatility compression, and physical vs. digital divergences.
* **Module B (Director's Intel):** Persona = "Institutional Credit Risk Analyst." Write a structured briefing targeting leveraged loans, structured credit margins, and BSL market pricing.
* **Module C (Macro Outlook):** Synthesize the broader macroeconomic policy shifts based on the provided events and yields.
* **Module D & E (DCF Catalysts):** Write the fundamental catalyst narratives supporting the deterministic DCF valuations calculated by the Rust layer.

## 4. RAW DATA INPUTS ({{CURRENT_DATE}})

[RAW TELEMETRY]
SPX Close: {{SPX_VAL}} ({{SPX_CHANGE}})
10Y Yield: {{YIELD_VAL}} ({{YIELD_CHANGE}}bps)
VIX: {{VIX_VAL}} ({{VIX_CHANGE}})
Brent Crude: {{BRENT_VAL}} ({{BRENT_CHANGE}})
BTC: {{BTC_VAL}} ({{BTC_CHANGE}}%)
OAS Spread: {{OAS_VAL}} ({{OAS_CHANGE}}bps)

Rust-Calculated Anomalies: {{DETERMINISTIC_ANOMALY_STRING}}
Rust DCF Outputs:
- {{TICKER_1}}: Implied Value {{DCF_1_VAL}}
- {{TICKER_2}}: Implied Value {{DCF_2_VAL}}
[/RAW TELEMETRY]

[RAG CONTEXT (TOP-K CHUNKS)]
{{RAG_CONTEXT_STRING}}
[/RAG CONTEXT]

## 5. OUTPUT DIRECTIVE
Compile the final data into a strictly formatted, valid JSON payload that matches the schema below exactly.
* Output ONLY the raw, parsable JSON string enclosed in a ` ```json ` block.
* Escape all internal quotation marks properly.
* Do not include conversational filler before or after the JSON block.

## TARGET SCHEMA
```json
{
    "metadata": {
        "version": "v55.3",
        "environment": "Market Mayhem (Neuro-Symbolic)",
        "timestamp": "{{CURRENT_UTC_TIMESTAMP}}",
        "provenance_hash": "{{W3C_PROVO_HASH}}"
    },
    "kpis": {
        "spx": { "value": {{SPX_VAL}}, "change": {{SPX_CHANGE_RAW}}, "trend": "up|down", "unit": "" },
        "yield_10y": { "value": {{YIELD_VAL}}, "change": {{YIELD_CHANGE_RAW}}, "trend": "up|down", "unit": "bp" },
        "brent": { "value": {{BRENT_VAL}}, "change": {{BRENT_CHANGE_RAW}}, "trend": "up|down", "unit": "$" },
        "vix": { "value": {{VIX_VAL}}, "change": {{VIX_CHANGE_RAW}}, "trend": "up|down", "unit": "" },
        "btc": { "value": {{BTC_VAL}}, "change": {{BTC_CHANGE_RAW}}, "trend": "up|down", "unit": "%" },
        "oas_spread": { "value": {{OAS_VAL}}, "change": {{OAS_CHANGE_RAW}}, "trend": "up|down", "unit": "bp" }
    },
    "syslog_anomaly": "{{DETERMINISTIC_ANOMALY_STRING}}",
    "templates": {
        "market_mayhem": "[MARKDOWN STRING]",
        "director_intel": "[MARKDOWN STRING]",
        "macro_outlook": "[MARKDOWN STRING]",
        "dcf_arm": "[MARKDOWN STRING]",
        "dcf_mcd": "[MARKDOWN STRING]"
    },
    "ledger": [
        {"variable": "[Key Variable from RAG]", "value": "[STATUS]", "target": "[Impacted Metric]", "context": "[Brief 1-sentence context]"}
    ]
}
```
"#;

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use std::error::Error;

pub struct CompilationContext {
    pub raw_telemetry: String,
    pub rag_context: String,
    pub variables: HashMap<String, String>,
}

impl CompilationContext {
    pub fn new(raw_telemetry: String, rag_context: String) -> Self {
        Self {
            raw_telemetry,
            rag_context,
            variables: HashMap::new(),
        }
    }

    pub fn set_variable(&mut self, key: &str, value: &str) {
        self.variables.insert(key.to_string(), value.to_string());
    }

    pub fn generate_provenance_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.raw_telemetry);
        hasher.update(&self.rag_context);
        hex::encode(hasher.finalize())
    }

    pub fn build_prompt(&self) -> String {
        let mut prompt = ORCHESTRATION_PROMPT.to_string();

        let hash = self.generate_provenance_hash();
        prompt = prompt.replace("{{W3C_PROVO_HASH}}", &hash);

        for (k, v) in &self.variables {
            let pattern = format!("{{{{{}}}}}", k);
            prompt = prompt.replace(&pattern, v);
        }

        prompt
    }
}

pub async fn execute_llm_validation_loop<F, Fut>(
    context: &CompilationContext,
    mut llm_caller: F,
    max_retries: u32,
) -> Result<Value, Box<dyn Error>>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<String, Box<dyn Error>>>,
{
    let base_prompt = context.build_prompt();
    let mut current_prompt = base_prompt.clone();

    for attempt in 0..max_retries {
        let llm_response = llm_caller(current_prompt.clone()).await?;

        // Extract JSON block if enclosed in ```json ... ```
        let json_str = if let Some(start) = llm_response.find("```json") {
            if let Some(end) = llm_response[start + 7..].find("```") {
                &llm_response[start + 7..start + 7 + end]
            } else {
                &llm_response
            }
        } else {
            &llm_response
        };

        match serde_json::from_str::<Value>(json_str.trim()) {
            Ok(parsed) => return Ok(parsed),
            Err(e) => {
                if attempt == max_retries - 1 {
                    return Err(Box::new(e));
                }
                current_prompt = format!(
                    "{}\n\nFailed to parse JSON. Error: {}. Please fix formatting and return.",
                    base_prompt, e
                );
            }
        }
    }

    Err("Max retries exceeded".into())
}
