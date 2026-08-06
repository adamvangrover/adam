// SYS_PROMPT_REGIME_SHIFT_EVAL
{
  "command": "EVALUATE_REGIME_SHIFT",
  "trigger": "VOLATILITY_BREAKOUT",
  "instructions": "The system has detected a multi-sigma move in the VIX concurrently with a 20bps intra-day spike in the US 10-Year Treasury Yield.
  1. INGEST the latest intraday pricing for the broader market index and the HY credit spread.
  2. CALCULATE the probability of a Markov-Chain state transition from 'Hyper-Expansion' to 'Stagflationary Shock'.
  3. If transition probability > 0.65, INITIATE the 'State 2: Synthetic Fallback' protocol to recalculate structural default risks (Merton PD) based on the new regime's drift parameters."
}