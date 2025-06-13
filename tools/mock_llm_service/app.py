from flask import Flask, request, jsonify
import os
import json
import re
import logging
import random

app = Flask(__name__)

# --- Global Variables & Constants ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROBABILITY_MAP_FILE = os.path.join(DATA_DIR, 'PROBABILITY_MAP_CORE.json')
PROBABILITY_MAP_CORE = []

# --- Orchestrator Thresholds ---
CORE_MATCH_CONFIDENCE_THRESHOLD = 0.65
# Dark Forest Triggers
DF_VEGA_THRESHOLD = 5
DF_DELTA_LOW_THRESHOLD = 0.15 # If Vega is high, Delta must be very low
DF_ULTRA_LOW_DELTA_THRESHOLD = 0.1 # If Delta is ultra low, trigger DF regardless of Vega (almost no match)
# Random Walk Triggers (evaluated if not Dark Forest)
RW_VEGA_THRESHOLD = 3
RW_DELTA_LOW_THRESHOLD = 0.35
RW_GAMMA_THRESHOLD = 3
RW_DELTA_MODERATE_THRESHOLD = 0.4 # If Gamma is high, Delta must be somewhat low
# Scatterplot Triggers (evaluated if not Dark Forest or Random Walk)
SCATTER_DELTA_LOW_THRESHOLD = 0.5
SCATTER_GAMMA_THRESHOLD = 2


STOP_WORDS = {
    "the", "a", "is", "an", "and", "of", "to", "in", "it", "that", "this", "for", "not", "as",
    "with", "was", "on", "at", "by", "be", "or", "but", "if", "from", "has", "are", "were",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "why", "when", "where", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "summary", "data", "model", "simulation", "simulated", "provide", "based",
    "guidelines", "system", "key", "observations", "conclusions", "brief", "synthesized",
    "adhering", "potential", "implications", "actionable", "insights", "appropriate", "employ",
    "chain-of-thought", "reasoning", "beneficial", "clarity", "context", "wsm", "llm", "plugin",
    "analysis", "analyze", "following", "comprehensive", "report", "request", "objective",
    "id", "timestamp", "source", "module", "run", "generated", "world", "service", "mock",
    "endpoint", "using", "reference", "probability", "map", "loaded", "json", "rule", "rules"
}

THEMATIC_SCATTER_SNIPPETS = [
    {"theme": "general_market_uncertainty", "keywords": ["market", "economy", "uncertain", "volatile", "sentiment", "outlook"],
     "snippet": "The current market environment presents a complex tapestry of signals, suggesting that underlying volatility may persist. Investors should prioritize robust risk management frameworks."},
    {"theme": "long_term_perspective", "keywords": ["long-term", "future", "strategic", "horizon", "trend"],
     "snippet": "Adopting a long-term perspective is often beneficial, allowing strategic positions to mature beyond short-term market noise. Consider fundamental drivers when assessing future potential."},
    {"theme": "innovation_opportunity", "keywords": ["innovation", "emerging", "tech", "opportunity", "growth", "disruptive"],
     "snippet": "Periods of change often highlight opportunities in innovation. Emerging technologies or business models, while carrying risk, may offer significant growth potential for well-researched ventures."},
    {"theme": "data_driven_caution", "keywords": ["data", "analysis", "caution", "review", "model", "validate", "verify"],
     "snippet": "A thorough, data-driven review is essential before committing to significant strategic shifts. Ensure all available information is considered to align with our principle of informed decision-making."},
    {"theme": "geopolitical_factors", "keywords": ["geopolitical", "global", "policy", "international", "regulatory", "risk"],
     "snippet": "Global geopolitical factors continue to play a crucial role in market dynamics. Monitoring policy shifts and international relations is key to anticipating potential impacts."},
    {"theme": "need_for_more_granularity", "keywords": ["specific", "detail", "further", "breakdown", "additional_info", "clarify"],
     "snippet": "To provide a more targeted insight, a more granular breakdown of the query or additional specific data points would be beneficial for a comprehensive analysis."}
]

RANDOM_WALK_ANALYSIS_FLOW = {
    "Identify_Factors": [
        "Key elements to consider from the simulation appear to be {factor1} and its interplay with {factor2}.",
        "The data suggests {factor1} and {factor2} are significant in this context, potentially leading to {hypothetical_outcome}.",
        "Upon initial review, {factor1} stands out, especially when correlated with trends in {factor2}."
    ],
    "Assess_Initial_Impact": [
        "Initially, {factor1} seems to exert a {positive_negative_neutral} influence on {hypothetical_outcome}.",
        "One might observe that {factor2} contributes to a {trend_description} for {hypothetical_outcome}, particularly regarding {context_element}.",
        "The impact of {factor1} on {hypothetical_outcome} appears to be {positive_negative_neutral}, though {factor2} introduces complexities."
    ],
    "Consider_Context_Nuances": [
        "However, factoring in {context_element}, this interpretation may require adjustment or further validation.",
        "Nuances such as {context_element} (related to {factor2}) add layers to this initial assessment of {hypothetical_outcome}.",
        "It's crucial to consider {context_element}, which could moderate the effects of {factor1} on {hypothetical_outcome}."
    ],
    "Formulate_Cautious_Hypothesis": [
        "A tentative hypothesis is that {factor1} will lead to {hypothetical_outcome}, assuming {context_element} remains influential and {factor2} stabilizes.",
        "This could suggest a potential trajectory towards {hypothetical_outcome}, contingent on further data regarding {factor2} and its impact from {context_element}.",
        "Therefore, one might hypothesize that {hypothetical_outcome} is plausible if {factor1} and {factor2} continue their current course, tempered by {context_element}."
    ],
    "Concluding_Remark_Uncertainty": [
        "This exploratory reasoning highlights potential pathways but also underscores the inherent uncertainties and the need for adaptive analysis.",
        "Ultimately, the dynamic interplay of these factors ({factor1}, {factor2}, {context_element}) warrants ongoing monitoring and a flexible strategic outlook regarding {hypothetical_outcome}.",
        "Further detailed modeling would be necessary to confirm these initial thoughts on {hypothetical_outcome}, especially concerning the stability of {factor1} and {factor2}."
    ]
}

DARK_FOREST_RESPONSE_TEMPLATES = [
    "The current query or simulated data presents a high degree of novelty or ambiguity. Adhering to our principle of Transparency & Explainability, providing a specific analytical conclusion at this stage would be speculative. Further clarification of the query or more specific data inputs are recommended.",
    "Consistent with Adam's focus on Data-Driven Analysis, the provided information is insufficient to form a robust conclusion or a confident simulation. The system requires more defined parameters or context to generate a meaningful insight.",
    "The scenario described appears to fall outside the well-defined patterns the simulation is currently optimized for. To ensure Actionable Intelligence, we advise refining the query or exploring alternative WSM configurations that provide clearer data signals.",
    "Acknowledging the limits of the current simulation: the input lacks sufficient grounding in established data patterns for a reliable analysis. A more structured query or additional contextual data would be beneficial.",
    "This query touches upon areas of significant uncertainty. While exploration is valuable, Adam's core function is to provide insights based on reasonable data foundations. Please consider rephrasing or providing more specific inputs for {key_topic}."
]

# --- Utility Functions ---
def load_probability_map():
    global PROBABILITY_MAP_CORE
    try:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR); logging.info(f"Created data directory: {DATA_DIR}")
        if os.path.exists(PROBABILITY_MAP_FILE):
            with open(PROBABILITY_MAP_FILE, 'r', encoding='utf-8') as f: PROBABILITY_MAP_CORE = json.load(f)
            logging.info(f"Successfully loaded {len(PROBABILITY_MAP_CORE)} rules from {PROBABILITY_MAP_FILE}")
        else:
            logging.warning(f"{PROBABILITY_MAP_FILE} not found. Using minimal fallback rule.")
            PROBABILITY_MAP_CORE = [{"rule_name": "DefaultMissingMapFallback", "description": "Fallback rule due to missing JSON map.", "theme": "System Alert", "keywords": [], "entity_placeholders": [], "negative_keywords": [], "adam_principles_invoked": ["Transparency & Explainability"], "conclusion_template": "Core probability map JSON file not found. Service is using a minimal fallback response.", "confidence": 0.1}]
    except Exception as e:
        logging.error(f"Error loading or creating probability map: {e}", exc_info=True)
        PROBABILITY_MAP_CORE = [{"rule_name": "ErrorLoadingMapFallback", "conclusion_template": "Error loading probability map. Service is using an error fallback response. Details: " + str(e), "confidence": 0.05, "keywords": [], "entity_placeholders": [], "adam_principles_invoked": ["Transparency & Explainability"]}]

def substitute_placeholders(template_string: str, prompt_text_lower: str, entity_placeholders: list, custom_subs: dict = None) -> str:
    substituted_string = template_string
    sub_map = custom_subs if custom_subs else {}
    if not entity_placeholders and not custom_subs: return substituted_string
    for ph in entity_placeholders:
        if ph not in sub_map:
            replacement_value = f"[{ph}_data_unavailable]"
            if ph == "stock_symbol":
                symbols_found = re.findall(r'\b(symb1|symb2|aapl|msft|goog|prompt_aapl|prompt_msft|prompt_goog|prompt_stk1|prompt_stk2)\b', prompt_text_lower)
                if not symbols_found: symbols_found = re.findall(r'\b([A-Z]{2,5})\b', prompt_text_lower.upper())
                replacement_value = symbols_found[0].upper() if symbols_found else "[unidentified stock]"
            elif ph == "gdp_growth_sim_value":
                match = re.search(r'gdp growth[^%.,]*?([\d\.]+%?)', prompt_text_lower)
                if not match: match = re.search(r'gdp_growth_sim[^0-9]*?([\d\.]+)', prompt_text_lower)
                replacement_value = match.group(1) if match else "[N/A_GDP]"
            elif ph == "inflation_sim_value":
                match = re.search(r'inflation[^%.,]*?([\d\.]+%?)', prompt_text_lower)
                if not match: match = re.search(r'inflation_sim[^0-9]*?([\d\.]+)', prompt_text_lower)
                replacement_value = match.group(1) if match else "[N/A_Inflation]"
            sub_map[ph] = replacement_value
    for key, value in sub_map.items():
        placeholder_tag = f"{{{key}}}"
        substituted_string = substituted_string.replace(placeholder_tag, str(value))
        logging.debug(f"Substituted '{placeholder_tag}' with '{value}'")
    return substituted_string

def calculate_greek_sensitivities(prompt_text_lower: str, probability_map_core: list) -> dict:
    words_in_prompt = [word for word in re.split(r'\W+', prompt_text_lower) if word and word not in STOP_WORDS]
    prompt_word_freq = {word: words_in_prompt.count(word) for word in set(words_in_prompt)}
    uncertainty_keywords_list = ["uncertain", "volatility", "volatile", "maybe", "potential", "emerging", "risk", "speculative", "if", "could", "would", "unclear", "ambiguous", "fluctuating", "caution"]
    vega_score = sum(1 for ukw in uncertainty_keywords_list if ukw in prompt_word_freq)
    logging.debug(f"Greek Module: Vega score: {vega_score} based on uncertainty keywords found: {[ukw for ukw in uncertainty_keywords_list if ukw in prompt_word_freq]}.")
    closest_core_rules, highest_delta_score = [], 0.0
    for rule in probability_map_core:
        rule_keywords, rule_negative_keywords = rule.get("keywords", []), rule.get("negative_keywords", [])
        if not rule_keywords: continue
        num_keywords_found = sum(1 for kw in rule_keywords if kw.lower() in prompt_word_freq)
        penalty = 1.0 if any(neg_kw.lower() in prompt_word_freq for neg_kw in rule_negative_keywords) else 0.0
        if penalty > 0: logging.debug(f"Greek Module: Rule '{rule.get('rule_name')}' penalized due to negative keyword.")
        match_score = (num_keywords_found / len(rule_keywords)) * (1 - penalty) if len(rule_keywords) > 0 else 0
        if match_score > highest_delta_score: highest_delta_score = match_score
        if match_score > 0.01: closest_core_rules.append({"name": rule.get("rule_name", "UnnamedRule"), "delta_score": round(match_score, 3), "confidence_of_rule": rule.get("confidence", 0.0)})
    closest_core_rules.sort(key=lambda r: r["delta_score"], reverse=True)
    logging.debug(f"Greek Module: Closest core rules candidates: {closest_core_rules[:5]}")
    gamma_score = sum(1 for r_info in closest_core_rules if 0.3 <= r_info["delta_score"] <= 0.7)
    logging.debug(f"Greek Module: Gamma score: {gamma_score} based on rules with moderate delta scores.")
    return {"closest_core_rules_analysis": closest_core_rules[:5], "gamma_like_score": gamma_score, "vega_like_score": vega_score, "overall_confidence_in_core_match": round(highest_delta_score, 3), "prompt_word_freq": prompt_word_freq}

def generate_core_adjacent_response(prompt_text_lower: str, core_rule_obj: dict, prompt_words_freq: dict, all_rules: list) -> str:
    varied_conclusion_template = core_rule_obj.get("conclusion_template", "No conclusion template found for core rule.")
    core_keywords = set(kw.lower() for kw in core_rule_obj.get("keywords", []))
    variant_keywords_found = [word for word, freq in prompt_words_freq.items() if word not in core_keywords and not word.isdigit()]
    logging.debug(f"Core-Adjacent: Variant keywords considered: {variant_keywords_found[:10]}")
    variation_applied = False
    if any(kw in variant_keywords_found for kw in ["cautious", "moderately", "slight", "however", "but", "caveat", "warning", "nonetheless"]):
        varied_conclusion_template += " However, some contextual elements suggest a degree of moderation or specific caveats should be considered."
        variation_applied = True; logging.info("Core-Adjacent: Applied 'caution/moderation' variation.")
    elif any(kw in variant_keywords_found for kw in ["strong", "very", "clear_trend", "confirmed", "high_impact", "significant"]):
        varied_conclusion_template += " The contextual data appears to further reinforce this core observation strongly."
        variation_applied = True; logging.info("Core-Adjacent: Applied 'strong reinforcement' variation.")
    if "Single Stock Analysis" in core_rule_obj.get("theme", "") and not variation_applied:
        for vk in variant_keywords_found:
            if vk in ["gdp_low", "weak_economy", "recession_risk", "inflation_high", "price_pressure"]:
                varied_conclusion_template += f" It's also pertinent to note the broader economic context, such as signals related to '{vk}', which may influence this outlook."
                logging.info(f"Core-Adjacent: Applied 'macroeconomic context ({vk})' variation to stock analysis.")
                break
    if not variation_applied: logging.info("Core-Adjacent: No specific variation heuristic applied.")
    return substitute_placeholders(varied_conclusion_template, prompt_text_lower, core_rule_obj.get("entity_placeholders", []))

def generate_scatterplot_response(prompt_words_freq: dict, greek_analysis_results: dict) -> str:
    selected_snippets_text, candidate_snippets = [], []
    for entry in THEMATIC_SCATTER_SNIPPETS:
        matches = sum(1 for kw in entry["keywords"] if kw in prompt_words_freq)
        if matches > 0: candidate_snippets.append({"snippet": entry["snippet"], "matches": matches, "theme": entry["theme"]})
    logging.debug(f"Scatterplot: Candidate snippets: {[(cs['theme'], cs['matches']) for cs in candidate_snippets]}")
    candidate_snippets.sort(key=lambda x: x["matches"], reverse=True)
    num_to_select = 0
    if len(prompt_words_freq) > 15 : num_to_select = 2
    elif len(prompt_words_freq) > 5 : num_to_select = 1
    if greek_analysis_results.get("gamma_like_score", 0) >= 2 and num_to_select < 2 : num_to_select = min(2, len(candidate_snippets))
    if greek_analysis_results.get("vega_like_score", 0) >= 3 and num_to_select < 2: num_to_select = min(2, len(candidate_snippets))
    num_to_select = min(len(candidate_snippets), num_to_select if num_to_select > 0 else 1)
    if candidate_snippets: selected_snippets_text = [cs["snippet"] for cs in candidate_snippets[:num_to_select]]
    if not selected_snippets_text:
        fallback_themes = ["data_driven_caution", "need_for_more_granularity"]
        for theme_name in fallback_themes:
            snippet_obj = next((s for s in THEMATIC_SCATTER_SNIPPETS if s["theme"] == theme_name), None)
            if snippet_obj: selected_snippets_text.append(snippet_obj["snippet"]);
            if len(selected_snippets_text) >=1 : break
        if not selected_snippets_text: selected_snippets_text.append("A general review suggests multiple factors. More specific data would enable clearer analysis.")
    final_response = selected_snippets_text[0] if len(selected_snippets_text) == 1 else "Considering various perspectives based on the input:\n- " + "\n- ".join(selected_snippets_text) if len(selected_snippets_text) > 1 else "The analysis points to a complex situation. Detailed investigation is recommended."
    final_response += "\n\n(This exploratory analysis offers general perspectives based on the query's nature and should not be considered definitive financial advice.)"
    logging.info(f"Scatterplot response generated with {len(selected_snippets_text)} snippet(s).")
    return final_response

def generate_random_walk_response(prompt_words_freq: dict, greek_analysis_results: dict) -> str:
    walk_steps_text = []
    walk_order = ["Identify_Factors", "Assess_Initial_Impact", "Consider_Context_Nuances", "Formulate_Cautious_Hypothesis", "Concluding_Remark_Uncertainty"]
    num_steps_in_walk = random.randint(min(3, len(walk_order)), min(len(walk_order), len(walk_order)))
    if num_steps_in_walk == len(walk_order): current_walk_path = walk_order
    elif num_steps_in_walk == len(walk_order) -1 and "Concluding_Remark_Uncertainty" not in random.sample(walk_order[:-1], num_steps_in_walk):
        path_sample = random.sample(walk_order[:-1], num_steps_in_walk -1); path_sample.append("Concluding_Remark_Uncertainty"); current_walk_path = path_sample
    else:
        current_walk_path = random.sample(walk_order, num_steps_in_walk)
        current_walk_path.sort(key=lambda x: walk_order.index(x))
    logging.debug(f"RandomWalk: Path: {current_walk_path}")
    sorted_prompt_words = sorted(prompt_words_freq.items(), key=lambda item: item[1], reverse=True)
    significant_words = [word for word, freq in sorted_prompt_words if len(word) > 2 and not word.isdigit()]
    if not significant_words: significant_words = [word for word, freq in sorted_prompt_words if not word.isdigit()]
    if not significant_words: significant_words = [word for word, freq in sorted_prompt_words]
    factor1 = significant_words[0] if len(significant_words) > 0 else "key_simulated_variables"
    factor2 = significant_words[1] if len(significant_words) > 1 else "observed_trends"
    context_element = significant_words[2] if len(significant_words) > 2 else "broader_market_conditions"
    hypothetical_outcome = random.choice(["a revised market outlook", "a shift in strategic focus", "a re-evaluation of current assumptions"])
    positive_negative_neutral = random.choice(["a positive", "a negative", "a neutral", "a mixed", "a complex"])
    trend_description = random.choice(["a noticeable shift", "a developing pattern", "a subtle change", "an accelerating trend"])
    logging.debug(f"RandomWalk: Factors for placeholders: f1='{factor1}', f2='{factor2}', ctx='{context_element}'")
    substitutions = {"factor1": factor1, "factor2": factor2, "context_element": context_element, "hypothetical_outcome": hypothetical_outcome, "positive_negative_neutral": positive_negative_neutral, "trend_description": trend_description}
    for step_name in current_walk_path:
        phrase_template = random.choice(RANDOM_WALK_ANALYSIS_FLOW.get(step_name, ["A generic observation was made."]))
        populated_phrase = substitute_placeholders(phrase_template, "", [], custom_subs=substitutions)
        walk_steps_text.append(populated_phrase)
    final_response = " ".join(walk_steps_text)
    final_response += "\n\n(This response outlines a structured exploratory thought process based on the input's nature and should be considered illustrative.)"
    logging.info(f"RandomWalk response generated with {len(walk_steps_text)} steps. Path: {current_walk_path}")
    return final_response

def generate_dark_forest_response(prompt_words_freq: dict, greek_analysis_results: dict) -> str:
    selected_template = random.choice(DARK_FOREST_RESPONSE_TEMPLATES)
    final_response = selected_template
    key_topic_placeholder = "{key_topic}"
    if key_topic_placeholder in selected_template:
        sorted_prompt_words = sorted(prompt_words_freq.items(), key=lambda item: item[1], reverse=True)
        contextual_term = "the main subject of your query"
        for word, freq in sorted_prompt_words[:5]:
            if len(word) > 3 and word not in ["data", "analysis", "based", "provide", "using", "current", "simulation", "simulated", "llm", "prompt"]:
                contextual_term = f"'{word}'"; break
        final_response = selected_template.replace(key_topic_placeholder, contextual_term)
    logging.info(f"DarkForest response generated. Template (potentially contextualized): {final_response[:120]}...")
    return final_response

# --- Flask Endpoint ---
@app.route('/mock_complete', methods=['POST'])
def mock_complete():
    try:
        data = request.get_json()
        prompt_text = data.get('prompt')
        if prompt_text is None: return jsonify({"error": "No prompt provided"}), 400

        lower_prompt = prompt_text.lower()
        logging.info(f"Mock LLM received prompt (first 1000 chars): {prompt_text[:1000]}")

        # 1. Calculate Greek Sensitivities
        greek_sensitivities_data = calculate_greek_sensitivities(lower_prompt, PROBABILITY_MAP_CORE)
        logging.info(f"Greek Sensitivities Calculated: {greek_sensitivities_data}")
        prompt_word_freq = greek_sensitivities_data.pop("prompt_word_freq", {})

        response_content = ""
        response_type_for_debug = "Undetermined"
        orchestrator_reasoning = "Initial state."
        selected_rule_object_for_debug = {} # Store the whole selected rule for debug if applicable

        delta_score = greek_sensitivities_data.get("overall_confidence_in_core_match", 0.0)
        gamma_score = greek_sensitivities_data.get("gamma_like_score", 0)
        vega_score = greek_sensitivities_data.get("vega_like_score", 0)

        is_map_functional = PROBABILITY_MAP_CORE and PROBABILITY_MAP_CORE[0].get("rule_name") not in ["DefaultMissingMapFallback", "ErrorLoadingMapFallback"]

        # 2. Dynamic Response Orchestrator
        if delta_score >= CORE_MATCH_CONFIDENCE_THRESHOLD and is_map_functional:
            # 2a. Core Rule Matching (High Confidence Path)
            orchestrator_reasoning = f"Core rule confidence ({delta_score:.2f}) met threshold ({CORE_MATCH_CONFIDENCE_THRESHOLD}). Using Core/Adjacent path."
            logging.info(orchestrator_reasoning)

            # Find the best matching rule (this part is somewhat repeated from greek_sensitivities, could optimize)
            # For now, re-evaluate to get the full selected_rule object
            matched_rules_for_core, _ = [], set() # _ is found_keywords_in_prompt_for_core
            for rule in PROBABILITY_MAP_CORE:
                rule_keywords, rule_negative_keywords = rule.get("keywords", []), rule.get("negative_keywords", [])
                if not rule_keywords or "Fallback" in rule.get("rule_name",""): continue # Skip fallbacks for direct high-confidence match

                if all(kw.lower() in lower_prompt for kw in rule_keywords) and \
                   not any(neg_kw.lower() in lower_prompt for neg_kw in rule_negative_keywords):
                    matched_rules_for_core.append(rule)

            if matched_rules_for_core:
                selected_rule = max(matched_rules_for_core, key=lambda r: r.get("confidence", 0.0) * (1 - (1 if any(neg_kw.lower() in lower_prompt for neg_kw in r.get("negative_keywords",[])) else 0) ) ) # Factor in confidence and ensure no negative keyword match
                # Check if the delta score of this selected_rule is indeed high (using the pre-calculated greek scores for consistency)
                # Find this rule in greek_sensitivities_data.closest_core_rules_analysis
                rule_delta_from_greeks = 0.0
                for r_greek in greek_sensitivities_data.get("closest_core_rules_analysis", []):
                    if r_greek.get("name") == selected_rule.get("rule_name"):
                        rule_delta_from_greeks = r_greek.get("delta_score",0.0)
                        break

                if rule_delta_from_greeks >= CORE_MATCH_CONFIDENCE_THRESHOLD : # Confirm this specific rule's delta
                    selected_rule_object_for_debug = selected_rule
                    logging.info(f"High-Confidence Core Rule Selected: '{selected_rule.get('rule_name')}' with original confidence {selected_rule.get('confidence')} and delta {rule_delta_from_greeks:.2f}")
                    core_adjacent_template = generate_core_adjacent_response(lower_prompt, selected_rule, prompt_word_freq, PROBABILITY_MAP_CORE)
                    response_content = substitute_placeholders(core_adjacent_template, lower_prompt, selected_rule.get("entity_placeholders", []))
                    response_type_for_debug = "CoreRule_With_AdjacentVariation" # Assume variation might occur
                else:
                     # This case should be rare if CORE_MATCH_CONFIDENCE_THRESHOLD is aligned with how highest_delta_score is calculated
                    logging.warning(f"A rule was found but its specific delta score ({rule_delta_from_greeks:.2f}) was below threshold. Re-evaluating for dynamic strategy.")
                    delta_score = rule_delta_from_greeks # Update delta_score to this rule's actual for re-evaluation below
            else: # No rule fully matched keywords without negative conflicts
                 logging.info("High overall_confidence_in_core_match, but no specific rule fully matched without conflicts. Re-evaluating for dynamic strategy.")
                 # delta_score remains the general highest keyword overlap, proceed to dynamic strategies

        # 3. Dynamic Strategy Orchestration (Low Core Confidence or Re-evaluation Path)
        if not response_content: # If Core Rule path didn't set response_content
            # 3a. Dark Forest Conditions
            if not is_map_functional:
                orchestrator_reasoning = "Dark Forest: Probability map not functional."
                use_dark_forest = True
            elif vega_score >= DF_VEGA_THRESHOLD and delta_score < DF_DELTA_LOW_THRESHOLD:
                orchestrator_reasoning = f"Dark Forest: High Vega ({vega_score}) and Very Low Delta ({delta_score:.2f})."
                use_dark_forest = True
            elif delta_score < DF_ULTRA_LOW_DELTA_THRESHOLD:
                orchestrator_reasoning = f"Dark Forest: Ultra Low Delta ({delta_score:.2f})."
                use_dark_forest = True
            else: use_dark_forest = False

            if use_dark_forest:
                logging.info(orchestrator_reasoning)
                response_content = generate_dark_forest_response(prompt_word_freq, greek_sensitivities_data)
                response_type_for_debug = "DarkForestResponse"
            else:
                # 3b. Random Walk Conditions
                if (vega_score >= RW_VEGA_THRESHOLD and delta_score < RW_DELTA_LOW_THRESHOLD) or \
                   (gamma_score >= RW_GAMMA_THRESHOLD and delta_score < RW_DELTA_MODERATE_THRESHOLD):
                    orchestrator_reasoning = f"Random Walk: Vega ({vega_score}) / Gamma ({gamma_score}) conditions met with Delta ({delta_score:.2f})."
                    logging.info(orchestrator_reasoning)
                    response_content = generate_random_walk_response(prompt_word_freq, greek_sensitivities_data)
                    response_type_for_debug = "RandomWalkResponse"
                # 3c. Scatterplot Conditions
                elif (delta_score < SCATTER_DELTA_LOW_THRESHOLD or gamma_score >= SCATTER_GAMMA_THRESHOLD):
                    orchestrator_reasoning = f"Scatterplot: Delta ({delta_score:.2f}) / Gamma ({gamma_score}) conditions met."
                    logging.info(orchestrator_reasoning)
                    response_content = generate_scatterplot_response(prompt_word_freq, greek_sensitivities_data)
                    response_type_for_debug = "ScatterplotResponse"
                # 3d. Fallback to Low Confidence Core Rule
                else:
                    orchestrator_reasoning = f"Low Confidence Core Rule: No dynamic strategy triggered, using best available core rule despite low confidence (Delta: {delta_score:.2f})."
                    logging.info(orchestrator_reasoning)
                    # Find best available rule even if below CORE_MATCH_CONFIDENCE_THRESHOLD
                    best_low_conf_rule = None
                    if greek_sensitivities_data.get("closest_core_rules_analysis"):
                        top_rule_name = greek_sensitivities_data["closest_core_rules_analysis"][0]["name"]
                        best_low_conf_rule = next((r for r in PROBABILITY_MAP_CORE if r.get("rule_name") == top_rule_name), None)

                    if not best_low_conf_rule: # If no rule even remotely matched, use generic observation
                        best_low_conf_rule = next((r for r in PROBABILITY_MAP_CORE if r.get("rule_name") == "DefaultGenericObservation"), PROBABILITY_MAP_CORE[0] if PROBABILITY_MAP_CORE else None)

                    if best_low_conf_rule:
                        selected_rule_object_for_debug = best_low_conf_rule
                        logging.info(f"Low-Confidence Core Rule Selected: '{best_low_conf_rule.get('rule_name')}'")
                        core_adjacent_template = generate_core_adjacent_response(lower_prompt, best_low_conf_rule, prompt_word_freq, PROBABILITY_MAP_CORE)
                        response_content = substitute_placeholders(core_adjacent_template, lower_prompt, best_low_conf_rule.get("entity_placeholders", []))
                        response_type_for_debug = "LowConfidenceCoreRule"
                    else: # Absolute fallback if map is totally empty (should be caught by is_map_functional)
                        orchestrator_reasoning = "Critical Fallback: No rules available and map non-functional."
                        logging.error(orchestrator_reasoning)
                        response_content = "Service is unable to process the request due to missing rule configurations."
                        response_type_for_debug = "ErrorFallback"


        # 5. Construct Final JSON Response
        logging.info(f"Mock LLM: Final Response Content: {response_content}")
        final_response_data = {
            "choices": [{"message": {"role": "assistant", "content": response_content}}],
            "debug_info": {
                "response_type": response_type_for_debug,
                "orchestrator_reasoning": orchestrator_reasoning,
                "matched_rule_name": selected_rule_object_for_debug.get("rule_name") if selected_rule_object_for_debug else None,
                "core_confidence_of_matched_rule": selected_rule_object_for_debug.get("confidence") if selected_rule_object_for_debug and response_type_for_debug not in ["ScatterplotResponse", "RandomWalkResponse", "DarkForestResponse"] else None,
                "overall_prompt_match_to_best_rule_keywords_delta": delta_score, # This is the general delta
                "adam_principles_hint": selected_rule_object_for_debug.get("adam_principles_invoked", []) if selected_rule_object_for_debug else (["Transparency & Explainability"] if response_type_for_debug == "DarkForestResponse" else []),
                "greek_vector_analysis": greek_sensitivities_data
            }
        }
        return jsonify(final_response_data)
    except Exception as e:
        logging.error(f"Error in mock_complete: {e}", exc_info=True)
        return jsonify({"error": str(e), "message": f"Error in mock_complete: {e}"}), 500

# --- Flask App Startup ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting Mock LLM Service...")
    load_probability_map()
    if PROBABILITY_MAP_CORE:
        logging.info(f"Mock LLM Service initialized with {len(PROBABILITY_MAP_CORE)} rule(s). First rule: {PROBABILITY_MAP_CORE[0].get('rule_name')}")
    else:
        logging.error("PROBABILITY_MAP_CORE is empty after loading. Service might not function as expected.")
    app.run(port=5001, debug=True)
