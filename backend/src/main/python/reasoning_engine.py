import json
import yaml # For loading YAML files
import os
from typing import List, Dict, Optional, Any

from .knowledge_graph import KnowledgeGraph, Node
from .kg_builder import get_kg_instance
# Assuming impact_calculator.py is in the same directory or accessible via PYTHONPATH
from .impact_calculator import calculate_driver_impacts as calculate_impacts_func


# --- LLM Integration (Simulation) ---
def simulate_llm_call(prompt: str, company_name: str) -> str:
    print("\n--- SIMULATING LLM CALL ---")
    print(f"Prompt for {company_name} (first 500 chars):\n{prompt[:500]}...\n") # Reduce noise: print(f"Prompt for {company_name} (first 500 chars):\n{prompt[:500]}...\n")
    if "Apple Inc." in company_name or "AAPL" in prompt:
        # Example of how LLM might incorporate calculated impacts if they were in the prompt
        if "Calculated Value/Change: -2.00" in prompt and "Probability: 65%" in prompt:
             return (
                f"Synthesized narrative for {company_name} (AAPL): Apple's strong innovation (DRV002, DRV006) drives growth. "
                "However, it faces macroeconomic pressures; for instance, interest rate sensitivity (DRV001) indicates a potential P/E compression "
                "of approximately -2.0 points with a 65% probability under current rate hike scenarios. " # Example based on a hypothetical calc
                "Supply chain vulnerabilities (DRV004) also persist. The overall outlook is cautiously optimistic, balancing innovation against these headwinds."
            )
        return (
            f"Synthesized narrative for {company_name} (AAPL): Apple's strong innovation (DRV002, DRV006) continues to drive growth, "
            "but it must navigate supply chain vulnerabilities (DRV004) and macroeconomic pressures like interest rate changes (DRV001). "
            "The balance of these factors suggests cautious optimism, contingent on effective risk mitigation."
        )
    elif "Microsoft Corp." in company_name or "MSFT" in prompt:
        return (
            f"Synthesized narrative for {company_name} (MSFT): Microsoft's diversified model, particularly strength in cloud and AI (related to DRV006), "
            "positions it well. It remains susceptible to broad economic shifts (DRV001). "
            "Overall outlook is positive, assuming continued market leadership in key segments."
        )
    else:
        return (
            f"A comprehensive analysis for {company_name} indicates a complex interplay of its specific drivers (e.g., DRV_COMPANY_SPECIFIC_XYZ) "
            "and broader market conditions. Key opportunities may be balanced by certain operational or market risks. "
            "Refer to detailed driver analysis and calculated impacts for deeper insights."
        )
    print("--- LLM SIMULATION COMPLETE ---\n")


class ReasoningEngine:
    def __init__(self, kg: Optional[KnowledgeGraph] = None, semantic_library_path: str = "semantic_library"):
        self.kg = kg if kg is not None else get_kg_instance()
        self.semantic_library_path = self._get_semantic_library_path(semantic_library_path)
        self.drivers_catalog = self._load_yaml_file(os.path.join(self.semantic_library_path, 'drivers_knowledge_base.yaml'))
        self.narrative_strategies = self._load_yaml_file(os.path.join(self.semantic_library_path, 'narrative_strategies.yaml'))
        self.narrative_templates = self.narrative_strategies # Alias for potential backward compatibility

    def _get_semantic_library_path(self, relative_path: str) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(base_dir, relative_path)

    def _load_yaml_file(self, filepath: str) -> Optional[Dict]:
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: YAML file not found at {filepath}")
            return None
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {filepath}: {e}")
            return None

    def get_driver_definition(self, driver_id: str) -> Optional[Dict]:
        if self.drivers_catalog and 'drivers' in self.drivers_catalog:
            for driver_def in self.drivers_catalog['drivers']:
                if driver_def['id'] == driver_id:
                    return driver_def
        return None

    def get_narrative_strategy(self, strategy_id: str) -> Optional[Dict]:
        if self.narrative_strategies and 'strategies' in self.narrative_strategies:
            for strategy_def in self.narrative_strategies['strategies']:
                if strategy_def['strategy_id'] == strategy_id:
                    return strategy_def
        return None

    def get_driver_details(self, driver_id: str) -> Optional[Dict]:
        driver_node = self.kg.get_node(driver_id)
        if driver_node and driver_node.label == 'Driver':
            return driver_node.properties
        return None

    def get_company_direct_drivers(self, company_id: str) -> List[Node]:
        company_node = self.kg.get_node(company_id)
        if not company_node or company_node.label != 'Company': return []
        driver_nodes = self.kg.get_neighbors(company_id, relationship_type='AFFECTED_BY_DRIVER')
        return [node for node in driver_nodes if node.label == 'Driver']

    def get_company_industry_drivers(self, company_id: str) -> List[Node]:
        company_node = self.kg.get_node(company_id)
        if not company_node or company_node.label != 'Company': return []
        industry_nodes = self.kg.get_neighbors(company_id, relationship_type='BELONGS_TO_INDUSTRY')
        if not industry_nodes: return []
        industry_node = industry_nodes[0]
        if not industry_node or industry_node.label != 'Industry': return []
        industry_driver_nodes = self.kg.get_neighbors(industry_node.id, relationship_type='AFFECTED_BY_DRIVER')
        return [node for node in industry_driver_nodes if node.label == 'Driver']

    def get_all_company_drivers(self, company_id: str) -> List[Dict]:
        direct_drivers = self.get_company_direct_drivers(company_id)
        industry_drivers = self.get_company_industry_drivers(company_id)
        all_drivers_map: Dict[str, Node] = {driver.id: driver for driver in direct_drivers}
        for driver_node in industry_drivers:
            if driver_node.id not in all_drivers_map:
                all_drivers_map[driver_node.id] = driver_node
        return [driver.properties for driver in all_drivers_map.values()]

    def _build_impact_calculation_context(self, company_id: str, company_node: Node) -> Dict[str, Any]:
        context = {"company": {}, "macro": {}, "industry": {}}
        if company_node and company_node.properties:
            context["company"]["id"] = company_id
            context["company"]["name"] = company_node.properties.get("name")
            company_financials = company_node.properties.get('financials', {})
            if company_financials:
                for k, v in company_financials.items(): context["company"][f"financials.{k}"] = v
            company_trading_levels = company_node.properties.get('tradingLevels', {})
            if company_trading_levels:
                for k, v in company_trading_levels.items(): context["company"][f"tradingLevels.{k}"] = v
            context["company"]["industryId"] = company_node.properties.get("industryId")
            # Add other company-specific context values referenced by driver parameters if needed
            # Example: context["company"]["product_launch_status"] = "successful" (from KG or event system)
            # context["company"]["product_competitive_advantage"] = "high"
            # context["company"]["product_media_coverage"] = "positive"


        industry_nodes = self.kg.get_neighbors(company_id, relationship_type='BELONGS_TO_INDUSTRY')
        if industry_nodes:
            industry_node = industry_nodes[0]
            context["industry"]["id"] = industry_node.id
            context["industry"]["name"] = industry_node.properties.get("name")

        macro_factor_ids_to_load = ["MACRO_IR", "MACRO_GDP", "MACRO_INFL", "MACRO_SC"]
        for mf_id in macro_factor_ids_to_load:
            mf_node = self.kg.get_node(mf_id)
            if mf_node and mf_node.properties:
                context["macro"][mf_id] = mf_node.properties
                if 'currentValue' in mf_node.properties: context["macro"][f"{mf_id}.currentValue"] = mf_node.properties['currentValue']
                if 'trend' in mf_node.properties: context["macro"][f"{mf_id}.trend"] = mf_node.properties['trend']

        # For DRV001 conditions and parameters (example values, should come from dynamic context)
        if context.get("macro", {}).get("MACRO_IR", {}).get("trend") == "Increasing":
             context["macro.interest_rate_trend"] = "Increasing"
             context["macro.MACRO_IR.change_pct_for_nim_calc"] = -0.005 # e.g. -0.5% NIM impact for 100bps rate hike
             context["macro.MACRO_IR.pe_compression_factor"] = -0.10   # e.g. -10% P/E impact
             context["macro.MACRO_IR.rate_increase_abs"] = 100         # e.g. 100 bps absolute increase
        elif context.get("macro", {}).get("MACRO_IR", {}).get("trend") == "Stable":
             context["macro.interest_rate_trend"] = "Stable"
             context["macro.MACRO_IR.change_pct_for_nim_calc"] = 0.0
             context["macro.MACRO_IR.pe_compression_factor"] = 0.0
             context["macro.MACRO_IR.rate_increase_abs"] = 0.0
        else:
             context["macro.interest_rate_trend"] = context.get("macro", {}).get("MACRO_IR", {}).get("trend", "Unknown")
             context["macro.MACRO_IR.change_pct_for_nim_calc"] = 0.002 # Example for decreasing
             context["macro.MACRO_IR.pe_compression_factor"] = 0.05  # Example for decreasing
             context["macro.MACRO_IR.rate_increase_abs"] = -50     # Example for decreasing

        # For DRV002 (Product Launch) conditions (example values)
        context["company.product_launch_status"] = "successful" # This should ideally come from KG/event data
        context["company.product_competitive_advantage"] = "high"
        context["company.product_media_coverage"] = "positive"


        return context

    def _get_calculated_impacts_for_company(self, company_id: str, company_node: Node, active_driver_ids: List[str]) -> List[Dict]:
        if not self.drivers_catalog or 'drivers' not in self.drivers_catalog:
            print("Warning: Drivers knowledge base not loaded. Cannot calculate impacts.")
            return []
        all_calculated_impacts = []
        calculation_context = self._build_impact_calculation_context(company_id, company_node)
        for driver_id in active_driver_ids:
            driver_definition = self.get_driver_definition(driver_id)
            if not driver_definition:
                print(f"Warning: Definition for driver '{driver_id}' not found in catalog. Skipping impact calculation.")
                continue
            calculated_impacts_for_driver = calculate_impacts_func(driver_definition, calculation_context)
            all_calculated_impacts.extend(calculated_impacts_for_driver)
        return all_calculated_impacts

    def _substitute_placeholders(self, text: str, context: Dict[str, Any]) -> str:
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if isinstance(value, (str, int, float)):
                text = text.replace(placeholder, str(value))
        return text

    def _build_llm_prompt_from_template(
        self,
        company_node: Node,
        active_drivers_properties: List[Dict],
        calculated_impacts: List[Dict],
        strategy_id: str
    ) -> str:
        template = self.get_narrative_strategy(strategy_id)
        if not template:
            return f"Error: Narrative strategy '{strategy_id}' not found. Please define it in narrative_strategies.yaml."

        company_properties = company_node.properties
        company_name = company_properties.get('name', company_node.id)

        template_context = {
            "company_name": company_name,
            "company_id": company_node.id,
        }
        industry_nodes = self.kg.get_neighbors(company_node.id, relationship_type='BELONGS_TO_INDUSTRY')
        if industry_nodes:
            template_context["industry_name"] = industry_nodes[0].properties.get('name', 'N/A')

        prompt_parts = []
        prompt_parts.append(f"Task: Generate a financial narrative for {company_name} based on the '{template.get('description', strategy_id)}' template.")
        prompt_parts.append(f"Target Audience: {template.get('target_audience', 'General')}")

        for block in template.get('narrative_flow', []):
            block_type = block.get('section')

            block_prompt = f"\n--- Section: {block_type if block_type else 'Unnamed Block'} ---"

            if block.get('llm_instructions'):
                block_prompt += "\nInstructions for this section:\n"
                instructions = block['llm_instructions']
                if isinstance(instructions, list):
                    for instr in instructions:
                        block_prompt += f"- {self._substitute_placeholders(instr, template_context)}\n"
                else:
                    block_prompt += f"- {self._substitute_placeholders(instructions, template_context)}\n"

            if block_type == "KEY_ACTIVE_DRIVERS_IDENTIFIED":
                block_prompt += "Relevant Active Drivers Data (Qualitative Overview):\n"
                if not active_drivers_properties:
                    block_prompt += "  - No specific drivers were pre-identified for focused analysis.\n"
                else:
                    for i, driver_props in enumerate(active_drivers_properties):
                        driver_id_from_kg = driver_props.get('id', 'N/A')
                        driver_def_from_catalog = self.get_driver_definition(driver_id_from_kg)

                        block_prompt += f"  Driver {i+1}: {driver_props.get('name', 'N/A')} ({driver_id_from_kg})\n"
                        block_prompt += f"    Type: {driver_props.get('type', 'N/A')}\n"

                        if driver_def_from_catalog and driver_def_from_catalog.get('narrative_logic', {}).get('key_insights_to_extract'):
                             insights = driver_def_from_catalog['narrative_logic']['key_insights_to_extract']
                             insights_to_print = insights if isinstance(insights, list) else [insights]
                             for insight in insights_to_print:
                                 block_prompt += f"    Insight: {self._substitute_placeholders(insight, template_context)}\n"
                        else:
                            block_prompt += f"    Description: {driver_props.get('description', 'No description available in KG properties.')}\n"
            elif block_type == "KEY_FINANCIAL_HIGHLIGHTS":
                block_prompt += "Key Financials (from company data):\n"
                company_financials = company_properties.get('financials', {})
                if company_financials:
                    for key, value in company_financials.items():
                        if value is not None:
                             block_prompt += f"    - {key.replace('_', ' ').title()}: {value}\n"
                else:
                    block_prompt += "  - No specific financial highlights available in context.\n"

            elif block_type == "QUANTITATIVE_IMPACT_ANALYSIS":
                block_prompt += "Calculated Probabilistic Impacts of Key Drivers:\n"
                if not calculated_impacts:
                    block_prompt += "  - No specific quantitative impacts were calculated or provided for the active drivers.\n"
                else:
                    impacts_by_driver: Dict[str, List[Dict]] = {}
                    for impact in calculated_impacts:
                        src_driver_id = impact.get("source_driver_id", "UnknownDriver")
                        if src_driver_id not in impacts_by_driver: impacts_by_driver[src_driver_id] = []
                        impacts_by_driver[src_driver_id].append(impact)

                    for driver_id_from_impact, impacts_list in impacts_by_driver.items():
                        driver_def_for_impact = self.get_driver_definition(driver_id_from_impact)
                        driver_name_for_impact = driver_def_for_impact.get('name', driver_id_from_impact) if driver_def_for_impact else driver_id_from_impact
                        block_prompt += f"  Driver: {driver_name_for_impact} ({driver_id_from_impact}):\n"
                        for i, impact_data in enumerate(impacts_list):
                            block_prompt += f"    Impact {i+1}:\n"
                            block_prompt += f"      Target Variable: {impact_data.get('target_variable', 'N/A')}\n"
                            block_prompt += f"      Description: {impact_data.get('effect_description', 'N/A')}\n"
                            calc_val = impact_data.get('calculated_impact_value', 'N/A')
                            if isinstance(calc_val, float): calc_val_str = f"{calc_val:.2f}" # Format float
                            else: calc_val_str = str(calc_val)
                            block_prompt += f"      Calculated Value/Change: {calc_val_str}\n"
                            prob = impact_data.get('probability_of_occurrence')
                            prob_display = f"{prob*100:.0f}%" if isinstance(prob, (float, int)) else 'N/A'
                            block_prompt += f"      Probability: {prob_display}\n"
                            block_prompt += f"      Time Horizon: {impact_data.get('time_horizon', 'N/A')}\n"
                            block_prompt += f"      Conditions Assumed: {str(impact_data.get('conditions_evaluated', 'N/A'))}\n" # Ensure conditions are string

                            if driver_def_for_impact and driver_def_for_impact.get('narrative_logic', {}).get('explanation_patterns_llm'):
                                patterns = driver_def_for_impact['narrative_logic']['explanation_patterns_llm']
                                pattern_specific_context = {
                                    **template_context,
                                    "driver_name": driver_name_for_impact,
                                    "target_variable": impact_data.get('target_variable', 'N/A'),
                                    "calculated_impact_value": calc_val_str,
                                    "probability_percentage": prob_display,
                                    "time_horizon": impact_data.get('time_horizon', 'N/A'),
                                    "effect_description": impact_data.get('effect_description', 'N/A')
                                }
                                if isinstance(patterns, list):
                                    for pattern in patterns:
                                        block_prompt += f"      LLM Guidance: {self._substitute_placeholders(pattern, pattern_specific_context)}\n"
                                elif isinstance(patterns, str):
                                    block_prompt += f"      LLM Guidance: {self._substitute_placeholders(patterns, pattern_specific_context)}\n"
                        block_prompt += "\n"
            prompt_parts.append(block_prompt)

        if template.get('overall_llm_instructions'):
            prompt_parts.append("\n--- Overall Instructions for Generation ---")
            overall_instr = template['overall_llm_instructions']
            if isinstance(overall_instr, list):
                for instr in overall_instr:
                    prompt_parts.append(f"- {self._substitute_placeholders(instr, template_context)}")
            else:
                prompt_parts.append(f"- {self._substitute_placeholders(overall_instr, template_context)}")

        return "\n".join(prompt_parts)

    def get_structured_explanation_data(self, company_id: str) -> Dict:
        company_node = self.kg.get_node(company_id)
        if not company_node or company_node.label != 'Company':
            return {"error": "Company not found"}

        active_drivers_properties = self.get_all_company_drivers(company_id)
        active_driver_ids = [d['id'] for d in active_drivers_properties if 'id' in d]

        calculated_impacts = self._get_calculated_impacts_for_company(company_id, company_node, active_driver_ids)

        return {
            "company_id": company_id,
            "company_name": company_node.properties.get('name'),
            "company_node_properties": company_node.properties,
            "num_drivers_found": len(active_drivers_properties),
            "drivers": active_drivers_properties,
            "calculated_impacts": calculated_impacts
        }

    def generate_narrative_explanation_with_llm(self, company_id: str, strategy_id: str = "STRAT_COMPANY_FINANCIAL_IMPACT_OVERVIEW") -> Dict:
        structured_data = self.get_structured_explanation_data(company_id)
        if "error" in structured_data: return structured_data

        company_node = self.kg.get_node(company_id)
        if not company_node: return {"error": "Company node vanished unexpectedly"}

        prompt = self._build_llm_prompt_from_template(
            company_node,
            structured_data["drivers"],
            structured_data.get("calculated_impacts", []),
            strategy_id
        )

        if prompt.startswith("Error:"):
            return {"error": prompt}

        llm_narrative = simulate_llm_call(prompt, company_name=structured_data["company_name"])

        return {
            "company_id": company_id,
            "company_name": structured_data["company_name"],
            "num_drivers_found": structured_data["num_drivers_found"],
            "drivers": structured_data["drivers"], # These are from KG, not from catalog
            "calculated_impacts": structured_data.get("calculated_impacts", []),
            "narrative_summary": llm_narrative,
            "strategy_used": strategy_id,
            # "debug_prompt": prompt
        }


if __name__ == '__main__':
    engine = ReasoningEngine()

    if not engine.drivers_catalog: print("Failed to load drivers_knowledge_base.yaml")
    if not engine.narrative_strategies: print("Failed to load narrative_strategies.yaml")

    company_ticker = "AAPL"
    print(f"\n--- Structured Data for {company_ticker} (including calculated impacts) ---")
    structured_data = engine.get_structured_explanation_data(company_ticker)
    if "error" in structured_data:
        print(f"Error getting structured data: {structured_data['error']}")
    else:
        print(f"Company: {structured_data['company_name']}")
        print(f"Number of Drivers from KG: {structured_data['num_drivers_found']}")

        print("\nCalculated Impacts:")
        if structured_data.get('calculated_impacts'):
            for impact in structured_data['calculated_impacts']:
                print(json.dumps(impact, indent=2))
        else:
            print("  No impacts calculated or returned.")

    print(f"\n--- Reasoning for {company_ticker} (Strategy: STRAT_COMPANY_FINANCIAL_IMPACT_OVERVIEW) ---")
    explanation = engine.generate_narrative_explanation_with_llm(company_ticker, strategy_id="STRAT_COMPANY_FINANCIAL_IMPACT_OVERVIEW")
    if "error" in explanation:
        print(f"Error: {explanation['error']}")
    else:
        print(f"\nCompany: {explanation['company_name']}")
        print(f"Strategy Used: {explanation.get('strategy_used')}")
        print("\nLLM-Generated Narrative Summary:")
        print(explanation.get("narrative_summary"))
        # print("\n--- DEBUG PROMPT ---")
        # print(explanation.get("debug_prompt"))


    company_ticker_other = "JPM"
    print(f"\n--- Reasoning for {company_ticker_other} (Strategy: STRAT_RISK_DEEP_DIVE) ---")
    explanation_other = engine.generate_narrative_explanation_with_llm(company_ticker_other, strategy_id="STRAT_RISK_DEEP_DIVE")
    if "error" in explanation_other:
        print(f"Error: {explanation_other['error']}")
    else:
        print(f"\nLLM-Generated Narrative Summary for {explanation_other['company_name']}:")
        print(explanation_other.get("narrative_summary"))
```
