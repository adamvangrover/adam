import json
import os
import re

class SNCAnalystWorkflow:
    """
    A portable, modular workflow engine for the SNC Analyst pipeline.
    It chains together AOPL prompts to perform end-to-end reasoning.
    """

    def __init__(self, prompt_library_path="prompt_library/AOPL-v1.0"):
        self.lib_path = prompt_library_path
        self.state = {}

    def load_prompt(self, relative_path):
        """Loads a prompt template from the library."""
        full_path = os.path.join(self.lib_path, relative_path)
        with open(full_path, "r") as f:
            return f.read()

    def run_ingestion(self, raw_text):
        """
        Step 1: Document Ingestion
        Simulates the execution of AOPL-ING-001.
        """
        print(f"[Workflow] Step 1: Ingesting Document ({len(raw_text)} chars)...")
        prompt_template = self.load_prompt("ingestion/financial_document_parser.md")

        # In a real system, this would call an LLM.
        # Here, we simulate the output for the portable showcase.
        simulated_output = {
            "mda": raw_text[:500], # Mocking segmentation
            "financials": "Revenue: $4.2B, EBITDA: $777M, Debt: $2.1B",
            "risk_factors": "Supply chain volatility..."
        }

        self.state["ingestion_result"] = simulated_output
        print("[Workflow] Ingestion Complete. Sections identified: " + ", ".join(simulated_output.keys()))
        return simulated_output

    def run_data_extraction(self):
        """
        Step 2: Data Extraction
        Simulates AOPL-OS-004.
        """
        print("[Workflow] Step 2: Extracting Financial Data...")
        prompt_template = self.load_prompt("analyst_os/data_extraction.md")
        source_text = self.state["ingestion_result"]["financials"]

        # Simulated extraction logic (Hardcoded for demo consistency with snc_cover.html)
        extracted_data = {
            "revenue_current": "$4.2B",
            "revenue_yoy_var": "+12%",
            "revenue_direction": "Expanded",
            "ebitda_current": "$777M",
            "ebitda_margin": "18.5%",
            "margin_direction": "Contracted",
            "net_leverage": "2.8x",
            "leverage_direction": "Improved",
            "liquidity_avail": "$350M",
            "liquidity_status": "Strong",
            "covenant_max": "4.5x",
            "covenant_status": "well below",
            "qualitative_drivers": {
                "driver_1": "increased volume",
                "driver_2": "inflationary pressure",
                "driver_3": "supply chain disruptions"
            },
            "segment_name": "North American"
        }

        self.state["extracted_data"] = extracted_data
        print(f"[Workflow] Data Extracted: Leverage={extracted_data['net_leverage']}, Revenue={extracted_data['revenue_current']}")
        return extracted_data

    def run_skeleton_generation(self):
        """
        Step 3: Narrative Skeleton Generation
        Simulates AOPL-OS-001.
        """
        print("[Workflow] Step 3: Generating Narrative Skeleton...")
        prompt_template = self.load_prompt("analyst_os/skeleton_generation.md")

        # The prompt defines the output format. We simulate the LLM following that instruction.
        skeleton = (
            "Top-line performance was {{revenue_direction}} year-over-year, settling at {{revenue_current}}. "
            "This variance of {{revenue_yoy_var}} was primarily driven by {{driver_1}} in the {{segment_name}} segment.\n\n"
            "EBITDA margins {{margin_direction}} to {{ebitda_margin}}, reflecting the {{driver_2}}. "
            "Management has noted that {{driver_3}} will likely persist into the next quarter.\n\n"
            "Liquidity remains {{liquidity_status}} with {{liquidity_avail}} available under the revolver. "
            "The leverage profile has {{leverage_direction}} to {{net_leverage}}, which is {{covenant_status}} the maximum covenant of {{covenant_max}}."
        )

        self.state["skeleton"] = skeleton
        print("[Workflow] Skeleton Generated (No numbers, placeholders only).")
        return skeleton

    def run_injection(self):
        """
        Step 4: Data Injection
        Merges Step 2 (JSON) into Step 3 (Skeleton).
        """
        print("[Workflow] Step 4: Injecting Data into Skeleton...")
        skeleton = self.state["skeleton"]
        data = self.state["extracted_data"]
        drivers = data["qualitative_drivers"]

        # Flatten for easy replacement
        flat_data = {**data, **drivers}

        final_text = skeleton
        for key, value in flat_data.items():
            if key != "qualitative_drivers":
                # Case insensitive replacement for placeholders
                pattern = re.compile(re.escape("{{" + key + "}}"), re.IGNORECASE)
                final_text = pattern.sub(str(value), final_text)

        self.state["final_memo"] = final_text
        print("[Workflow] Injection Complete. Memo Ready.")
        print("-" * 20)
        print(final_text)
        print("-" * 20)
        return final_text

    def run_rating_logic(self):
        """
        Step 5: Regulatory Rating
        Simulates AOPL-SNC-001.
        """
        print("[Workflow] Step 5: Determining Regulatory Rating...")
        prompt_template = self.load_prompt("snc/regulatory_rating_logic.md")
        data = self.state["extracted_data"]

        # Logic simulation
        leverage = float(data["net_leverage"].replace("x", ""))

        rating = "Pass"
        rationale = f"Leverage at {leverage}x is well below the 6.0x regulatory threshold. Liquidity is {data['liquidity_status']}."

        if leverage > 6.0:
            rating = "Special Mention"
            rationale = f"Leverage at {leverage}x exceeds the 6.0x guidance, warranting closer monitoring."

        result = {"rating": rating, "rationale": rationale}
        self.state["rating_decision"] = result
        print(f"[Workflow] Decision: {rating.upper()} | {rationale}")
        return result

if __name__ == "__main__":
    # Test Run
    workflow = SNCAnalystWorkflow()

    sample_text = """
    Titan Energy Partners Q3 Earnings Call.
    We are pleased to report revenue of $4.2 billion, up 12% YoY.
    Adjusted EBITDA was $777 million, representing a margin of 18.5%.
    Net Leverage ended the quarter at 2.8x.
    """

    workflow.run_ingestion(sample_text)
    workflow.run_data_extraction()
    workflow.run_skeleton_generation()
    workflow.run_injection()
    workflow.run_rating_logic()
