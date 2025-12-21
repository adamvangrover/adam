
import os
import sys
from datetime import datetime

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.newsletter_layout.generator import NewsletterGenerator
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    generator = NewsletterGenerator()
    timestamp_str = datetime.now().strftime("%Y%m%d")
    output_base = "core/libraries_and_archives/newsletters"

    # 1. Market Mayhem
    mm_path = os.path.join(output_base, f"Market_Mayhem_{timestamp_str}.md")
    logger.info("Generating Market Mayhem...")
    generator.generate("market_mayhem.md", mm_path)

    # 2. Tech Watch
    tw_path = os.path.join(output_base, f"Tech_Watch_{timestamp_str}.md")
    logger.info("Generating Tech Watch...")
    generator.generate("tech_watch.md", tw_path)

    # 3. Weekly Recap (Rich Format)
    wr_path = os.path.join(output_base, f"Weekly_Recap_{timestamp_str}.md")
    logger.info("Generating Weekly Recap...")
    generator.generate("weekly_recap.md", wr_path)

    # 4. Industry Report
    ir_path = os.path.join(output_base, f"Industry_Report_{timestamp_str}.md")
    logger.info("Generating Industry Report...")
    generator.generate("industry_report.md", ir_path)

    # 5. Deep Dive
    dd_path = os.path.join(output_base, f"Deep_Dive_{timestamp_str}.md")
    logger.info("Generating Deep Dive...")
    generator.generate("deep_dive.md", dd_path)

    # 6. Equity Research
    er_path = os.path.join(output_base, f"Equity_Research_{timestamp_str}.md")
    logger.info("Generating Equity Research...")
    generator.generate("equity_research.md", er_path)

    # 7. House View
    hv_path = os.path.join(output_base, f"House_View_{timestamp_str}.md")
    logger.info("Generating House View...")
    generator.generate("house_view.md", hv_path)

    print("Newsletters generated successfully.")
    print(f"- {mm_path}")
    print(f"- {tw_path}")
    print(f"- {wr_path}")
    print(f"- {ir_path}")
    print(f"- {dd_path}")
    print(f"- {er_path}")
    print(f"- {hv_path}")

if __name__ == "__main__":
    main()
