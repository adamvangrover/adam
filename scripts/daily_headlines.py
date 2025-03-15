import requests
import feedparser
import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os
from time import sleep

# --- Developer Notes ---
# This script fetches headlines from various RSS feeds and sends them in a daily email.
#
# Key improvements and expansion areas:
# 1. More comprehensive list of RSS feeds covering the requested categories.
# 2. Enhanced error handling for network issues, feed parsing, and email sending.
# 3. Clearer separation of concerns with functions for each step.
# 4. Use of environment variables for sensitive information (email credentials).
# 5. Basic HTML formatting for better readability in the email.
# 6. Placeholder comments for potential future enhancements (e.g., keyword filtering).
#
# To use this script:
# 1. Install the required libraries: `pip install requests feedparser`
# 2. Set up environment variables for your email address, password, and optionally the receiver address.
#    - On Linux/macOS: `export HEADLINE_EMAIL_ADDRESS="your_email@example.com"` and so on.
#    - On Windows: Use the `set` command in the command prompt or PowerShell.
# 3. Populate the `rss_feeds` dictionary with the URLs of your preferred RSS feeds for each category.
# 4. Schedule this script to run daily using cron (Linux/macOS) or Task Scheduler (Windows).

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define your RSS feed URLs here ---
rss_feeds = {
    "Finance - Bloomberg": "https://www.bloomberg.com/feeds/markets.rss",
    "Finance - Wall Street Journal": "https://www.wsj.com/xml/rss/3_7085.xml", # Markets section
    "Markets - Financial Times": "https://ft.com/rss/markets",
    "AI & Technology - TechCrunch": "https://techcrunch.com/feed/",
    "AI & Technology - The Verge": "https://www.theverge.com/rss/index.xml",
    "Software - Hacker News": "https://news.ycombinator.com/rss",
    # Add more software-related feeds here, e.g., from specific technology blogs.
    "Healthcare - Reuters": "https://www.reuters.com/rss/newsPackage/health-news",
    # Consider feeds from STAT News, Kaiser Health News, etc.
    "Politics - Associated Press": "https://rss.app/feeds/91QDwogUu6CMzRrE.xml", # Example AP Politics feed
    # Add feeds from other preferred political news sources.
    "Commodities - Reuters": "https://www.reuters.com/rss/newsPackage/commoditiesNews/",
    # Look for more specific feeds for different commodities if desired.
    "Private Credit - Private Debt Investor": "https://www.privatedebtinvestor.com/feed/", # Example, verify if correct
    # Search for other relevant Private Credit news feeds.
    "M&A - Mergers & Acquisitions": "https://www.themiddlemarket.com/feed", # Example, verify if correct
    # Look for M&A sections on major financial news sites like Bloomberg or Reuters.
}

# --- Email Configuration ---
EMAIL_ADDRESS = os.getenv("HEADLINE_EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("HEADLINE_EMAIL_PASSWORD")
RECEIVER_ADDRESS = os.getenv("HEADLINE_RECEIVER_ADDRESS", EMAIL_ADDRESS)  # Defaults to sender if not set

# --- Constants ---
REQUEST_TIMEOUT_SECONDS = 15
SMTP_SERVER = 'smtp.gmail.com' # Consider your email provider's SMTP server
SMTP_PORT = 465 # For SSL

# --- Helper functions ---
def fetch_and_parse_headlines(feed_url):
    """
    Fetches and parses headlines from an RSS feed.

    Developer Note:
    - Implements a timeout to prevent indefinite hanging on unresponsive feeds.
    - Uses try-except blocks for robust error handling.
    - Consider adding retry logic for transient network errors if needed.
    """
    try:
        logging.info(f"Fetching feed from: {feed_url}")
        response = requests.get(feed_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()  # Raise exception for bad status codes
        feed = feedparser.parse(response.content)
        headlines = [(entry.title, entry.link) for entry in feed.entries]
        logging.info(f"Successfully fetched {len(headlines)} headlines from {feed_url}")
        return headlines
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching feed from {feed_url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error parsing feed from {feed_url}: {e}")
        return []

def format_email_body(all_headlines):
    """
    Formats the headlines into an HTML email body.

    Developer Note:
    - Uses basic HTML for formatting; this can be expanded for more sophisticated layouts.
    - Includes a timestamp to indicate when the email was generated.
    - If a category has no headlines, it explicitly states that.
    """
    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; margin: 20px; }}
          h2 {{ color: #0056b3; }}
          h3 {{ color: #333; margin-top: 1.5em; }}
          ul {{ list-style-type: none; padding: 0; }}
          li {{ margin-bottom: 0.5em; }}
          a {{ text-decoration: none; font-weight: bold; color: #007bff; }}
          a:hover {{ text-decoration: underline; }}
          .no-headlines {{ color: gray; font-style: italic; }}
        </style>
      </head>
      <body>
        <h2>Daily Headlines for {datetime.now().strftime('%Y-%m-%d')}</h2>
        <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}</p>
    """

    for category, headlines in all_headlines.items():
        html_body += f"<h3>{category}</h3><ul>"
        if headlines:
            for title, link in headlines:
                html_body += f"<li><a href='{link}'>{title}</a></li>"
        else:
            html_body += f"<li class='no-headlines'>No headlines found for {category}.</li>"
        html_body += "</ul>"

    html_body += """
      </body>
    </html>
    """
    return html_body

def send_email(subject, body, receiver_email):
    """
    Sends the email with the formatted headlines.

    Developer Note:
    - Uses SMTP_SSL for secure email transmission.
    - Includes error handling for SMTP-related issues.
    - Consider using a more robust email sending service for production environments.
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logging.error("Error: Email sender address and password not configured in environment variables.")
        return

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            msg = MIMEText(body, 'html')
            msg['Subject'] = subject
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = receiver_email
            server.sendmail(EMAIL_ADDRESS, receiver_email, msg.as_string())
            logging.info(f"Email sent successfully to {receiver_email}")
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error while sending email: {e}")
    except Exception as e:
        logging.error(f"Error sending email: {e}")

def validate_config():
    """
    Validates the presence of required environment variables and RSS feeds.

    Developer Note:
    - Logs critical errors and returns False if essential configuration is missing.
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logging.critical("Email address and password must be set in environment variables.")
        return False
    if not rss_feeds:
        logging.critical("No RSS feeds are configured. Please add feed URLs to the 'rss_feeds' dictionary.")
        return False
    return True

def main():
    """
    Fetches headlines from multiple feeds, formats them into an HTML email, and sends the email.

    Developer Note:
    - The main function orchestrates the entire process.
    - It iterates through the configured RSS feeds, fetches headlines, and then formats and sends the email.
    - A small delay is introduced between feed requests to be considerate to the servers.
    """
    logging.info(f"--- Starting Daily Headlines job for {datetime.now().strftime('%Y-%m-%d')} ---")

    if not validate_config():
        return

    all_headlines = {}

    for category, url in rss_feeds.items():
        logging.info(f"Fetching headlines for: {category}")
        headlines = fetch_and_parse_headlines(url)
        all_headlines[category] = headlines

        # Optional: Introduce a slight delay between requests to avoid hitting the feed server too frequently
        sleep(2)

    subject = f"Daily Headlines: {datetime.now().strftime('%Y-%m-%d')}"
    email_body = format_email_body(all_headlines)
    send_email(subject, email_body, RECEIVER_ADDRESS)

if __name__ == "__main__":
    main()
