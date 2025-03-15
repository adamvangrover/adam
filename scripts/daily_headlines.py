import requests
import feedparser
import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os
from time import sleep
import time

# --- Developer Notes ---
# This script fetches headlines from various RSS feeds and sends them in a daily email.
# Key improvements include enhanced error handling, modularity, and clear documentation.
#
# To use:
# 1. Set environment variables for email configuration.
# 2. Add more RSS feeds if needed.
# 3. Schedule this script to run daily using cron or Task Scheduler.

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define your RSS feed URLs here ---
rss_feeds = {
    # Finance
    "Finance - Bloomberg": "https://www.bloomberg.com/feeds/markets.rss",
    "Finance - Wall Street Journal": "https://www.wsj.com/xml/rss/3_7085.xml", 
    "Markets - Financial Times": "https://ft.com/rss/markets",
    
    # AI & Technology
    "AI & Technology - TechCrunch": "https://techcrunch.com/feed/",
    "AI & Technology - The Verge": "https://www.theverge.com/rss/index.xml",
    "AI & Technology - Wired": "https://www.wired.com/feed/rss/",
    "AI & Technology - MIT Technology Review": "https://www.technologyreview.com/feed/",

    # Software
    "Software - Hacker News": "https://news.ycombinator.com/rss",
    "Software - GitHub": "https://github.blog/feed/",
    
    # Healthcare
    "Healthcare - Reuters": "https://www.reuters.com/rss/newsPackage/health-news",
    "Healthcare - STAT News": "https://www.statnews.com/feed/",
    "Healthcare - Kaiser Health News": "https://khn.org/feed/",
    
    # Politics
    "Politics - Associated Press": "https://rss.app/feeds/91QDwogUu6CMzRrE.xml", # Example AP Politics feed
    "Politics - BBC News": "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "Politics - The Guardian": "https://www.theguardian.com/world/politics/rss",
    
    # Commodities
    "Commodities - Reuters": "https://www.reuters.com/rss/newsPackage/commoditiesNews/",
    "Commodities - Bloomberg": "https://www.bloomberg.com/feeds/commodities.rss",
    
    # Private Credit
    "Private Credit - Private Debt Investor": "https://www.privatedebtinvestor.com/feed/",

    # M&A
    "M&A - Mergers & Acquisitions": "https://www.themiddlemarket.com/feed", 
    "M&A - PitchBook": "https://pitchbook.com/news/rss",
}

# --- Email Configuration ---
EMAIL_ADDRESS = os.getenv("HEADLINE_EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("HEADLINE_EMAIL_PASSWORD")
RECEIVER_ADDRESS = os.getenv("HEADLINE_RECEIVER_ADDRESS", EMAIL_ADDRESS)  

# --- Constants ---
REQUEST_TIMEOUT_SECONDS = 15
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

# --- Helper functions ---
def fetch_and_parse_headlines(feed_url):
    """
    Fetches and parses headlines from an RSS feed with retry logic and exponential backoff.
    """
    max_retries = 3
    delay = 1  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Fetching feed from: {feed_url}")
            response = requests.get(feed_url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            headlines = [(entry.title, entry.link) for entry in feed.entries]
            logging.info(f"Successfully fetched {len(headlines)} headlines from {feed_url}")
            return headlines
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching feed from {feed_url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.error(f"Max retries reached for {feed_url}.")
                return []
        except Exception as e:
            logging.error(f"Error parsing feed from {feed_url}: {e}")
            return []

def format_email_body(all_headlines):
    """
    Formats the headlines into an HTML email body.
    """
    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; margin: 20px; }}
          h2 {{ color: #0056b3; }}
          h3 {{ color: #333; }}
          ul {{ list-style-type: none; padding: 0; }}
          li {{ margin-bottom: 0.5em; }}
          a {{ text-decoration: none; font-weight: bold; color: #007bff; }}
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
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logging.error("Error: Email sender address and password not configured.")
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
    Validates the presence of required environment variables.
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logging.critical("Email address and password must be set.")
        return False
    if not rss_feeds:
        logging.critical("No RSS feeds are configured.")
        return False
    return True

def main():
    """
    Main function orchestrates the entire process.
    """
    logging.info(f"--- Starting Daily Headlines job for {datetime.now().strftime('%Y-%m-%d')} ---")

    if not validate_config():
        return

    all_headlines = {}

    for category, url in rss_feeds.items():
        logging.info(f"Fetching headlines for: {category}")
        headlines = fetch_and_parse_headlines(url)
        all_headlines[category] = headlines

        sleep(2)  # Respectful delay between requests

    subject = f"Daily Headlines: {datetime.now().strftime('%Y-%m-%d')}"
    email_body = format_email_body(all_headlines)
    send_email(subject, email_body, RECEIVER_ADDRESS)

if __name__ == "__main__":
    main()
