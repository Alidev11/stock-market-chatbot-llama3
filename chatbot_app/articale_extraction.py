from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time
from datetime import datetime

def scrape_articles(output_dir="articles"):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_driver_path = r"C:\Users\hp\Desktop\chromedriver_win32\chromedriver.exe"
    os.environ['PATH'] += os.pathsep + chrome_driver_path
    driver = webdriver.Chrome(options=chrome_options)

    # Define URLs for scraping
    urls = [
        "https://www.marketwatch.com/investing/stock/tsla?mod=search_symbol", 
        "https://finance.yahoo.com/quote/TSLA/news?p=TSLA"  # Yahoo Finance
    ]
    
    # List to store articles
    all_articles = []

    for url in urls:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load
        
        if "marketwatch" in url:
            # Scrape from MarketWatch
            try:
                h = driver.find_element(By.CLASS_NAME, 'page--quote')
                home = h.find_element(By.ID, 'maincontent')
                center = home.find_element(By.CLASS_NAME, '.region.region--primary')
                articles_section = center.find_element(By.CSS_SELECTOR, ".j-moreHeadlineWrapper")
                articles = articles_section.find_elements(By.CSS_SELECTOR, ".element--article")
                
                for article in articles:
                    try:
                        timestamp = int(article.get_attribute("data-timestamp"))
                        title = article.find_element(By.CSS_SELECTOR, "a.figure__image").get_attribute("href")
                        link = article.find_element(By.CSS_SELECTOR, "a.figure__image").get_attribute("href")
                        all_articles.append({
                            "title": title,
                            "link": link,
                            "timestamp": datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "MarketWatch"
                        })
                    except Exception as e:
                        print(f"Error processing an article from MarketWatch: {e}")
            except Exception as e:
                        print(f"Error processing an article from Yahoo Finance: {e}")
        elif "yahoo" in url:
            # Scrape from Yahoo Finance
            try:
                articles = driver.find_elements(By.CSS_SELECTOR, "li.js-stream-content")
                
                for article in articles:
                    try:
                        timestamp = int(article.get_attribute("data-time"))
                        title = article.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        link = article.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        all_articles.append({
                            "title": title,
                            "link": link,
                            "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "Yahoo Finance"
                        })
                    except Exception as e:
                        print(f"Error processing an article from Yahoo Finance: {e}")
 
            except Exception as e:
                        print(f"Error processing an article from Yahoo Finance: {e}")        
    # Close the driver
    driver.quit()

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save articles to markdown files
    for article in all_articles:
        date_str = datetime.strptime(article["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        file_path = os.path.join(output_dir, f"{date_str}.md")
        with open(file_path, "a") as file:
            file.write(f"# {article['title']}\n")
            file.write(f"**Source**: {article['source']}\n")
            file.write(f"**Timestamp**: {article['timestamp']}\n")
            file.write(f"[Read full article]({article['link']})\n\n")

    return all_articles


# Example usage: calling the function to scrape articles and save them
scrape_articles(output_dir="data")
