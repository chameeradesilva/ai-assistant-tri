import requests
import json
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base URL for the API
base_url = "https://www.tri.lk/wp-admin/admin-ajax.php"

# List of table IDs to scrape
table_ids = [581, 616, 3397]

# Headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.tri.lk/view-all-publications/",
}

def create_output_dir():
    """Create output directory if it doesn't exist"""
    Path("output/csv").mkdir(parents=True, exist_ok=True)
    Path("output/json").mkdir(parents=True, exist_ok=True)
    Path("output/parquet").mkdir(parents=True, exist_ok=True)
    Path("output/avro").mkdir(parents=True, exist_ok=True)

def scrape_table(table_id):
    """Scrape data for a single table"""
    logger.info(f"Scraping data for table ID: {table_id}...")
    
    params = {
        "action": "wp_ajax_ninja_tables_public_action",
        "table_id": table_id,
        "target_action": "get-all-data",
        "default_sorting": "old_first",
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching data for table {table_id}: {str(e)}")
        return None

def process_table_data(data, table_id):
    """Process raw table data into structured format"""
    processed_entries = []
    
    if not data:
        return processed_entries
    
    for item in data:
        try:
            value = item.get("value", {})
            # Some tables might have different HTML structure
            html_content = value.get("title", value.get("content", ""))
            if not html_content:
                logger.warning(f"No HTML content found for item in table {table_id}")
                continue
                
            soup = BeautifulSoup(html_content, "html.parser")
            a_tag = soup.find("a")
            
            if a_tag:
                entry = {
                    "table_id": table_id,
                    "id": value.get("___id___", ""),
                    "circular_no": value.get("circular_no", ""),
                    "title": a_tag.text.strip(),
                    "pdf_url": f"https://www.tri.lk{a_tag['href']}" if a_tag['href'].startswith('/') else a_tag['href'],
                    "issued_in": value.get("issued_in", ""),
                }
                processed_entries.append(entry)
            else:
                logger.warning(f"No link found in HTML content for item in table {table_id}")
        except Exception as e:
            logger.error(f"Error processing item in table {table_id}: {str(e)}")
            continue
    
    return processed_entries

def save_to_json(data, filename):
    """Save data to JSON format"""
    with open(f"output/json/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_to_csv(data, filename):
    """Save data to CSV format"""
    df = pd.DataFrame(data)
    df.to_csv(f"output/csv/{filename}.csv", index=False, encoding='utf-8')

def save_to_parquet(data, filename):
    """Save data to Parquet format"""
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"output/parquet/{filename}.parquet")

def save_to_avro(data, filename):
    """Save data to Avro format"""
    schema = {
        'name': 'Publication',
        'type': 'record',
        'fields': [
            {'name': 'table_id', 'type': 'int'},
            {'name': 'id', 'type': 'string'},
            {'name': 'circular_no', 'type': 'string'},
            {'name': 'title', 'type': 'string'},
            {'name': 'pdf_url', 'type': 'string'},
            {'name': 'issued_in', 'type': 'string'},
        ]
    }
    
    with open(f"output/avro/{filename}.avro", 'wb') as f:
        fastavro.writer(f, schema, data)

def main():
    create_output_dir()
    all_results = []
    
    for table_id in tqdm(table_ids, desc="Processing tables"):
        # Scrape data
        raw_data = scrape_table(table_id)
        if raw_data:
            # Process data
            processed_data = process_table_data(raw_data, table_id)
            all_results.extend(processed_data)
            
            # Save individual table data
            filename = f"tri_publications_table_{table_id}"
            save_to_json(processed_data, filename)
            save_to_csv(processed_data, filename)
            save_to_parquet(processed_data, filename)
            save_to_avro(processed_data, filename)
            
            logger.info(f"Saved data for table {table_id} in all formats")
    
    # Save combined data
    if all_results:
        filename = "tri_publications_all_tables"
        save_to_json(all_results, filename)
        save_to_csv(all_results, filename)
        save_to_parquet(all_results, filename)
        save_to_avro(all_results, filename)
        logger.info("Saved combined data in all formats")
    
    logger.info("Scraping completed successfully!")

if __name__ == "__main__":
    main()