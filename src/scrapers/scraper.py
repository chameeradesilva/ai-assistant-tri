import requests
import json
from pathlib import Path
import logging
import logging.config
import yaml
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
import os

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Setup logging configuration from YAML file"""
    try:
        if os.path.exists(default_path):
            with open(default_path, 'rt') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
            logging.warning(f"Logging configuration file not found at {default_path}. Using basic configuration.")
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.error(f"Error loading logging configuration: {str(e)}")

# Initialize logging configuration
setup_logging()

# Get logger for this module
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
    logger.debug("Creating output directories...")
    Path("output/csv").mkdir(parents=True, exist_ok=True)
    Path("output/json").mkdir(parents=True, exist_ok=True)
    Path("output/parquet").mkdir(parents=True, exist_ok=True)
    Path("output/avro").mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created successfully")

def scrape_table(table_id):
    """Scrape data for a single table"""
    logger.info(f"Starting scraping for table ID: {table_id}")
    logger.debug(f"Using base URL: {base_url}")
    
    params = {
        "action": "wp_ajax_ninja_tables_public_action",
        "table_id": table_id,
        "target_action": "get-all-data",
        "default_sorting": "old_first",
    }
    logger.debug(f"Request parameters: {params}")
    
    try:
        logger.debug(f"Sending GET request to {base_url}")
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully retrieved data for table {table_id} ({len(data)} entries)")
        return data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data for table {table_id}", exc_info=True)
        return None

def process_table_data(data, table_id):
    """Process raw table data into structured format"""
    logger.info(f"Processing data for table {table_id}")
    processed_entries = []
    
    if not data:
        logger.warning(f"No data to process for table {table_id}")
        return processed_entries
    
    logger.debug(f"Processing {len(data)} entries from table {table_id}")
    for item in data:
        try:
            value = item.get("value", {})
            html_content = value.get("title", value.get("content", ""))
            
            if not html_content:
                logger.warning(f"Empty HTML content found in table {table_id}")
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
                logger.debug(f"Processed entry: {entry['id']} - {entry['title'][:50]}...")
                processed_entries.append(entry)
            else:
                logger.warning(f"No link found in HTML content for item in table {table_id}")
        except Exception as e:
            logger.error(f"Error processing item in table {table_id}", exc_info=True)
            continue
    
    logger.info(f"Successfully processed {len(processed_entries)} entries from table {table_id}")
    return processed_entries

def save_to_json(data, filename):
    """Save data to JSON format"""
    logger.debug(f"Saving {len(data)} entries to JSON: {filename}")
    output_path = f"output/json/{filename}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved JSON file: {output_path}")

def save_to_csv(data, filename):
    """Save data to CSV format"""
    logger.debug(f"Saving {len(data)} entries to CSV: {filename}")
    output_path = f"output/csv/{filename}.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Successfully saved CSV file: {output_path}")

def save_to_parquet(data, filename):
    """Save data to Parquet format"""
    logger.debug(f"Saving {len(data)} entries to Parquet: {filename}")
    output_path = f"output/parquet/{filename}.parquet"
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    logger.info(f"Successfully saved Parquet file: {output_path}")

def save_to_avro(data, filename):
    """Save data to Avro format"""
    logger.debug(f"Saving {len(data)} entries to Avro: {filename}")
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
    
    output_path = f"output/avro/{filename}.avro"
    with open(output_path, 'wb') as f:
        fastavro.writer(f, schema, data)
    logger.info(f"Successfully saved Avro file: {output_path}")

def main():
    logger.info("=== Starting TRI Publications Scraper ===")
    create_output_dir()
    all_results = []
    
    logger.info(f"Processing {len(table_ids)} tables: {table_ids}")
    for table_id in tqdm(table_ids, desc="Processing tables"):
        logger.info(f"\nProcessing table {table_id}")
        # Scrape data
        raw_data = scrape_table(table_id)
        if raw_data:
            # Process data
            processed_data = process_table_data(raw_data, table_id)
            all_results.extend(processed_data)
            
            # Save individual table data
            filename = f"tri_publications_table_{table_id}"
            logger.info(f"Saving data for table {table_id} in all formats")
            save_to_json(processed_data, filename)
            save_to_csv(processed_data, filename)
            save_to_parquet(processed_data, filename)
            save_to_avro(processed_data, filename)
            
            logger.info(f"Successfully saved all formats for table {table_id}")
    
    # Save combined data
    if all_results:
        filename = "tri_publications_all_tables"
        logger.info(f"Saving combined data ({len(all_results)} entries) in all formats")
        save_to_json(all_results, filename)
        save_to_csv(all_results, filename)
        save_to_parquet(all_results, filename)
        save_to_avro(all_results, filename)
        logger.info("Successfully saved combined data in all formats")
    
    logger.info("=== Scraping completed successfully! ===")

if __name__ == "__main__":
    main()