import os
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_downloader')

def download_file(url, destination):
    os.makedirs(Path(destination).parent, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        try:
            # Get local file size
            local_size = os.path.getsize(destination)
            filename = os.path.basename(destination)
            
            # If the file exists and has a reasonable size, assume it's valid
            # This avoids issues when container restarts with expired pre-signed URLs
            if local_size > 0:
                # Skip size validation against S3 URL if file exists with non-zero size
                logger.info(f"Using existing file {filename}: ({local_size/(1024*1024):.1f} MB)")
                return destination
                
            # Only try remote validation for zero-size files
            if local_size == 0:
                logger.info(f"File {filename} exists but has zero size - will redownload")
        except Exception as e:
            filename = os.path.basename(destination)
            logger.warning(f"Error checking existing file {filename}, will attempt to redownload: {str(e)}")
    
    # Setup session with retries
    session = requests.Session()
    retries = Retry(total=5, 
                   backoff_factor=1, 
                   status_forcelist=[500, 502, 503, 504, 520, 524, 429],
                   allowed_methods=["GET"])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        filename = os.path.basename(destination)
        logger.info(f"Starting download: {filename}")
        start_time = time.time()
        
        # Use timeout to prevent hanging requests (30s connect, 10min read)
        response = session.get(url, stream=True, timeout=(30, 600))
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        last_log_time = start_time
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 30 seconds for large files
                    current_time = time.time()
                    if current_time - last_log_time > 30:
                        elapsed = current_time - start_time
                        speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        logger.info(f"Progress for {filename}: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB of {total_size/(1024*1024):.1f} MB) at {speed:.2f} MB/s")
                        last_log_time = current_time
        
        elapsed = time.time() - start_time
        speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
        logger.info(f"Completed {filename}: {downloaded/(1024*1024):.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")
        
        # Verify file size if content-length was provided
        if total_size > 0 and downloaded != total_size:
            logger.warning(f"Size mismatch for {filename}: expected {total_size} bytes, got {downloaded} bytes")
            
        return destination
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error downloading {filename}: {str(e)}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout downloading {filename}: {str(e)}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading {filename}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}", exc_info=True)
        return None

def main():
    download_dir = os.environ.get('DOWNLOAD_MODEL_DIR', '/workspace/model')
    os.makedirs(download_dir, exist_ok=True)
    
    files_count = int(os.environ.get('MODEL_FILES_COUNT', '0'))
    max_workers = int(os.environ.get('DOWNLOAD_MAX_WORKERS', '32'))
    
    if files_count == 0:
        logger.error("No files to download (MODEL_FILES_COUNT is 0 or not set)")
        return 1
    
    logger.info(f"Starting parallel download of {files_count} files with {max_workers} workers...")
    logger.info(f"Using 1MB chunk size optimized for large files (4-5GB)")
    logger.info(f"Files that already exist with reasonable size will be used without redownloading")
    
    # Collect download tasks
    downloads = []
    total_size_estimate = 0
    for i in range(files_count):
        filename = os.environ.get(f'MODEL_FILE_{i}_NAME')
        url = os.environ.get(f'MODEL_FILE_{i}_URL')
        if not filename or not url:
            logger.warning(f"Missing filename or URL for index {i}")
            continue
        downloads.append((url, os.path.join(download_dir, filename)))
    
    start_time = time.time()
    success_count = 0
    failed_files = []
    
    # Process downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, dest): dest for url, dest in downloads}
        
        for future in as_completed(futures):
            dest = futures[future]
            filename = os.path.basename(dest)
            
            try:
                result = future.result()
                if result:
                    success_count += 1
                    if success_count % 5 == 0 or success_count == len(downloads):
                        elapsed = time.time() - start_time
                        logger.info(f"Downloaded {success_count}/{len(downloads)} files in {elapsed:.1f}s")
                else:
                    failed_files.append(filename)
            except Exception as e:
                logger.error(f"Unexpected error for {filename}: {str(e)}", exc_info=True)
                failed_files.append(filename)
    
    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"Download process completed in {elapsed:.1f} seconds")
    
    if success_count == len(downloads):
        logger.info("All files downloaded successfully")
        return 0
    else:
        logger.error(f"Failed to download {len(failed_files)} of {len(downloads)} files")
        if failed_files:
            logger.error(f"Failed files: {', '.join(failed_files[:10])}" + 
                        (f" and {len(failed_files)-10} more" if len(failed_files) > 10 else ""))
        return 1

if __name__ == "__main__":
    sys.exit(main()) 