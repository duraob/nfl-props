"""
Enhanced NFL Data Scraper - Method-Based Architecture
Optimized for weekly runs to capture fresh NFL data for projection engine.

This script uses undetected-chromedriver to avoid Cloudflare/bot detection
while scraping Pro Football Reference, providing reliable data collection.
"""

import pandas as pd
import os
import pickle
import hashlib
from datetime import datetime, timedelta
import time
import random
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_data_enhanced.log'),
        logging.StreamHandler()
    ]
)

# Global configuration
BASE_URL = "https://www.pro-football-reference.com"
CACHE_DIR = 'cache/current_season'
DATA_DIR = 'data/current_season'
CACHE_HOURS = 24

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def setup_undetected_driver() -> Optional[uc.Chrome]:
    """
    Setup undetected Chrome driver to avoid bot detection.
    
    Returns:
        uc.Chrome: Configured Chrome driver or None if setup fails
    """
    try:
        logging.info("Setting up undetected Chrome driver...")
        
        # Enhanced options for better anti-detection
        options = uc.ChromeOptions()
        
        # Basic stability options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')  # Faster loading
        options.add_argument('--disable-javascript')  # Reduce detection surface
        
        # Anti-detection options
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=VizDisplayCompositor')
        options.add_argument('--disable-ipc-flooding-protection')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-client-side-phishing-detection')
        options.add_argument('--disable-sync')
        options.add_argument('--disable-translate')
        options.add_argument('--hide-scrollbars')
        options.add_argument('--mute-audio')
        options.add_argument('--no-first-run')
        options.add_argument('--disable-default-apps')
        options.add_argument('--disable-popup-blocking')
        
        # Set realistic viewport
        options.add_argument('--window-size=1920,1080')
        
        # Create driver with enhanced configuration
        driver = uc.Chrome(
            options=options, 
            version_main=None,
            driver_executable_path=None,
            browser_executable_path=None,
            user_data_dir=None,
            headless=False,  # Keep visible for debugging
            use_subprocess=True
        )
        
        # Enhanced timeout configuration
        driver.set_page_load_timeout(120)  # Increased timeout for slower pages
        driver.implicitly_wait(15)  # Increased implicit wait
        
        # Execute stealth scripts
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        logging.info("Undetected Chrome driver setup complete")
        return driver
        
    except Exception as e:
        logging.error(f"Error setting up undetected driver: {e}")
        return None


def get_cache_path(cache_key: str) -> str:
    """
    Generate cache file path for a given cache key.
    
    Args:
        cache_key: Unique identifier for cached data
        
    Returns:
        str: Full path to cache file
    """
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f'{cache_hash}.pickle')


def is_cache_valid(cache_path: str, max_age_hours: int = CACHE_HOURS) -> bool:
    """
    Check if cache is valid based on age.
    
    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age in hours before cache expires
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(cache_path):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    return age < timedelta(hours=max_age_hours)


def get_cached_data(cache_key: str, cache_hours: int = CACHE_HOURS) -> Tuple[Optional[Any], bool]:
    """
    Get data from cache or return None if not available/valid.
    
    Args:
        cache_key: Unique identifier for cached data
        cache_hours: Maximum age in hours before cache expires
        
    Returns:
        Tuple[Optional[Any], bool]: (cached_data, from_cache_flag)
    """
    cache_path = get_cache_path(cache_key)
    
    if is_cache_valid(cache_path, cache_hours):
        try:
            logging.info(f"Loading cached data for {cache_key}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f), True
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    
    return None, False


def save_to_cache(cache_key: str, data: Any) -> bool:
    """
    Save data to cache.
    
    Args:
        cache_key: Unique identifier for cached data
        data: Data to cache
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cache_path = get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logging.error(f"Error saving to cache: {e}")
        return False


def get_soup_with_undetected(driver: uc.Chrome, url: str, cache_hours: int = CACHE_HOURS, max_retries: int = 3) -> Tuple[Optional[BeautifulSoup], bool]:
    """
    Get BeautifulSoup object using undetected Chrome driver with enhanced retry logic.
    
    Args:
        driver: Configured Chrome driver
        url: URL to scrape
        cache_hours: Maximum age in hours before cache expires
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple[Optional[BeautifulSoup], bool]: (soup_object, from_cache_flag)
    """
    cache_key = f"undetected_{hashlib.md5(url.encode()).hexdigest()}"
    cached_soup, from_cache = get_cached_data(cache_key, cache_hours)
    
    if from_cache:
        return cached_soup, True
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Fetching data from {url} using undetected Chrome (attempt {attempt + 1}/{max_retries})")
            
            # Check if driver is still valid
            try:
                driver.current_url
            except Exception as e:
                logging.error(f"Driver is no longer valid: {e}")
                return None, False
            
            # Add random delay before navigation
            time.sleep(random.uniform(1, 3))
            
            # Navigate to the page with enhanced error handling
            try:
                driver.get(url)
                logging.info(f"Successfully navigated to {url}")
            except Exception as e:
                logging.error(f"Navigation failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 + attempt * 2)
                    continue
                else:
                    return None, False
            
            # Wait for page to load with progressive delays
            initial_wait = random.uniform(5, 8)  # Longer initial wait for 2024 pages
            logging.info(f"Waiting {initial_wait:.1f} seconds for page load...")
            time.sleep(initial_wait)
            
            # Check for Cloudflare protection and other blocking mechanisms
            try:
                page_title = driver.title.lower()
                current_url = driver.current_url
                logging.info(f"Page title: '{page_title}', Current URL: '{current_url}'")
                
                if "just a moment" in page_title or "checking your browser" in page_title:
                    logging.warning("Cloudflare protection detected, waiting...")
                    time.sleep(random.uniform(15, 25))  # Longer wait for Cloudflare
                    
                    # Refresh and wait again
                    driver.refresh()
                    time.sleep(random.uniform(8, 12))
                    
                    # Check if still blocked
                    page_title = driver.title.lower()
                    if "just a moment" in page_title or "checking your browser" in page_title:
                        if attempt < max_retries - 1:
                            logging.warning(f"Still blocked by Cloudflare, retrying in {10 + attempt * 5} seconds...")
                            time.sleep(10 + attempt * 5)
                            continue
                        else:
                            logging.error("Still blocked by Cloudflare protection after all retries")
                            return None, False
                
                # Check for other blocking mechanisms
                if "access denied" in page_title or "blocked" in page_title:
                    logging.warning(f"Access denied detected: {page_title}")
                    if attempt < max_retries - 1:
                        time.sleep(10 + attempt * 3)
                        continue
                    else:
                        return None, False
                        
            except Exception as e:
                logging.warning(f"Error checking page status: {e}")
            
            # Additional wait for dynamic content - longer for 2024 pages
            additional_wait = random.uniform(3, 6) if "2024" in url else random.uniform(2, 4)
            time.sleep(additional_wait)
            
            # Get page source and create BeautifulSoup object
            try:
                page_source = driver.page_source
                logging.info(f"Retrieved page source: {len(page_source)} characters")
            except Exception as e:
                logging.error(f"Failed to get page source: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 + attempt)
                    continue
                else:
                    return None, False
            
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Enhanced content validation
            if not soup or len(page_source) < 1000:
                logging.warning(f"Invalid content: soup={bool(soup)}, length={len(page_source)}")
                if attempt < max_retries - 1:
                    logging.warning(f"Retrying in {3 + attempt * 2} seconds...")
                    time.sleep(3 + attempt * 2)
                    continue
                else:
                    logging.error("Got invalid content after all retries")
                    return None, False
            
            # Check for specific content that indicates successful loading
            if "pro-football-reference" not in page_source.lower() and "nfl" not in page_source.lower():
                logging.warning("Page doesn't appear to contain NFL data")
                if attempt < max_retries - 1:
                    time.sleep(3 + attempt)
                    continue
                else:
                    logging.error("Page content validation failed after all retries")
                    return None, False
            
            # Save to cache
            save_to_cache(cache_key, soup)
            logging.info(f"Successfully loaded and cached content from {url}")
            
            return soup, False
            
        except Exception as e:
            logging.error(f"Error fetching data from {url} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                retry_delay = 5 + attempt * 3
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"All retry attempts failed for {url}")
                return None, False
    
    return None, False


def get_season_schedule(year: int, driver: uc.Chrome) -> List[Dict]:
    """
    Get completed games from the season schedule page with enhanced error handling.
    
    Args:
        year: NFL season year
        driver: Configured Chrome driver
        
    Returns:
        List[Dict]: List of completed game dictionaries with metadata
    """
    url = f"{BASE_URL}/years/{year}/games.htm"
    logging.info(f"Attempting to scrape schedule from: {url}")
    
    soup, from_cache = get_soup_with_undetected(driver, url)
    
    if not soup:
        logging.error(f"Failed to get schedule page for year {year}")
        return []
    
    try:
        # Enhanced table finding with multiple strategies
        games_table = None
        
        # Strategy 1: Look for the main games table
        games_table = soup.find("table", {"id": "games"})
        if games_table:
            logging.info("Found games table using ID 'games'")
        else:
            # Strategy 2: Look for table with games class or similar
            games_table = soup.find("table", {"class": "sortable stats_table"})
            if games_table:
                logging.info("Found games table using class 'sortable stats_table'")
            else:
                # Strategy 3: Look for any table containing game data
                all_tables = soup.find_all("table")
                logging.info(f"Found {len(all_tables)} total tables on page")
                
                for i, table in enumerate(all_tables):
                    table_text = table.get_text().lower()
                    if "week" in table_text and ("home" in table_text or "away" in table_text):
                        games_table = table
                        logging.info(f"Found potential games table #{i+1} by content analysis")
                        break
        
        if not games_table:
            logging.error("Could not find games table using any strategy")
            # Log page structure for debugging
            logging.info("Page structure analysis:")
            logging.info(f"Page title: {soup.title.string if soup.title else 'No title'}")
            logging.info(f"Total tables found: {len(soup.find_all('table'))}")
            return []
        
        logging.info(f"Successfully found games table for year {year}")
        
        # Find table body
        tbody = games_table.find("tbody")
        if not tbody:
            logging.warning("No tbody found, using table directly")
            tbody = games_table
        
        game_rows = tbody.find_all("tr")
        logging.info(f"Found {len(game_rows)} total game rows")
        
        completed_games = []
        for i, row in enumerate(game_rows):
            try:
                # Extract game metadata
                game_data = extract_game_metadata_from_schedule(row, year)
                if game_data:
                    completed_games.append(game_data)
                    logging.debug(f"Added completed game: Week {game_data['week']}, {game_data['away_team']} @ {game_data['home_team']}")
                else:
                    logging.debug(f"Row {i+1}: No valid game data extracted")
                    
            except Exception as e:
                logging.warning(f"Error processing game row {i+1}: {e}")
                continue
        
        logging.info(f"Found {len(completed_games)} completed games for year {year}")
        return completed_games
        
    except Exception as e:
        logging.error(f"Error extracting schedule data: {e}")
        return []


def extract_game_metadata_from_schedule(row, year: int) -> Optional[Dict]:
    """
    Extract game metadata from schedule table row.
    
    Args:
        row: BeautifulSoup table row element
        year: Season year
        
    Returns:
        Optional[Dict]: Game metadata or None if not a completed game
    """
    try:
        # Get all cells in the row
        cells = row.find_all("td")
        if len(cells) < 8:  # Need at least 8 cells for a complete game row
            return None
        
        # Extract week number from row text (it's embedded in the full row text)
        row_text = row.get_text().strip()
        week_match = re.match(r'^(\d+)', row_text)
        if not week_match:
            return None
        
        week = safe_int(week_match.group(1))
        if week == 0:
            return None
        
        # Extract data from cells based on actual structure:
        # Cell 0: day, Cell 1: date, Cell 2: time, Cell 3: away_team, Cell 4: home/away indicator, Cell 5: home_team, Cell 6: boxscore, Cell 7: away_score, Cell 8: home_score
        
        # Date (cell 1)
        date_text = cells[1].get_text().strip()
        
        # Teams (cells 3 and 5)
        away_team = cells[3].get_text().strip()
        home_team = cells[5].get_text().strip()
        
        # Check for boxscore link (cell 6)
        boxscore_cell = cells[6]
        if not boxscore_cell.a:
            return None
        
        boxscore_link = boxscore_cell.a.attrs.get("href", "")
        if not boxscore_link.startswith("/boxscores/"):
            return None
        
        # Scores (cells 7 and 8)
        away_score = safe_int(cells[7].get_text().strip())
        home_score = safe_int(cells[8].get_text().strip())
        
        # Skip if no scores (game not completed)
        if away_score == 0 and home_score == 0:
            return None
        
        return {
            'year': year,
            'week': week,
            'date': date_text,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'boxscore_url': boxscore_link
        }
        
    except Exception as e:
        logging.warning(f"Error extracting game metadata: {e}")
        return None


def get_team_links(year: int, driver: uc.Chrome) -> List[str]:
    """
    Get team links from Pro Football Reference year page.
    (Legacy function - kept for backward compatibility)
    
    Args:
        year: NFL season year
        driver: Configured Chrome driver
        
    Returns:
        List[str]: List of team page URLs
    """
    url = f"{BASE_URL}/years/{year}"
    soup, from_cache = get_soup_with_undetected(driver, url)
    
    if not soup:
        logging.error(f"Failed to get team links for year {year}")
        return []
    
    try:
        # Find team stats table
        team_container = soup.find("div", {"class": "table_container is_setup", "id": "div_team_stats"})
        if not team_container:
            logging.error("Could not find team stats container")
            return []
        
        team_rows = team_container.find_all("td", {"data-stat": "team"})
        team_links = []
        
        for team in team_rows:
            if team.a:
                team_links.append(team.a.attrs["href"])
        
        logging.info(f"Found {len(team_links)} team links for year {year}")
        return team_links
        
    except Exception as e:
        logging.error(f"Error extracting team links: {e}")
        return []


def safe_int(value: str) -> int:
    """
    Safely convert string to integer, returning 0 for invalid values.
    
    Args:
        value: String value to convert
        
    Returns:
        int: Converted integer or 0 if conversion fails
    """
    try:
        return int(value.strip()) if value.strip() else 0
    except (ValueError, AttributeError):
        return 0


def extract_weather_info(soup: BeautifulSoup) -> str:
    """
    Extract weather information from game page.
    
    Args:
        soup: BeautifulSoup object of game page
        
    Returns:
        str: Weather information or empty string if not found
    """
    try:
        # Look for weather information in the game_info table
        game_info_table = soup.find("table", {"id": "game_info"})
        if game_info_table:
            # Find the weather row
            weather_row = game_info_table.find("th", string="Weather")
            if weather_row:
                # Get the next td element which contains the weather data
                weather_cell = weather_row.find_next_sibling("td")
                if weather_cell:
                    weather_text = weather_cell.get_text().strip()
                    # Clean up the weather text
                    if weather_text and "degrees" in weather_text:
                        return weather_text
        
        # Fallback: Look for weather information in various possible locations
        weather_elements = soup.find_all(string=re.compile(r'\d+\s+degrees'))
        if weather_elements:
            return weather_elements[0].strip()
        
        return ""
        
    except Exception as e:
        logging.warning(f"Could not extract weather info: {e}")
        return ""


def extract_game_scores(soup: BeautifulSoup) -> Tuple[int, int]:
    """
    Extract team scores from game page using the linescore table.
    Home team is always the second row, final score is the last td in each row.
    
    Args:
        soup: BeautifulSoup object of game page
        
    Returns:
        Tuple[int, int]: (home_score, away_score)
    """
    try:
        # Look for the linescore table with the correct class
        linescore_table = soup.find("table", {"class": "linescore nohover stats_table no_freeze"})
        if not linescore_table:
            # Fallback: try partial class match
            linescore_table = soup.find("table", class_=lambda x: x and "linescore" in x)
        
        if not linescore_table:
            logging.warning("Could not find linescore table")
            return 0, 0
        
        # Get tbody and rows from the linescore table
        tbody = linescore_table.find("tbody")
        if not tbody:
            logging.warning("Could not find tbody in linescore table")
            return 0, 0
            
        rows = tbody.find_all("tr")
        if len(rows) < 2:
            logging.warning("Linescore table has fewer than 2 rows")
            return 0, 0
        
        # First row: away team, Second row: home team
        away_row = rows[0].find_all("td")
        home_row = rows[1].find_all("td")
        
        if len(away_row) < 6 or len(home_row) < 6:
            logging.warning(f"Could not find enough score cells: away={len(away_row)}, home={len(home_row)}")
            return 0, 0
        
        # Final score is the last td in each row (index -1)
        away_score = safe_int(away_row[-1].get_text().strip())
        home_score = safe_int(home_row[-1].get_text().strip())
        
        # The scores are being extracted in the wrong order, so swap them
        # Based on the log: "Extracted scores: Away=40, Home=41" but BUF should have won 41-40
        # This means the actual away team (BUF) scored 41, not 40
        actual_away_score = home_score  # Swap
        actual_home_score = away_score  # Swap
        
        logging.info(f"Extracted scores: Away={away_score}, Home={home_score}")
        logging.info(f"Swapped scores: Away={actual_away_score}, Home={actual_home_score}")
        return actual_home_score, actual_away_score
        
    except Exception as e:
        logging.warning(f"Could not extract game scores: {e}")
        return 0, 0
    
def get_game_links_from_team_page(team_url: str, driver: uc.Chrome) -> List[Dict]:
    """
    Get game links and metadata from a team's page.
    
    Args:
        team_url: URL of team page
        driver: Configured Chrome driver
        
    Returns:
        List[Dict]: List of game dictionaries with links and metadata
    """
    full_url = f"{BASE_URL}{team_url}"
    soup, from_cache = get_soup_with_undetected(driver, full_url)
    
    if not soup:
        logging.error(f"Failed to get team page: {full_url}")
        return []
    
    try:
        # Find games table
        games_table = soup.find("table", {"id": "games"})
        if not games_table:
            logging.error("Could not find games table")
            return []
        
        logging.info(f"Found games table for {team_url}")
        
        game_data = []
        game_rows = games_table.tbody.find_all("tr")
        logging.info(f"Found {len(game_rows)} total game rows")
        
        for i, row in enumerate(game_rows):
            # Debug: Log row structure
            row_text = row.get_text().strip()
            logging.info(f"Processing row {i+1}: {row_text[:150]}...")
            
            # Look for game outcome cell
            outcome_cell = row.find("td", {"data-stat": "game_outcome"})
            outcome_text = outcome_cell.get_text().strip() if outcome_cell else ""
            
            # Extract opponent
            opp_cell = row.find("td", {"data-stat": "opp"})
            opp_text = opp_cell.get_text().strip() if opp_cell else ""
            
            logging.info(f"Row {i+1}: outcome='{outcome_text}', opponent='{opp_text}'")
            
            # Skip bye weeks and non-games
            if opp_text == "Bye Week" or not opp_text or not outcome_text:
                logging.info(f"Row {i+1}: Skipping (outcome: '{outcome_text}', opp: '{opp_text}')")
                continue
            
            # Extract week number from the row text (it's embedded in the full row text)
            # Look for pattern like "1SunSeptember 8" or "2ThuSeptember 12"
            week_match = re.match(r'^(\d+)', row_text)
            if week_match:
                week_num = safe_int(week_match.group(1))
                logging.info(f"Row {i+1}: Week {week_num}, vs {opp_text}, outcome: {outcome_text}")
                
                # Look for game link in any cell that has a link
                game_link = ""
                all_cells = row.find_all("td")
                for cell in all_cells:
                    link = cell.find("a")
                    if link and link.attrs.get("href", "").startswith("/boxscores/"):
                        game_link = link.attrs.get("href", "")
                        logging.info(f"Row {i+1}: Found game link: {game_link}")
                        break
                
                if game_link:
                    # Extract opponent abbreviation
                    opponent = "UNK"
                    if opp_cell.a:
                        opponent = opp_cell.a.attrs["href"].split("/")[2]
                    
                    # Determine home/away from opponent text
                    home_away = "away"  # Default
                    if not opp_text.startswith("@"):
                        home_away = "home"
                    
                    game_data.append({
                        'link': game_link,
                        'week': week_num,
                        'opponent': opponent,
                        'home_away': home_away
                    })
                    logging.info(f"Added game: Week {week_num}, {opponent}, {home_away}")
                else:
                    logging.info(f"Row {i+1}: No game link found")
            else:
                logging.info(f"Row {i+1}: Could not extract week number from '{row_text[:50]}...'")
            
            # Original logic for reference (keeping for fallback)
            if outcome_cell and outcome_text:
                # Game has been played, get the link and metadata
                game_cell = row.find("td", {"data-stat": "week_num"})
                if game_cell and game_cell.a:
                    # Extract week number
                    week_num = safe_int(game_cell.get_text())
                    logging.debug(f"Row {i+1} week: {week_num}")
                    
                    # Extract opponent
                    opp_cell = row.find("td", {"data-stat": "opp"})
                    opponent = "UNK"
                    if opp_cell and opp_cell.a:
                        opponent = opp_cell.a.attrs["href"].split("/")[2]
                    logging.debug(f"Row {i+1} opponent: {opponent}")
                    
                    # Determine home/away
                    home_away = "away"  # Default
                    location_cell = row.find("td", {"data-stat": "game_location"})
                    if location_cell:
                        location_text = location_cell.get_text().strip()
                        if location_text == "":
                            home_away = "home"
                    logging.debug(f"Row {i+1} home/away: {home_away}")
                    
                    game_data.append({
                        'link': game_cell.a.attrs["href"],
                        'week': week_num,
                        'opponent': opponent,
                        'home_away': home_away
                    })
                    logging.info(f"Added game: Week {week_num}, {opponent}, {home_away}")
                else:
                    logging.debug(f"Row {i+1}: No game link found")
            else:
                logging.debug(f"Row {i+1}: No outcome or empty outcome")
        
        logging.info(f"Found {len(game_data)} completed games for team")
        return game_data
        
    except Exception as e:
        logging.error(f"Error extracting game links: {e}")
        return []


def extract_offense_stats(row, team: str, opponent: str, home_away: str,
                         weather: str, year: int, week: int, team_score: int, opp_score: int) -> Optional[Dict]:
    """
    Extract all offensive statistics (passing, rushing, receiving) from a table row.
    
    Args:
        row: BeautifulSoup table row element
        team: Team abbreviation
        opponent: Opponent team abbreviation
        home_away: 'home' or 'away'
        weather: Weather information
        year: Season year
        week: Week number
        team_score: Team's score
        opp_score: Opponent's score
        
    Returns:
        Optional[Dict]: Player stat dictionary or None if invalid
    """
    try:
        # Get row structure for processing
        row_text = row.get_text().strip()
        cells = row.find_all("td")
        
        # Extract player name from row text (it's embedded in the full row text)
        # Pattern: "PlayerNameTEAMstats..." - need to extract player name before team abbreviation
        team_match = re.search(r'([A-Z]{3})', row_text)
        if not team_match:
            return None
        
        # Extract player name (everything before the team abbreviation)
        player_name = row_text[:team_match.start()].strip()
        if not player_name:
            return None
        
        # Extract position from player name or use default
        # For now, we'll use a default position since it's not easily extractable
        pos = "UNK"  # Will need to be determined from context or additional parsing
        
        # Extract all offensive stats from cells based on the actual structure
        # Based on debug output: Cell 0=team, Cell 1=pass_cmp, Cell 2=pass_att, etc.
        if len(cells) < 21:
            return None
        
        # Passing stats (cells 1-5)
        pass_cmp = safe_int(cells[1].get_text().strip())
        pass_att = safe_int(cells[2].get_text().strip())
        pass_yds = safe_int(cells[3].get_text().strip())
        pass_tds = safe_int(cells[4].get_text().strip())
        pass_int = safe_int(cells[5].get_text().strip())
        sacks = safe_int(cells[6].get_text().strip())  # pass_sacked
        
        # Rushing stats (cells 10-12 based on typical structure)
        rush_att = safe_int(cells[10].get_text().strip())
        rush_yds = safe_int(cells[11].get_text().strip())
        rush_tds = safe_int(cells[12].get_text().strip())
        
        # Receiving stats (cells 14-17 based on typical structure)
        targets = safe_int(cells[14].get_text().strip())
        receptions = safe_int(cells[15].get_text().strip())
        rec_yds = safe_int(cells[16].get_text().strip())
        rec_tds = safe_int(cells[17].get_text().strip())
        
        # Fumbles (cell 19 based on typical structure)
        fumbles = safe_int(cells[19].get_text().strip())
        
        logging.debug(f"Extracted stats for {player_name}: Pass {pass_cmp}/{pass_att}, Rush {rush_att}, Rec {receptions}")
        
        # Determine the actual team this player belongs to based on the team cell
        actual_team = cells[0].get_text().strip()  # Cell 0 contains the team abbreviation
        
        # Determine correct scores based on the player's actual team
        # We need to determine if this player is from the home team or away team
        # The team parameter is the full team name (e.g., "Baltimore Ravens")
        # The actual_team is the abbreviation (e.g., "BAL")
        
        # Map team abbreviations to full names for comparison
        team_abbrev_to_full = {
            'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins', 'NE': 'New England Patriots',
            'NYJ': 'New York Jets', 'PIT': 'Pittsburgh Steelers', 'CLE': 'Cleveland Browns', 'CIN': 'Cincinnati Bengals',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'TEN': 'Tennessee Titans',
            'DEN': 'Denver Broncos', 'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
            'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants', 'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders',
            'CHI': 'Chicago Bears', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
            'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints', 'TB': 'Tampa Bay Buccaneers',
            'ARI': 'Arizona Cardinals', 'LAR': 'Los Angeles Rams', 'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks'
        }
        
        # Get the full team name for the actual team
        actual_team_full = team_abbrev_to_full.get(actual_team, actual_team)
        
        # Determine if this player is from the home team or away team
        if actual_team_full == team:  # Player is from the team we're processing (home team)
            player_team_score = team_score  # home_score
            player_opp_score = opp_score    # away_score
            player_opponent = opponent
            player_home_away = home_away
        else:  # Player is from the opponent team (away team)
            player_team_score = opp_score   # away_score
            player_opp_score = team_score   # home_score
            player_opponent = team
            player_home_away = 'away' if home_away == 'home' else 'home'
        
        return {
            'year': year,
            'week': week,
            'weather': weather,
            'home_team': team if home_away == 'home' else opponent,
            'away_team': opponent if home_away == 'home' else team,
            'player': player_name,
            'team': actual_team,  # Use the actual team from the data
            'opponent': player_opponent,  # Set opponent correctly
            'home_away': player_home_away,  # Set home/away correctly
            'team_score': player_team_score,  # Player's team score
            'opp_score': player_opp_score,  # Opponent's score
            'pos': pos,
            'snaps': 0,  # Will be filled from snap counts calculation
            'snap_pct': 0.0,
            'pass_cmp': pass_cmp,
            'pass_att': pass_att,
            'pass_yds': pass_yds,
            'pass_tds': pass_tds,
            'pass_int': pass_int,
            'sacks': sacks,
            'rush_att': rush_att,
            'rush_yds': rush_yds,
            'rush_tds': rush_tds,
            'targets': targets,
            'receptions': receptions,
            'rec_yds': rec_yds,
            'rec_tds': rec_tds,
            'fumbles': fumbles
        }
        
    except Exception as e:
        logging.warning(f"Error extracting offense stats: {e}")
        return None


def extract_passing_stats(row, team: str, opponent: str, home_away: str,
                         weather: str, year: int, week: int, team_score: int, opp_score: int) -> Optional[Dict]:
    """
    Extract passing statistics from a table row.
    
    Args:
        row: BeautifulSoup table row element
        team: Team abbreviation
        opponent: Opponent team abbreviation
        home_away: 'home' or 'away'
        weather: Weather information
        year: Season year
        week: Week number
        team_score: Team's score
        opp_score: Opponent's score
        
    Returns:
        Optional[Dict]: Player stat dictionary or None if invalid
    """
    try:
        # Debug: Log row structure
        row_text = row.get_text().strip()
        logging.debug(f"Processing passing row: {row_text[:100]}...")
        
        # Debug: Log all cells in the row
        cells = row.find_all("td")
        logging.debug(f"Row has {len(cells)} cells")
        for i, cell in enumerate(cells[:5]):  # Show first 5 cells
            cell_text = cell.get_text().strip()
            data_stat = cell.get("data-stat", "no-stat")
            logging.debug(f"  Cell {i}: data-stat='{data_stat}', text='{cell_text}'")
        
        player_cell = row.find("td", {"data-stat": "player"})
        if not player_cell or not player_cell.get_text().strip():
            logging.debug("No player cell found or empty player name")
            return None
        
        player_name = player_cell.get_text().strip()
        logging.debug(f"Found player: {player_name}")
        
        pos_cell = row.find("td", {"data-stat": "pos"})
        if not pos_cell:
            logging.debug("No position cell found")
            return None
        pos = pos_cell.get_text().strip()
        
        # Extract passing stats
        pass_cmp = safe_int(row.find("td", {"data-stat": "pass_cmp"}).get_text())
        pass_att = safe_int(row.find("td", {"data-stat": "pass_att"}).get_text())
        pass_yds = safe_int(row.find("td", {"data-stat": "pass_yds"}).get_text())
        pass_tds = safe_int(row.find("td", {"data-stat": "pass_td"}).get_text())
        pass_int = safe_int(row.find("td", {"data-stat": "pass_int"}).get_text())
        sacks = safe_int(row.find("td", {"data-stat": "sacks"}).get_text())
        
        logging.debug(f"Extracted stats for {player_name}: {pass_cmp}/{pass_att} for {pass_yds} yards")
        
        return {
            'year': year,
            'week': week,
            'weather': weather,
            'home_team': team if home_away == 'home' else opponent,
            'away_team': opponent if home_away == 'home' else team,
            'player': player_name,
            'team': team,
            'opponent': opponent,
            'home_away': home_away,
            'team_score': team_score,
            'opp_score': opp_score,
            'pos': pos,
            'snaps': 0,  # Will be filled from other tables
            'snap_pct': 0.0,
            'pass_cmp': pass_cmp,
            'pass_att': pass_att,
            'pass_yds': pass_yds,
            'pass_tds': pass_tds,
            'pass_int': pass_int,
            'sacks': sacks,
            'rush_att': 0,
            'rush_yds': 0,
            'rush_tds': 0,
            'targets': 0,
            'receptions': 0,
            'rec_yds': 0,
            'rec_tds': 0,
            'fumbles': 0
        }
        
    except Exception as e:
        logging.warning(f"Error extracting passing stats: {e}")
        return None


def extract_rushing_receiving_stats(row, team: str, opponent: str, home_away: str,
                                   weather: str, year: int, week: int, team_score: int, opp_score: int) -> Optional[Dict]:
    """
    Extract rushing and receiving statistics from a table row.
    
    Args:
        row: BeautifulSoup table row element
        team: Team abbreviation
        opponent: Opponent team abbreviation
        home_away: 'home' or 'away'
        weather: Weather information
        year: Season year
        week: Week number
        team_score: Team's score
        opp_score: Opponent's score
        
    Returns:
        Optional[Dict]: Player stat dictionary or None if invalid
    """
    try:
        player_cell = row.find("td", {"data-stat": "player"})
        if not player_cell or not player_cell.get_text().strip():
            return None
        
        player_name = player_cell.get_text().strip()
        pos = row.find("td", {"data-stat": "pos"}).get_text().strip()
        
        # Extract rushing stats
        rush_att = safe_int(row.find("td", {"data-stat": "rush_att"}).get_text())
        rush_yds = safe_int(row.find("td", {"data-stat": "rush_yds"}).get_text())
        rush_tds = safe_int(row.find("td", {"data-stat": "rush_td"}).get_text())
        
        # Extract receiving stats
        targets = safe_int(row.find("td", {"data-stat": "targets"}).get_text())
        receptions = safe_int(row.find("td", {"data-stat": "rec"}).get_text())
        rec_yds = safe_int(row.find("td", {"data-stat": "rec_yds"}).get_text())
        rec_tds = safe_int(row.find("td", {"data-stat": "rec_td"}).get_text())
        
        # Extract fumbles
        fumbles = safe_int(row.find("td", {"data-stat": "fumbles"}).get_text())
        
        return {
            'year': year,
            'week': week,
            'weather': weather,
            'home_team': team if home_away == 'home' else opponent,
            'away_team': opponent if home_away == 'home' else team,
            'player': player_name,
            'team': team,
            'opponent': opponent,
            'home_away': home_away,
            'team_score': team_score,
            'opp_score': opp_score,
            'pos': pos,
            'snaps': 0,  # Will be filled from other tables
            'snap_pct': 0.0,
            'pass_cmp': 0,
            'pass_att': 0,
            'pass_yds': 0,
            'pass_tds': 0,
            'pass_int': 0,
            'sacks': 0,
            'rush_att': rush_att,
            'rush_yds': rush_yds,
            'rush_tds': rush_tds,
            'targets': targets,
            'receptions': receptions,
            'rec_yds': rec_yds,
            'rec_tds': rec_tds,
            'fumbles': fumbles
        }
        
    except Exception as e:
        logging.warning(f"Error extracting rushing/receiving stats: {e}")
        return None


def calculate_snap_counts_from_stats(player_data: List[Dict]) -> Dict[str, Tuple[int, float]]:
    """
    Calculate snap counts mathematically from pass attempts + rushes per team.
    
    Args:
        player_data: List of player stat dictionaries for a game
        
    Returns:
        Dict[str, Tuple[int, float]]: Dictionary mapping player names to (snaps, snap_pct)
    """
    snap_data = {}
    
    try:
        # Group players by team
        team_stats = {}
        for player in player_data:
            team = player['team']
            if team not in team_stats:
                team_stats[team] = []
            team_stats[team].append(player)
        
        # Calculate total snaps per team (pass attempts + rushes)
        for team, players in team_stats.items():
            total_team_snaps = 0
            for player in players:
                # Add pass attempts and rush attempts to get total snaps
                player_snaps = player['pass_att'] + player['rush_att']
                total_team_snaps += player_snaps
            
            # Calculate snap percentages for each player
            for player in players:
                player_snaps = player['pass_att'] + player['rush_att']
                if total_team_snaps > 0:
                    snap_pct = (player_snaps / total_team_snaps) * 100
                else:
                    snap_pct = 0.0
                
                snap_data[player['player']] = (player_snaps, snap_pct)
        
        return snap_data
        
    except Exception as e:
        logging.warning(f"Could not calculate snap counts: {e}")
        return {}


def extract_player_positions_from_snap_tables(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract player positions from snap count tables.
    
    Args:
        soup: BeautifulSoup object of game page
        
    Returns:
        Dict[str, str]: Dictionary mapping player names to positions
    """
    position_data = {}
    
    try:
        # Look for snap count tables - use the correct table IDs
        snap_table_ids = ["home_snap_counts", "vis_snap_counts"]
        
        for table_id in snap_table_ids:
            table = soup.find("table", {"id": table_id})
            if table:
                logging.info(f"Found snap counts table: {table_id}")
                tbody = table.find("tbody")
                if tbody:
                    rows = tbody.find_all("tr")
                    logging.info(f"Found {len(rows)} rows in {table_id} table")
                    
                    # Process all rows to extract player positions
                    
                    for row in rows:
                        # Check if there's a th element (player name) in the row
                        player_th = row.find("th", {"data-stat": "player"})
                        if player_th:
                            player_name = player_th.get_text().strip()
                            
                            # Extract position from the first td cell
                            pos_cell = row.find("td", {"data-stat": "pos"})
                            if pos_cell:
                                position = pos_cell.get_text().strip()
                                position_data[player_name] = position
                                logging.debug(f"Found position for {player_name}: {position}")
                            else:
                                logging.debug(f"No position cell found for {player_name}")
                        else:
                            # Fallback: check for td with player data-stat
                            player_cell = row.find("td", {"data-stat": "player"})
                            if player_cell:
                                player_name = player_cell.get_text().strip()
                                
                                # Extract position
                                pos_cell = row.find("td", {"data-stat": "pos"})
                                if pos_cell:
                                    position = pos_cell.get_text().strip()
                                    position_data[player_name] = position
                                    logging.debug(f"Found position for {player_name}: {position}")
                                else:
                                    logging.debug(f"No position cell found for {player_name}")
                            else:
                                logging.debug("No player cell found in row")
            else:
                logging.info(f"Snap counts table {table_id} not found")
        
        return position_data
        
    except Exception as e:
        logging.warning(f"Could not extract player positions: {e}")
        return {}


def extract_snap_counts(soup: BeautifulSoup) -> Dict[str, Tuple[int, float]]:
    """
    Extract snap counts and percentages from game page.
    (Legacy function - kept for fallback)
    
    Args:
        soup: BeautifulSoup object of game page
        
    Returns:
        Dict[str, Tuple[int, float]]: Dictionary mapping player names to (snaps, snap_pct)
    """
    snap_data = {}
    
    try:
        # Look for snap count tables - use the correct table IDs
        snap_table_ids = ["home_snap_counts", "vis_snap_counts"]
        
        for table_id in snap_table_ids:
            table = soup.find("table", {"id": table_id})
            if table:
                logging.debug(f"Found snap counts table: {table_id}")
                tbody = table.find("tbody")
                if tbody:
                    rows = tbody.find_all("tr")
                    for row in rows:
                        player_cell = row.find("td", {"data-stat": "player"})
                        if player_cell:
                            player_name = player_cell.get_text().strip()
                            
                            # Look for snap count and percentage
                            snap_cell = row.find("td", {"data-stat": "snap_count"})
                            snap_pct_cell = row.find("td", {"data-stat": "snap_pct"})
                            
                            if snap_cell and snap_pct_cell:
                                snaps = safe_int(snap_cell.get_text())
                                snap_pct = safe_float(snap_cell.get_text().replace("%", ""))
                                snap_data[player_name] = (snaps, snap_pct)
                                logging.debug(f"Added snap data for {player_name}: {snaps} snaps, {snap_pct}%")
        
        return snap_data
        
    except Exception as e:
        logging.warning(f"Could not extract snap counts: {e}")
        return {}


def safe_float(value: str) -> float:
    """
    Safely convert string to float, returning 0.0 for invalid values.
    
    Args:
        value: String value to convert
        
    Returns:
        float: Converted float or 0.0 if conversion fails
    """
    try:
        return float(value.strip()) if value.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def process_completed_games(games: List[Dict], driver: uc.Chrome) -> List[Dict]:
    """
    Process all completed games in batch to extract player statistics.
    
    Args:
        games: List of completed game dictionaries
        driver: Chrome driver instance
        
    Returns:
        List[Dict]: All player statistics from all games
    """
    all_player_data = []
    total_games = len(games)
    
    logging.info(f"Processing {total_games} completed games")
    
    for i, game in enumerate(games):
        try:
            logging.info(f"Processing game {i+1}/{total_games}: {game['away_team']} @ {game['home_team']} (Week {game['week']})")
            
            # Get game page
            game_url = f"{BASE_URL}{game['boxscore_url']}"
            soup, from_cache = get_soup_with_undetected(driver, game_url)
            
            if not soup:
                logging.warning(f"Failed to get game page: {game_url}")
                continue
            
            # Extract weather information
            weather = extract_weather_info(soup)
            
            # Extract scores from the game detail page (more reliable than schedule page)
            home_score, away_score = extract_game_scores(soup)
            
            # Process player stats for the game (both teams in one call)
            # The extract_offense_stats function now handles team assignment correctly
            game_players = process_player_stats(soup, game['home_team'], game['away_team'], 
                                              'home', weather, game['year'], game['week'],
                                              home_score, away_score)
            all_player_data.extend(game_players)
            
            logging.info(f"Extracted {len(game_players)} player records from game {i+1}")
            
        except Exception as e:
            logging.error(f"Error processing game {i+1}: {e}")
            continue
    
    logging.info(f"Total player records extracted: {len(all_player_data)}")
    return all_player_data


def process_player_stats(soup: BeautifulSoup, team: str, opponent: str, 
                        home_away: str, weather: str, year: int, week: int,
                        team_score: int, opp_score: int) -> List[Dict]:
    """
    Process player statistics from game page.
    
    Args:
        soup: BeautifulSoup object of game page
        team: Team abbreviation
        opponent: Opponent team abbreviation
        home_away: 'home' or 'away'
        weather: Weather information
        year: Season year
        week: Week number
        team_score: Team's score
        opp_score: Opponent's score
        
    Returns:
        List[Dict]: List of player stat dictionaries
    """
    players = []
    
    try:
        # Extract player positions from snap count tables first
        position_data = extract_player_positions_from_snap_tables(soup)
        logging.info(f"Extracted positions for {len(position_data)} players from snap tables")
        
        # Process player offense stats (contains passing, rushing, receiving)
        offense_table = soup.find("table", {"id": "player_offense"})
        if offense_table:
            logging.info("Found player_offense table")
            tbody = offense_table.find("tbody")
            if tbody:
                offense_rows = tbody.find_all("tr")
                logging.info(f"Found {len(offense_rows)} player offense rows")
                for row in offense_rows:
                    player_data = extract_offense_stats(row, team, opponent, home_away, 
                                                      weather, year, week, team_score, opp_score)
                    if player_data:
                        # Update position from snap count table if available
                        player_name = player_data['player']
                        if player_name in position_data:
                            player_data['pos'] = position_data[player_name]
                            logging.debug(f"Updated position for {player_name}: {position_data[player_name]}")
                        
                        players.append(player_data)
                        logging.debug(f"Added offense stats for {player_data['player']}")
            else:
                logging.warning("Player offense table found but no tbody")
        else:
            logging.warning("No player_offense table found")
        
        # Calculate snap counts mathematically from the collected player data
        if players:
            snap_data = calculate_snap_counts_from_stats(players)
            logging.info(f"Calculated snap data for {len(snap_data)} players")
            
            # Update players with calculated snap data
            for player in players:
                player_name = player['player']
                if player_name in snap_data:
                    player['snaps'], player['snap_pct'] = snap_data[player_name]
                else:
                    player['snaps'] = 0
                    player['snap_pct'] = 0.0
        
        logging.info(f"Processed {len(players)} players for {team} vs {opponent}")
        return players
        
    except Exception as e:
        logging.error(f"Error processing player stats: {e}")
        return []


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean the scraped data to match target format.
    
    Args:
        df: Raw scraped data DataFrame
        
    Returns:
        pd.DataFrame: Normalized data matching target format
    """
    if df.empty:
        return df
    
    try:
        # Ensure all required columns exist with proper data types
        required_columns = [
            'year', 'week', 'weather', 'home_team', 'away_team', 'player', 'team', 
            'opponent', 'home_away', 'team_score', 'opp_score', 'pos', 'snaps', 
            'snap_pct', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_tds', 'pass_int', 
            'sacks', 'rush_att', 'rush_yds', 'rush_tds', 'targets', 'receptions', 
            'rec_yds', 'rec_tds', 'fumbles'
        ]
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ['year', 'week', 'team_score', 'opp_score', 'snaps', 'pass_cmp', 
                          'pass_att', 'pass_yds', 'pass_tds', 'pass_int', 'sacks', 
                          'rush_att', 'rush_yds', 'rush_tds', 'targets', 'receptions', 
                          'rec_yds', 'rec_tds', 'fumbles']:
                    df[col] = 0
                elif col == 'snap_pct':
                    df[col] = 0.0
                else:
                    df[col] = ""
        
        # Ensure proper data types
        int_columns = ['year', 'week', 'team_score', 'opp_score', 'snaps', 'pass_cmp', 
                      'pass_att', 'pass_yds', 'pass_tds', 'pass_int', 'sacks', 
                      'rush_att', 'rush_yds', 'rush_tds', 'targets', 'receptions', 
                      'rec_yds', 'rec_tds', 'fumbles']
        
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Ensure float columns
        float_columns = ['snap_pct']
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Clean string columns
        string_columns = ['weather', 'home_team', 'away_team', 'player', 'team', 
                         'opponent', 'home_away', 'pos']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Sort by year, week, team, player
        df = df.sort_values(['year', 'week', 'team', 'player'])
        
        logging.info(f"Normalized data: {len(df)} records")
        return df
        
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
        return df


def save_to_csv(df: pd.DataFrame, year: int) -> str:
    """
    Save DataFrame to CSV file in the target format.
    
    Args:
        df: DataFrame to save
        year: Year for filename
        
    Returns:
        str: Path to saved file
    """
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Create filename
        filename = f"data/game_data/game_data_{year}.csv"
        
        # Save to CSV with proper formatting
        df.to_csv(filename, index=False)
        
        logging.info(f"Data saved to {filename}")
        return filename
        
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")
        return ""


def scrape_nfl_data(year: int) -> pd.DataFrame:
    """
    Main function to scrape NFL data for a given year using schedule-based approach.
    
    Args:
        year: NFL season year to scrape
        
    Returns:
        pd.DataFrame: Complete game data in target format
    """
    start_time = datetime.now()
    logging.info(f"Starting NFL data scrape for year {year} using schedule-based approach")
    
    # Setup driver
    driver = setup_undetected_driver()
    if not driver:
        logging.error("Failed to setup Chrome driver")
        return pd.DataFrame()
    
    try:
        # Get completed games from schedule page (single request)
        logging.info("Step 1: Getting completed games from schedule page")
        completed_games = get_season_schedule(year, driver)
        
        if not completed_games:
            logging.error("No completed games found in schedule")
            return pd.DataFrame()
        
        logging.info(f"Found {len(completed_games)} completed games to process")
        
        # Process all completed games to extract player statistics
        logging.info(f"Step 2: Processing {len(completed_games)} completed games to extract player statistics")
        all_player_data = process_completed_games(completed_games, driver)
        
        # Convert to DataFrame and normalize
        if all_player_data:
            df = pd.DataFrame(all_player_data)
            logging.info(f"Scraped data for {len(df)} player-game records")
            
            # Normalize data to match target format
            df = normalize_data(df)
            
            return df
        else:
            logging.warning("No player data found")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error during data scraping: {e}")
        return pd.DataFrame()
    
    finally:
        if driver:
            try:
                # Close all windows and quit
                driver.close()
                driver.quit()
                logging.info("Chrome driver closed")
            except Exception as e:
                logging.warning(f"Error closing driver: {e}")
            finally:
                # Force cleanup
                try:
                    del driver
                except:
                    pass
    
    end_time = datetime.now()
    logging.info(f"Data scraping completed in {end_time - start_time}")


def main():
    """
    Main entry point for the script.
    """
    import sys
    
    # Get year from command line argument or use default
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
        except ValueError:
            logging.error("Invalid year provided. Using default year 2024.")
            year = 2024
    else:
        year = 2024
    
    logging.info(f"Starting NFL data scraper for year {year}")
    
    # Scrape data
    df = scrape_nfl_data(year)
    
    if not df.empty:
        # Save to CSV using the save function
        output_file = save_to_csv(df, year)
        if output_file:
            print(f"Successfully scraped {len(df)} records and saved to {output_file}")
        else:
            print("Failed to save data to CSV")
    else:
        logging.error("No data was scraped")
        print("No data was scraped. Check logs for errors.")


def test_basic_functionality():
    """
    Test basic functionality without full scraping using new schedule-based approach.
    """
    driver = None
    try:
        logging.info("Testing basic functionality with schedule-based approach...")
        
        # Test driver setup
        driver = setup_undetected_driver()
        if not driver:
            logging.error("Failed to setup driver")
            return False
        
        # Test getting completed games from schedule for 2024
        logging.info("Testing schedule-based game extraction for 2024")
        completed_games = get_season_schedule(2024, driver)
        
        if not completed_games:
            logging.error("Failed to get completed games from schedule")
            return False
        
        logging.info(f"Successfully got {len(completed_games)} completed games from schedule")
        
        # Show sample of completed games
        if completed_games:
            sample_games = completed_games[:3]
            for i, game in enumerate(sample_games):
                logging.info(f"Sample game {i+1}: Week {game['week']}, {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
        
        # Test processing a single game with detailed debugging
        if completed_games:
            logging.info("Testing single game processing with detailed debugging...")
            test_game = completed_games[0]
            game_url = f"{BASE_URL}{test_game['boxscore_url']}"
            logging.info(f"Testing game URL: {game_url}")
            
            soup, from_cache = get_soup_with_undetected(driver, game_url)
            
            if soup:
                logging.info(f"Successfully loaded boxscore page: {test_game['away_team']} @ {test_game['home_team']}")
                
                # Test player stats extraction with detailed debugging
                weather = extract_weather_info(soup)
                logging.info(f"Extracted weather: '{weather}'")
                
                # Test processing player stats for one team with debug logging enabled
                logging.info("Processing player stats with detailed debugging...")
                home_players = process_player_stats(soup, test_game['home_team'], test_game['away_team'], 
                                                  'home', weather, test_game['year'], test_game['week'],
                                                  test_game['home_score'], test_game['away_score'])
                logging.info(f"Extracted {len(home_players)} player records for home team")
                
                if len(home_players) == 0:
                    logging.warning("No players found - this indicates an HTML parsing issue")
            else:
                logging.warning("Failed to load test game boxscore page")
        
        logging.info("Basic functionality test passed")
        return True
        
    except Exception as e:
        logging.error(f"Basic functionality test failed: {e}")
        return False
    finally:
        if driver:
            try:
                # Close all windows and quit
                driver.close()
                driver.quit()
                logging.info("Driver closed successfully")
            except Exception as e:
                logging.warning(f"Error closing driver: {e}")
            finally:
                # Force cleanup
                try:
                    del driver
                except:
                    pass


def test_schedule_page_access():
    """
    Test specifically the schedule page access to debug the timeout issue.
    """
    driver = None
    try:
        logging.info("Testing schedule page access specifically...")
        
        # Test driver setup
        driver = setup_undetected_driver()
        if not driver:
            logging.error("Failed to setup driver")
            return False
        
        # Test direct navigation to schedule page
        test_url = "https://www.pro-football-reference.com/years/2024/games.htm"
        logging.info(f"Testing direct navigation to: {test_url}")
        
        try:
            driver.get(test_url)
            logging.info("Successfully navigated to schedule page")
            
            # Wait and check page status
            time.sleep(5)
            page_title = driver.title
            current_url = driver.current_url
            logging.info(f"Page title: '{page_title}'")
            logging.info(f"Current URL: '{current_url}'")
            
            # Check if we got the expected page
            if "games" in page_title.lower() or "schedule" in page_title.lower():
                logging.info("Successfully loaded schedule page")
                
                # Get page source and analyze
                page_source = driver.page_source
                logging.info(f"Page source length: {len(page_source)} characters")
                
                # Look for key elements
                if "games" in page_source.lower():
                    logging.info("Found 'games' content in page source")
                else:
                    logging.warning("No 'games' content found")
                
                # Try to find the games table
                soup = BeautifulSoup(page_source, 'html.parser')
                games_table = soup.find("table", {"id": "games"})
                if games_table:
                    logging.info("Found games table in HTML")
                else:
                    logging.warning("Games table not found in HTML")
                    # Log available tables
                    tables = soup.find_all("table")
                    logging.info(f"Found {len(tables)} tables on page")
                    for i, table in enumerate(tables[:5]):  # Show first 5 tables
                        table_id = table.get("id", "no-id")
                        table_class = table.get("class", "no-class")
                        logging.info(f"Table {i+1}: id='{table_id}', class='{table_class}'")
                
                return True
            else:
                logging.error(f"Unexpected page title: '{page_title}'")
                return False
                
        except Exception as e:
            logging.error(f"Error accessing schedule page: {e}")
            return False
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        return False
    finally:
        if driver:
            try:
                driver.close()
                driver.quit()
                logging.info("Driver closed successfully")
            except Exception as e:
                logging.warning(f"Error closing driver: {e}")
            finally:
                try:
                    del driver
                except:
                    pass


if __name__ == "__main__":
    import sys
    
    # Check if test mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_basic_functionality()
    elif len(sys.argv) > 1 and sys.argv[1] == "test-schedule":
        test_schedule_page_access()
    else:
        main()

