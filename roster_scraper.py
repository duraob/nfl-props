"""
CBS Sports Depth Chart Scraper - Complete Version
Scrapes depth charts for all positions and captures both AFC and NFC players.
Uses the working position-based URLs and extracts all teams from each page.
"""

import pandas as pd
import os
import requests
from datetime import datetime
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import re
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('depth_chart_scraper_complete.log'),
        logging.StreamHandler()
    ]
)

# Global configuration
BASE_URL = "https://www.cbssports.com"
DATA_DIR = 'data/depth_charts'

# Position-specific URLs (these work)
POSITION_URLS = {
    'QB': f"{BASE_URL}/fantasy/football/depth-chart/QB/",
    'RB': f"{BASE_URL}/fantasy/football/depth-chart/RB/",
    'WR': f"{BASE_URL}/fantasy/football/depth-chart/WR/",
    'TE': f"{BASE_URL}/fantasy/football/depth-chart/TE/"
}

# NFL team abbreviations for both AFC and NFC
NFL_TEAMS = {
    'AFC': {
        'East': ['BUF', 'MIA', 'NE', 'NYJ'],
        'North': ['BAL', 'CIN', 'CLE', 'PIT'],
        'South': ['HOU', 'IND', 'JAX', 'TEN'],
        'West': ['DEN', 'KC', 'LV', 'LAC']
    },
    'NFC': {
        'East': ['DAL', 'NYG', 'PHI', 'WAS'],
        'North': ['CHI', 'DET', 'GB', 'MIN'],
        'South': ['ATL', 'CAR', 'NO', 'TB'],
        'West': ['ARI', 'LAR', 'SF', 'SEA']
    }
}

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)

def get_page_content(url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
    """
    Get page content using requests with retry logic.
    
    Args:
        url: URL to scrape
        max_retries: Maximum number of retry attempts
        
    Returns:
        Optional[BeautifulSoup]: BeautifulSoup object or None if failed
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if we got valid content
            if len(response.text) < 1000:
                logging.warning(f"Got short response ({len(response.text)} chars), retrying...")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {2 + attempt} seconds...")
                time.sleep(2 + attempt)
                continue
            else:
                return None
    
    return None


def extract_all_teams_from_position_page(soup: BeautifulSoup, position: str) -> List[Dict]:
    """
    Extract depth chart data for all teams from a position page.
    This captures both AFC and NFC players from BOTH tables on the page.
    
    Args:
        soup: BeautifulSoup object of the depth chart page
        position: Position being scraped (QB, RB, WR, TE)
        
    Returns:
        List[Dict]: List of player depth chart entries for all teams
    """
    players = []
    
    try:
        # Look for ALL CBS Sports tables (there should be 2: AFC and NFC)
        table_selectors = [
            "table.TableBase-table",
            "table[class*='TableBase']",
            "table[class*='depth']",
            "table[class*='roster']",
            "table[class*='depth-chart']"
        ]
        
        # Find ALL tables, not just the first one
        all_tables = []
        for selector in table_selectors:
            tables = soup.select(selector)
            if tables:
                all_tables.extend(tables)
                logging.info(f"Found {len(tables)} tables using selector: {selector}")
        
        # Fallback to any table if specific selectors don't work
        if not all_tables:
            all_tables = soup.find_all("table")
            logging.info(f"Using all available tables ({len(all_tables)} tables found)")
        
        if not all_tables:
            logging.warning(f"No depth chart tables found for {position}")
            # Try alternative parsing
            return extract_players_from_alternative_structure(soup, position)
        
        logging.info(f"Processing {len(all_tables)} tables for {position}")
        
        # Process each table (AFC and NFC)
        for table_idx, depth_table in enumerate(all_tables):
            try:
                # Extract rows from the table
                rows = depth_table.find_all("tr")
                logging.info(f"Table {table_idx + 1}: Found {len(rows)} rows")
                
                table_players = []
                for row_idx, row in enumerate(rows):
                    try:
                        # Skip header rows
                        if row_idx == 0:
                            continue
                        
                        cells = row.find_all(["td", "th"])
                        if len(cells) < 2:
                            continue
                        
                        # Extract ALL players from this row (starters and backups)
                        row_players = extract_all_players_from_depth_row(cells, position, row_idx)
                        if row_players:
                            table_players.extend(row_players)
                            
                    except Exception as e:
                        logging.warning(f"Error processing row {row_idx} in table {table_idx + 1} for {position}: {e}")
                        continue
                
                players.extend(table_players)
                logging.info(f"Table {table_idx + 1}: Extracted {len(table_players)} players")
                
            except Exception as e:
                logging.warning(f"Error processing table {table_idx + 1} for {position}: {e}")
                continue
        
        # If no players found in structured tables, try alternative parsing
        if not players:
            logging.info(f"No players found in structured tables for {position}, trying alternative parsing...")
            players = extract_players_from_alternative_structure(soup, position)
        
        logging.info(f"Total players extracted for {position}: {len(players)}")
        return players
        
    except Exception as e:
        logging.error(f"Error extracting depth chart for {position}: {e}")
        return []


def extract_all_players_from_depth_row(cells: List, position: str, row_idx: int) -> List[Dict]:
    """
    Extract ALL players from a depth chart table row including starters and backups.
    
    Args:
        cells: List of table cells (td/th elements)
        position: Position being processed
        row_idx: Row index for depth ranking
        
    Returns:
        List[Dict]: List of player data dictionaries
    """
    players = []
    
    try:
        if len(cells) < 2:
            return players
        
        # Extract team name (usually first cell)
        team_cell = cells[0]
        team_name = team_cell.get_text().strip()
        
        # Look for ALL player links in ALL cells (not just the first one)
        for cell_idx, cell in enumerate(cells[1:], 1):  # Skip team cell
            # Look for player links in this cell
            player_links = cell.find_all("a")
            for link in player_links:
                link_text = link.get_text().strip()
                if link_text and len(link_text) > 2:  # Valid player name
                    player_link = link.get("href", "")
                    
                    # Check for injury indicators in the player name or surrounding text
                    cell_text = cell.get_text().strip().lower()
                    injury_status = ""
                    
                    # More specific injury detection - look for whole words and common patterns
                    injury_patterns = [
                        r'\binjured\b', r'\bir\b', r'\bout\b', r'\bquestionable\b', r'\bdoubtful\b',
                        r'\b(placed on ir)\b', r'\b(ir list)\b', r'\b(injured reserve)\b',
                        r'\b(out for season)\b', r'\b(season ending)\b'
                    ]
                    
                    import re
                    has_injury_indicator = any(re.search(pattern, cell_text) for pattern in injury_patterns)
                    if has_injury_indicator:
                        injury_status = cell_text
                    
                    # Determine injury status
                    is_injured = bool(injury_status) or any(re.search(pattern, link_text.lower()) for pattern in injury_patterns)
                    
                    # Calculate depth rank (row position + cell position)
                    depth_rank = row_idx + (cell_idx * 0.1)
                    
                    # Determine depth label (just the row number)
                    depth_label = str(row_idx)
                    
                    players.append({
                        'position': position,
                        'team': team_name,
                        'player_name': link_text,
                        'player_link': player_link,
                        'injury_status': injury_status,
                        'is_injured': is_injured,
                        'depth_rank': depth_rank,
                        'depth_label': depth_label,
                        'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # If no player links found, try to extract from cell text
        if not players:
            for cell_idx, cell in enumerate(cells[1:], 1):
                cell_text = cell.get_text().strip()
                if cell_text and len(cell_text) > 2 and not cell_text.lower() in ['starter', 'backup', 'depth']:
                    # Check for injury indicators
                    injury_status = ""
                    
                    # More specific injury detection - look for whole words and common patterns
                    injury_patterns = [
                        r'\binjured\b', r'\bir\b', r'\bout\b', r'\bquestionable\b', r'\bdoubtful\b',
                        r'\b(placed on ir)\b', r'\b(ir list)\b', r'\b(injured reserve)\b',
                        r'\b(out for season)\b', r'\b(season ending)\b'
                    ]
                    
                    has_injury_indicator = any(re.search(pattern, cell_text.lower()) for pattern in injury_patterns)
                    if has_injury_indicator:
                        injury_status = cell_text
                    
                    # Determine injury status
                    is_injured = bool(injury_status)
                    
                    # Calculate depth rank
                    depth_rank = row_idx + (cell_idx * 0.1)
                    
                    # Determine depth label (just the row number)
                    depth_label = str(row_idx)
                    
                    players.append({
                        'position': position,
                        'team': team_name,
                        'player_name': cell_text,
                        'player_link': "",
                        'injury_status': injury_status,
                        'is_injured': is_injured,
                        'depth_rank': depth_rank,
                        'depth_label': depth_label,
                        'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return players
        
    except Exception as e:
        logging.warning(f"Error extracting players from row: {e}")
        return players


def extract_player_from_depth_row(cells: List, position: str, row_idx: int) -> Optional[Dict]:
    """
    Extract player information from a depth chart table row.
    Now captures all players including backups, not just starters.
    
    Args:
        cells: List of table cells (td/th elements)
        position: Position being processed
        row_idx: Row index for depth ranking
        
    Returns:
        Optional[Dict]: Player data dictionary or None if invalid
    """
    try:
        if len(cells) < 2:
            return None
        
        # Extract team name (usually first cell)
        team_cell = cells[0]
        team_name = team_cell.get_text().strip()
        
        # Look for ALL player links in cells (not just the first one)
        players_found = []
        
        for i, cell in enumerate(cells[1:], 1):  # Skip team cell
            # Look for player links in this cell
            player_links = cell.find_all("a")
            for link in player_links:
                link_text = link.get_text().strip()
                if link_text and len(link_text) > 2:  # Valid player name
                    player_link = link.get("href", "")
                    
                    # Check for injury indicators in the player name or surrounding text
                    cell_text = cell.get_text().strip().lower()
                    injury_status = ""
                    
                    # More specific injury detection - look for whole words and common patterns
                    injury_patterns = [
                        r'\binjured\b', r'\bir\b', r'\bout\b', r'\bquestionable\b', r'\bdoubtful\b',
                        r'\b(placed on ir)\b', r'\b(ir list)\b', r'\b(injured reserve)\b',
                        r'\b(out for season)\b', r'\b(season ending)\b'
                    ]
                    
                    has_injury_indicator = any(re.search(pattern, cell_text) for pattern in injury_patterns)
                    if has_injury_indicator:
                        injury_status = cell_text
                    
                    players_found.append({
                        'player_name': link_text,
                        'player_link': player_link,
                        'injury_status': injury_status,
                        'cell_index': i
                    })
        
        # If no player links found, try to extract from cell text
        if not players_found:
            for i, cell in enumerate(cells[1:], 1):
                cell_text = cell.get_text().strip()
                if cell_text and len(cell_text) > 2 and not cell_text.lower() in ['starter', 'backup', 'depth']:
                    # Check for injury indicators
                    injury_status = ""
                    if any(indicator in cell_text.lower() for indicator in ['injured', 'ir', 'out', 'questionable', 'doubtful']):
                        injury_status = cell_text
                    
                    players_found.append({
                        'player_name': cell_text,
                        'player_link': "",
                        'injury_status': injury_status,
                        'cell_index': i
                    })
        
        # If no players found, return None
        if not players_found:
            return None
        
        # Process each player found in this row
        all_players = []
        for player_idx, player_info in enumerate(players_found):
            player_name = player_info['player_name']
            player_link = player_info['player_link']
            injury_status = player_info['injury_status']
            
            # Determine injury status
            injury_patterns = [
                r'\binjured\b', r'\bir\b', r'\bout\b', r'\bquestionable\b', r'\bdoubtful\b',
                r'\b(placed on ir)\b', r'\b(ir list)\b', r'\b(injured reserve)\b',
                r'\b(out for season)\b', r'\b(season ending)\b'
            ]
            is_injured = bool(injury_status) or any(re.search(pattern, player_name.lower()) for pattern in injury_patterns)
            
            # Calculate depth rank (row position + player position within row)
            depth_rank = row_idx + (player_idx * 0.1)  # Slight offset for multiple players in same row
            
            # Determine depth label (just the row number)
            depth_label = str(row_idx)
            
            all_players.append({
                'position': position,
                'team': team_name,
                'player_name': player_name,
                'player_link': player_link,
                'injury_status': injury_status,
                'is_injured': is_injured,
                'depth_rank': depth_rank,
                'depth_label': depth_label,
                'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Return the first player (we'll handle multiple players in the calling function)
        return all_players[0] if all_players else None
        
    except Exception as e:
        logging.warning(f"Error extracting player from row: {e}")
        return None


def extract_players_from_alternative_structure(soup: BeautifulSoup, position: str) -> List[Dict]:
    """
    Alternative method to extract players when standard table parsing fails.
    
    Args:
        soup: BeautifulSoup object
        position: Position being processed
        
    Returns:
        List[Dict]: List of player data
    """
    players = []
    
    try:
        # Look for player links throughout the page
        player_links = soup.find_all("a", href=re.compile(r'/nfl/players/'))
        
        logging.info(f"Found {len(player_links)} player links on {position} page")
        
        for link in player_links:
            try:
                player_name = link.get_text().strip()
                player_link = link.get("href", "")
                
                if not player_name or len(player_name) < 3:
                    continue
                
                # Try to find associated team information
                # Look for parent elements that might contain team info
                team_name = "Unknown"
                parent = link.parent
                for _ in range(3):  # Check up to 3 parent levels
                    if parent:
                        parent_text = parent.get_text().strip()
                        # Look for team abbreviations or names
                        team_match = re.search(r'\b([A-Z]{2,4})\b', parent_text)
                        if team_match:
                            team_name = team_match.group(1)
                            break
                        parent = parent.parent
                
                # Check for injury indicators
                injury_patterns = [
                    r'\binjured\b', r'\bir\b', r'\bout\b', r'\bquestionable\b', r'\bdoubtful\b',
                    r'\b(placed on ir)\b', r'\b(ir list)\b', r'\b(injured reserve)\b',
                    r'\b(out for season)\b', r'\b(season ending)\b'
                ]
                is_injured = any(re.search(pattern, player_name.lower()) for pattern in injury_patterns)
                
                players.append({
                    'position': position,
                    'team': team_name,
                    'player_name': player_name,
                    'player_link': player_link,
                    'injury_status': 'Injured' if is_injured else '',
                    'is_injured': is_injured,
                    'depth_rank': 1,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
            except Exception as e:
                logging.warning(f"Error processing player link: {e}")
                continue
        
        return players
        
    except Exception as e:
        logging.error(f"Error in alternative extraction: {e}")
        return []


def scrape_position_depth_chart(position: str) -> List[Dict]:
    """
    Scrape depth chart for a specific position (captures all teams).
    
    Args:
        position: Position to scrape (QB, RB, WR, TE)
        
    Returns:
        List[Dict]: List of player depth chart entries
    """
    if position not in POSITION_URLS:
        logging.error(f"Invalid position: {position}")
        return []
    
    url = POSITION_URLS[position]
    logging.info(f"Scraping {position} depth chart from {url}")
    
    soup = get_page_content(url)
    
    if not soup:
        logging.error(f"Failed to get {position} depth chart page")
        return []
    
    # Extract depth chart data for all teams
    players = extract_all_teams_from_position_page(soup, position)
    
    logging.info(f"Scraped {len(players)} players for {position}")
    return players


def scrape_all_depth_charts() -> Dict[str, List[Dict]]:
    """
    Scrape depth charts for all positions (captures both AFC and NFC teams).
    
    Returns:
        Dict[str, List[Dict]]: Dictionary mapping positions to player lists
    """
    all_depth_charts = {}
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        try:
            logging.info(f"Scraping {position} depth chart...")
            players = scrape_position_depth_chart(position)
            all_depth_charts[position] = players
            
            # Add delay between positions to be respectful
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error scraping {position} depth chart: {e}")
            all_depth_charts[position] = []
    
    return all_depth_charts


def filter_healthy_players(all_depth_charts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Filter out injured players from depth charts.
    
    Args:
        all_depth_charts: Dictionary of position -> player lists
        
    Returns:
        Dict[str, List[Dict]]: Filtered depth charts with only healthy players
    """
    healthy_charts = {}
    
    for position, players in all_depth_charts.items():
        healthy_players = [player for player in players if not player.get('is_injured', False)]
        healthy_charts[position] = healthy_players
        
        injured_count = len(players) - len(healthy_players)
        logging.info(f"{position}: {len(healthy_players)} healthy players, {injured_count} injured players filtered out")
    
    return healthy_charts


def create_master_roster_dataframe(all_depth_charts: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create a consolidated master roster DataFrame from all depth charts.
    Includes deduplication to remove duplicate players.
    
    Args:
        all_depth_charts: Dictionary of position -> player lists
        
    Returns:
        pd.DataFrame: Consolidated master roster with duplicates removed
    """
    all_players = []
    
    for position, players in all_depth_charts.items():
        for player in players:
            all_players.append(player)
    
    if not all_players:
        logging.warning("No players found to create master roster")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_players)
    
    # Log original count before deduplication
    original_count = len(df)
    logging.info(f"Original player count: {original_count}")
    
    # Enhanced deduplication logic - prioritize records with complete information
    def get_record_quality_score(row):
        """Calculate a quality score for each record to determine which duplicate to keep."""
        score = 0
        
        # Strongly prefer records with full player names (not abbreviated)
        # Check for common abbreviation patterns
        name = row['player_name']
        is_abbreviated = (
            len(name) <= 3 or  # Very short names
            ('.' in name and len(name.split('.')[0]) <= 2) or  # J. Smith pattern
            (len(name.split()) == 1 and len(name) <= 4) or  # Single short names
            name.count(' ') == 0 and len(name) <= 5  # No spaces and short
        )
        
        if not is_abbreviated:
            score += 20  # Much higher score for full names
        else:
            score -= 10  # Penalty for abbreviated names
        
        # Prefer records with player links (more complete data)
        if row['player_link'] and len(row['player_link']) > 10:
            score += 5
        
        # Prefer records with depth labels (more structured data)
        if row.get('depth_label') and row['depth_label'] != 'Unknown':
            score += 3
        
        # Prefer records with lower depth rank (starters over backups for duplicates)
        if row.get('depth_rank'):
            score += (10 - min(row['depth_rank'], 10))  # Lower depth rank = higher score
        
        # Prefer records with injury status information
        if row.get('injury_status') and row['injury_status']:
            score += 2
        
        # Prefer records with conference information
        if row.get('conference') and row['conference'] != 'Unknown':
            score += 1
        
        return score
    
    # Add quality score to each record
    df['quality_score'] = df.apply(get_record_quality_score, axis=1)
    
    # Sort by quality score (highest first) to prioritize better records
    df_sorted = df.sort_values(['player_link', 'quality_score'], ascending=[True, False])
    
    # Keep the best record for each unique player link
    df_deduped = df_sorted.drop_duplicates(subset=['player_link'], keep='first')
    
    # Remove the quality score column as it's no longer needed
    df_deduped = df_deduped.drop('quality_score', axis=1)
    
    # Log deduplication results
    duplicates_removed = original_count - len(df_deduped)
    logging.info(f"Removed {duplicates_removed} duplicate players")
    logging.info(f"Final player count after deduplication: {len(df_deduped)}")
    
    # Log quality of remaining records
    if 'depth_label' in df_deduped.columns:
        depth_counts = df_deduped['depth_label'].value_counts()
        logging.info(f"Depth distribution after deduplication:")
        for depth, count in depth_counts.items():
            logging.info(f"  {depth}: {count} players")
    
    # No additional columns needed - keeping only essential data
    
    # Add conference information
    def get_conference(team):
        for conf, divisions in NFL_TEAMS.items():
            for div, teams in divisions.items():
                if team in teams:
                    return conf
        return 'Unknown'
    
    df_deduped['conference'] = df_deduped['team'].apply(get_conference)
    
    # Sort by conference, team, position, player name
    df_deduped = df_deduped.sort_values(['conference', 'team', 'position', 'player_name'])
    
    logging.info(f"Created master roster with {len(df_deduped)} unique players")
    return df_deduped


def save_master_roster_excel(df: pd.DataFrame, filename: str = None) -> str:
    """
    Save master roster to Excel file with formatting.
    
    Args:
        df: Master roster DataFrame
        filename: Optional custom filename
        
    Returns:
        str: Path to saved Excel file
    """
    if df.empty:
        logging.error("No data to save to Excel")
        return ""
    
    try:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"roster.xlsx"
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # Create Excel writer with formatting
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write main roster
            df.to_excel(writer, sheet_name='Master_Roster', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Master_Roster']
            
            # Apply formatting
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            
            # Format headers
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Create position-specific sheets
            for position in ['QB', 'RB', 'WR', 'TE']:
                position_df = df[df['position'] == position].copy()
                if not position_df.empty:
                    position_df.to_excel(writer, sheet_name=f'{position}_Roster', index=False)
                    
                    # Format position sheet
                    pos_worksheet = writer.sheets[f'{position}_Roster']
                    for cell in pos_worksheet[1]:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Create conference-specific sheets
            for conference in ['AFC', 'NFC']:
                conf_df = df[df['conference'] == conference].copy()
                if not conf_df.empty:
                    conf_df.to_excel(writer, sheet_name=f'{conference}_Roster', index=False)
                    
                    # Format conference sheet
                    conf_worksheet = writer.sheets[f'{conference}_Roster']
                    for cell in conf_worksheet[1]:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal="center", vertical="center")
        
        logging.info(f"Master roster saved to {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Error saving master roster to Excel: {e}")
        return ""


def scrape_complete_depth_charts() -> Tuple[pd.DataFrame, str]:
    """
    Main function to scrape complete depth charts for all positions and teams.
    
    Returns:
        Tuple[pd.DataFrame, str]: (master_roster_dataframe, excel_file_path)
    """
    start_time = datetime.now()
    logging.info("Starting complete depth chart scraping for all positions and teams")
    
    try:
        # Scrape all depth charts
        logging.info("Scraping depth charts for all positions...")
        all_depth_charts = scrape_all_depth_charts()
        
        if not any(all_depth_charts.values()):
            logging.error("No depth chart data found")
            return pd.DataFrame(), ""
        
        # Filter out injured players
        logging.info("Filtering out injured players...")
        healthy_charts = filter_healthy_players(all_depth_charts)
        
        # Create master roster
        logging.info("Creating master roster...")
        master_roster_df = create_master_roster_dataframe(healthy_charts)
        
        if master_roster_df.empty:
            logging.error("No data in master roster")
            return pd.DataFrame(), ""
        
        # Save to Excel
        logging.info("Saving master roster to Excel...")
        excel_path = save_master_roster_excel(master_roster_df)
        
        end_time = datetime.now()
        logging.info(f"Complete depth chart scraping completed in {end_time - start_time}")
        
        return master_roster_df, excel_path
        
    except Exception as e:
        logging.error(f"Error during complete depth chart scraping: {e}")
        return pd.DataFrame(), ""


def main():
    """
    Main entry point for the complete depth chart scraper.
    """
    logging.info("Starting CBS Sports complete depth chart scraper")
    
    # Scrape depth charts
    master_roster_df, excel_path = scrape_complete_depth_charts()
    
    if not master_roster_df.empty:
        print(f"Successfully scraped {len(master_roster_df)} players")
        print(f"Master roster saved to: {excel_path}")
        
        # Print summary by position
        position_counts = master_roster_df['position'].value_counts()
        print("\nPlayers by position:")
        for position, count in position_counts.items():
            print(f"  {position}: {count} players")
        
        # Print summary by conference
        conference_counts = master_roster_df['conference'].value_counts()
        print("\nPlayers by conference:")
        for conference, count in conference_counts.items():
            print(f"  {conference}: {count} players")
        
        # Print summary by team
        team_counts = master_roster_df['team'].value_counts()
        print(f"\nTeams represented: {len(team_counts)}")
        print("Top 10 teams by player count:")
        for team, count in team_counts.head(10).items():
            print(f"  {team}: {count} players")
    else:
        logging.error("No data was scraped")
        print("No data was scraped. Check logs for errors.")


if __name__ == "__main__":
    main()
