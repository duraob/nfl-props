"""
Enhanced NFL Insights Formatter

This program formats and displays AI analysis from grok_insights and stats_insights files
with improved readability and optional CSV export functionality.
"""

import json
import sys
import os
import csv
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_picks_from_analysis(analysis_text: str) -> List[Dict[str, str]]:
    """
    Extract individual picks from the analysis text.
    
    Args:
        analysis_text: The full analysis text from GROK
        
    Returns:
        List of dictionaries containing pick information
    """
    picks = []
    
    # Split by numbered sections (#### 1., #### 2., etc.)
    sections = re.split(r'#### \d+\.', analysis_text)
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        pick_data = {
            'pick_number': i,
            'type': 'PICK',
            'player': '',
            'prop_type': '',
            'line': '',
            'recommendation': '',
            'confidence': '',
            'reasoning': '',
            'risk_factors': ''
        }
        
        # Extract player name (new format: "Player Name (Team vs Team)" or old format: "PLAYER NAME: Player Name (Team)")
        # First try new format: " Geno Smith (Las Vegas Raiders vs. Chicago Bears)" (note the leading space)
        player_match = re.search(r'^\s*([^(]+?)\s*\(', section)
        if not player_match:
            # Try old format as fallback: "PLAYER NAME: Geno Smith (Seattle Seahawks)"
            player_match = re.search(r'PLAYER NAME: ([^(]+)', section)
        if player_match:
            pick_data['player'] = player_match.group(1).strip()
        
        # Extract prop type and line (new format: "Prop Type and Line" or old format: "PROP TYPE and LINE")
        prop_match = re.search(r'\*\*Prop Type and Line\*\*: (.+)', section)
        if not prop_match:
            # Try old format as fallback
            prop_match = re.search(r'\*\*PROP TYPE and LINE\*\*: (.+)', section)
        if prop_match:
            prop_line = prop_match.group(1).strip()
            # Split prop type and line (e.g., "Pass Interceptions 0.5")
            parts = prop_line.rsplit(' ', 1)
            if len(parts) == 2:
                pick_data['prop_type'] = parts[0]
                pick_data['line'] = parts[1]
            else:
                pick_data['prop_type'] = prop_line
        
        # Extract recommendation (new format: "Recommendation" or old format: "RECOMMENDATION")
        rec_match = re.search(r'\*\*Recommendation\*\*: (.+)', section)
        if not rec_match:
            # Try old format as fallback
            rec_match = re.search(r'\*\*RECOMMENDATION\*\*: (.+)', section)
        if rec_match:
            pick_data['recommendation'] = rec_match.group(1).strip()
        
        # Extract confidence level (new format: "Confidence Level" or old format: "CONFIDENCE LEVEL")
        conf_match = re.search(r'\*\*Confidence Level\*\*: (.+)', section)
        if not conf_match:
            # Try old format as fallback
            conf_match = re.search(r'\*\*CONFIDENCE LEVEL\*\*: (.+)', section)
        if conf_match:
            pick_data['confidence'] = conf_match.group(1).strip()
        
        # Extract key reasoning (new format: "Key Reasoning" or old format: "KEY REASONING")
        reason_match = re.search(r'\*\*Key Reasoning\*\*: (.+?)(?=\*\*Risk Factors\*\*|$)', section, re.DOTALL)
        if not reason_match:
            # Try old format as fallback
            reason_match = re.search(r'\*\*KEY REASONING\*\*: (.+?)(?=\*\*RISK FACTORS\*\*|$)', section, re.DOTALL)
        if reason_match:
            pick_data['reasoning'] = reason_match.group(1).strip()
        
        # Extract risk factors (new format: "Risk Factors" or old format: "RISK FACTORS")
        risk_match = re.search(r'\*\*Risk Factors\*\*: (.+)', section, re.DOTALL)
        if not risk_match:
            # Try old format as fallback
            risk_match = re.search(r'\*\*RISK FACTORS\*\*: (.+)', section, re.DOTALL)
        if risk_match:
            pick_data['risk_factors'] = risk_match.group(1).strip()
        
        picks.append(pick_data)
    
    return picks


def extract_stats_from_insights(insights_data: List[Dict]) -> List[Dict[str, str]]:
    """
    Extract individual stats insights from the insights data.
    
    Args:
        insights_data: List of insight dictionaries
        
    Returns:
        List of dictionaries containing stat information
    """
    stats = []
    
    for i, insight in enumerate(insights_data, 1):
        if isinstance(insight, dict) and 'insight' in insight:
            stat_data = {
                'pick_number': i,
                'type': 'STAT',
                'player': insight.get('player', ''),
                'prop_type': '',
                'line': '',
                'recommendation': '',
                'confidence': '',
                'reasoning': insight.get('insight', ''),
                'risk_factors': ''
            }
            stats.append(stat_data)
    
    return stats


def print_formatted_analysis(week_number: int, picks: List[Dict], stats: List[Dict]) -> None:
    """
    Print formatted analysis with proper spacing and readability.
    
    Args:
        week_number: NFL week number
        picks: List of pick dictionaries
        stats: List of stat dictionaries
    """
    import textwrap
    
    print("=" * 100)
    print(f"NFL WEEK {week_number} ANALYSIS")
    print("=" * 100)
    print()
    
    # Print PICKS section
    print("🎯 PICKS - AI Betting Recommendations")
    print("-" * 100)
    print()
    
    for pick in picks:
        print(f"#{pick['pick_number']}. {pick['player']}")
        print(f"   Prop: {pick['prop_type']}")
        print(f"   Recommendation: {pick['recommendation']} ({pick['confidence']} Confidence)")
        print()
        
        if pick['reasoning']:
            print(f"   💡 Reasoning:")
            # Use textwrap for proper formatting
            wrapped_reasoning = textwrap.fill(pick['reasoning'], width=90, initial_indent="      • ", subsequent_indent="        ")
            print(wrapped_reasoning)
            print()
        
        if pick['risk_factors']:
            print(f"   ⚠️  Risk Factors:")
            # Use textwrap for proper formatting
            wrapped_risks = textwrap.fill(pick['risk_factors'], width=90, initial_indent="      • ", subsequent_indent="        ")
            print(wrapped_risks)
            print()
        
        print("-" * 100)
        print()
    
    # Print STATS section
    print("📊 STATS - Historical Performance Insights")
    print("-" * 100)
    print()
    
    for stat in stats:
        if stat['player']:
            print(f"#{stat['pick_number']}. {stat['player']}")
        else:
            print(f"#{stat['pick_number']}. Historical Insight")
        
        if stat['reasoning']:
            # Use textwrap for proper formatting
            wrapped_insight = textwrap.fill(stat['reasoning'], width=90, initial_indent="   📈 ", subsequent_indent="       ")
            print(wrapped_insight)
        print()
    
    print("=" * 100)


def export_to_csv(week_number: int, picks: List[Dict], stats: List[Dict], output_file: str = None) -> str:
    """
    Export picks and stats to CSV file.
    
    Args:
        week_number: NFL week number
        picks: List of pick dictionaries
        stats: List of stat dictionaries
        output_file: Optional output file path
        
    Returns:
        Path to the created CSV file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"nfl_week_{week_number}_analysis_{timestamp}.csv"
    
    # Combine picks and stats
    all_data = picks + stats
    
    # Define CSV headers
    headers = [
        'pick_number',
        'type',
        'player',
        'prop_type',
        'line',
        'recommendation',
        'confidence',
        'reasoning',
        'risk_factors'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for row in all_data:
            # Clean up the data for CSV
            clean_row = {}
            for header in headers:
                value = row.get(header, '')
                # Remove newlines and extra whitespace for CSV
                if isinstance(value, str):
                    value = re.sub(r'\s+', ' ', value.strip())
                clean_row[header] = value
            writer.writerow(clean_row)
    
    return output_file


def export_to_html(week_number: int, picks: List[Dict], stats: List[Dict], output_file: str = None) -> str:
    """
    Export picks and stats to HTML file with Tailwind CSS styling.
    
    Args:
        week_number: NFL week number
        picks: List of pick dictionaries
        stats: List of stat dictionaries
        output_file: Optional output file path
        
    Returns:
        Path to the created HTML file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"nfl_week_{week_number}_analysis_{timestamp}.html"
    
    # HTML template with Tailwind CSS
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Week {week_number} Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        'nfl-blue': '#013369',
                        'nfl-red': '#D50A0A',
                        'nfl-gold': '#FFB612'
                    }}
                }}
            }}
        }}
    </script>
    <style>
        .gradient-bg {{
            background: linear-gradient(135deg, #013369 0%, #D50A0A 100%);
        }}
        .pick-card {{
            transition: transform 0.2s ease-in-out;
        }}
        .pick-card:hover {{
            transform: translateY(-2px);
        }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <div class="gradient-bg text-white py-8">
        <div class="container mx-auto px-4">
            <h1 class="text-4xl font-bold text-center mb-2">NFL Week {week_number} Analysis</h1>
            <p class="text-center text-lg opacity-90">AI-Powered Betting Recommendations & Insights</p>
            <div class="text-center mt-4 text-sm opacity-75">
                Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4 py-8 max-w-6xl">
"""
    
    # Add picks section
    if picks:
        html_content += f"""
        <!-- Picks Section -->
        <div class="mb-12">
            <div class="flex items-center mb-6">
                <div class="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mr-4">
                    <span class="text-white text-xl">🎯</span>
                </div>
                <h2 class="text-3xl font-bold text-gray-800">PICKS - AI Betting Recommendations</h2>
            </div>
            
            <div class="grid gap-6">
"""
        
        for pick in picks:
            # Determine confidence color
            confidence_color = "green" if "high" in pick['confidence'].lower() else "yellow" if "medium" in pick['confidence'].lower() else "red"
            
            html_content += f"""
                <div class="pick-card bg-white rounded-lg shadow-lg border-l-4 border-{confidence_color}-500 p-6">
                    <div class="flex items-start justify-between mb-4">
                        <div>
                            <h3 class="text-2xl font-bold text-gray-800">#{pick['pick_number']}. {pick['player']}</h3>
                            <div class="text-lg text-gray-600 mt-1">
                                <span class="font-semibold">{pick['prop_type']}</span>
                                {f" - Line: {pick['line']}" if pick['line'] else ""}
                            </div>
                        </div>
                        <div class="text-right">
                            <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-{confidence_color}-100 text-{confidence_color}-800">
                                {pick['recommendation']} ({pick['confidence']} Confidence)
                            </span>
                        </div>
                    </div>
"""
            
            if pick['reasoning']:
                html_content += f"""
                    <div class="mb-4">
                        <h4 class="text-lg font-semibold text-gray-700 mb-2 flex items-center">
                            <span class="mr-2">💡</span> Reasoning
                        </h4>
                        <p class="text-gray-600 leading-relaxed">{pick['reasoning']}</p>
                    </div>
"""
            
            if pick['risk_factors']:
                html_content += f"""
                    <div>
                        <h4 class="text-lg font-semibold text-gray-700 mb-2 flex items-center">
                            <span class="mr-2">⚠️</span> Risk Factors
                        </h4>
                        <p class="text-gray-600 leading-relaxed">{pick['risk_factors']}</p>
                    </div>
"""
            
            html_content += """
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    # Add stats section
    if stats:
        html_content += f"""
        <!-- Stats Section -->
        <div class="mb-12">
            <div class="flex items-center mb-6">
                <div class="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mr-4">
                    <span class="text-white text-xl">📊</span>
                </div>
                <h2 class="text-3xl font-bold text-gray-800">STATS - Historical Performance Insights</h2>
            </div>
            
            <div class="grid gap-4">
"""
        
        for stat in stats:
            html_content += f"""
                <div class="bg-white rounded-lg shadow-md border-l-4 border-blue-500 p-6">
                    <div class="flex items-start">
                        <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-4 flex-shrink-0">
                            <span class="text-blue-600 font-bold">#{stat['pick_number']}</span>
                        </div>
                        <div class="flex-1">
                            <h3 class="text-xl font-bold text-gray-800 mb-2">
                                {stat['player'] if stat['player'] else 'Historical Insight'}
                            </h3>
                            <p class="text-gray-600 leading-relaxed flex items-start">
                                <span class="mr-2 text-blue-500">📈</span>
                                {stat['reasoning']}
                            </p>
                        </div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    # Footer
    html_content += f"""
        <!-- Footer -->
        <div class="mt-12 pt-8 border-t border-gray-200">
            <div class="text-center text-gray-500">
                <p class="text-sm">Generated by NFL Analysis System</p>
                <p class="text-xs mt-1">This analysis is for informational purposes only. Please bet responsibly.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file


def main(week_number: int, export_csv: bool = False, export_html: bool = False) -> None:
    """
    Main function to format and display insights for a specific week.
    
    Args:
        week_number: NFL week number to display insights for
        export_csv: Whether to export to CSV file
        export_html: Whether to export to HTML file
    """
    try:
        # Construct file paths
        grok_file = f"data/insights/grok_insights_week_{week_number:02d}.json"
        stats_file = f"data/fun_stats/stats_insights_week_{week_number:02d}.json"
        
        picks = []
        stats = []
        
        # Load and process GROK insights
        try:
            grok_data = load_json_file(grok_file)
            if 'ai_analysis' in grok_data and 'analysis' in grok_data['ai_analysis']:
                analysis_text = grok_data['ai_analysis']['analysis']
                picks = extract_picks_from_analysis(analysis_text)
                print(f"[SUCCESS] Loaded {len(picks)} picks from GROK analysis")
            else:
                print(f"[WARNING] No analysis found in {grok_file}")
        except FileNotFoundError:
            print(f"[WARNING] GROK insights file not found: {grok_file}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error parsing GROK insights file: {e}")
        
        # Load and process stats insights
        try:
            stats_data = load_json_file(stats_file)
            if 'insights' in stats_data and stats_data['insights']:
                insights_list = stats_data['insights']
                
                # Check if insights is a list of dictionaries (new format) or a JSON string (old format)
                if isinstance(insights_list, list) and len(insights_list) > 0:
                    if isinstance(insights_list[0], dict):
                        # New format: list of dictionaries
                        stats = extract_stats_from_insights(insights_list)
                        print(f"[SUCCESS] Loaded {len(stats)} stats from insights (new format)")
                    else:
                        # Old format: list containing JSON string
                        insights_str = insights_list[0]
                        try:
                            # Try to parse the JSON string
                            parsed_insights = json.loads(insights_str)
                            stats = extract_stats_from_insights(parsed_insights)
                            print(f"[SUCCESS] Loaded {len(stats)} stats from insights (old format)")
                        except json.JSONDecodeError:
                            # If it's not valid JSON, try to extract insights manually
                            print(f"[WARNING] Could not parse insights JSON, trying manual extraction...")
                            # Look for insight patterns in the text
                            insight_pattern = r'"insight":\s*"([^"]+)"'
                            matches = re.findall(insight_pattern, insights_str)
                            stats = []
                            for i, match in enumerate(matches, 1):
                                stat_data = {
                                    'pick_number': i,
                                    'type': 'STAT',
                                    'player': '',
                                    'prop_type': '',
                                    'line': '',
                                    'recommendation': '',
                                    'confidence': '',
                                    'reasoning': match,
                                    'risk_factors': ''
                                }
                                stats.append(stat_data)
                            print(f"[SUCCESS] Loaded {len(stats)} stats from manual extraction")
                else:
                    print(f"[WARNING] No insights found in {stats_file}")
            else:
                print(f"[WARNING] No insights found in {stats_file}")
        except FileNotFoundError:
            print(f"[WARNING] Stats insights file not found: {stats_file}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error parsing stats insights file: {e}")
        
        # Print formatted analysis
        if picks or stats:
            print_formatted_analysis(week_number, picks, stats)
            
            # Export to CSV if requested
            if export_csv:
                csv_file = export_to_csv(week_number, picks, stats)
                print(f"[CSV] Exported to CSV: {csv_file}")
            
            # Export to HTML if requested
            if export_html:
                html_file = export_to_html(week_number, picks, stats)
                print(f"[HTML] Exported to HTML: {html_file}")
                print(f"   Open the HTML file in your browser to view the formatted analysis!")
        else:
            print("[ERROR] No data found to display")
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python insights_formatter.py <week_number> [--csv] [--html]")
        print("Example: python insights_formatter.py 2")
        print("Example: python insights_formatter.py 2 --csv")
        print("Example: python insights_formatter.py 2 --html")
        print("Example: python insights_formatter.py 2 --csv --html")
        sys.exit(1)
    
    try:
        week_number = int(sys.argv[1])
        export_csv = '--csv' in sys.argv
        export_html = '--html' in sys.argv
        
        main(week_number, export_csv, export_html)
    except ValueError:
        print("[ERROR] Week number must be an integer")
        print("Usage: python insights_formatter.py <week_number> [--csv] [--html]")
        sys.exit(1)
