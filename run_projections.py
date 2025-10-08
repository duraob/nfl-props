"""
Main entry point for running NFL projections using the new modular projection engine.

This script provides the same interface as the old projection.py but uses the new
modular projection engine with enhanced ML capabilities and Monte Carlo simulations.

Usage:
    python run_projections.py                    # Run Week 1 projections
    python run_projections.py 2                  # Run Week 2 projections  
    python run_projections.py --week 3           # Run Week 3 projections
    python run_projections.py --help             # Show help
"""

import sys
import os
from datetime import datetime
from projection_engine import ProjectionEngine

def print_help():
    """Print usage information."""
    print("NFL Projection Engine - Main Entry Point")
    print("=" * 50)
    print("Usage:")
    print("  python run_projections.py                    # Run Week 1 projections")
    print("  python run_projections.py 2                  # Run Week 2 projections")
    print("  python run_projections.py --week 3           # Run Week 3 projections")
    print("  python run_projections.py --help             # Show this help")
    print()
    print("Features:")
    print("  • Monte Carlo simulations with 10,000 iterations")
    print("  • Advanced ML variance analysis")
    print("  • Probability distributions and confidence ratings")
    print("  • Home/away splits and weather adjustments")
    print("  • Active roster filtering with injury reports")
    print("  • Time-weighted historical data analysis")
    print()

def run_projections(week=1, n_simulations=10000):
    """
    Run projections for a specific week using the new modular engine.
    
    Args:
        week: Week number for projections (default: 1)
        n_simulations: Number of Monte Carlo simulations (default: 10000)
    """
    print("🚀 NFL Projection Engine - Main Entry Point")
    print("=" * 60)
    print(f"📅 Generating projections for Week {week}")
    print(f"🎲 Running {n_simulations:,} Monte Carlo simulations")
    print(f"⏰ Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize the projection engine
        engine = ProjectionEngine(
            n_simulations=n_simulations,
            random_seed=42  # For reproducible results
        )
        
        # Run projections
        results = engine.run_projections(
            week_2025_file="data/game_data/game_data_2025.csv",
            weeks_2024_file="data/game_data/game_data_2024.csv", 
            target_weeks_2024=[9, 10, 11, 12, 13, 14, 15, 16, 17],
            projection_week=week,
            schedule_file="data/nfl-2025-EasternStandardTime.csv",
            roster_file="data/roster.xlsx"
        )
        
        if results:
            print("\n✅ Projections completed successfully!")
            print(f"📊 Generated projections for {len(results['base_projections'])} players")
            
            # Check which results are available
            if 'confidence_matrix' in results:
                print(f"📈 Confidence ratings: {len(results['confidence_matrix'])} players analyzed")
            if 'probability_distributions' in results:
                print(f"🎯 Probability distributions: {len(results['probability_distributions'])} players")
            
            # Show sample results
            if 'base_projections' in results and not results['base_projections'].empty:
                print("\n📋 Sample Projections (Top 5 Players by Passing Yards):")
                sample_cols = ['name', 'pos', 'pass_yd', 'rush_yd', 'rec_yd', 'pass_td', 'rush_td', 'rec_td']
                available_cols = [col for col in sample_cols if col in results['base_projections'].columns]
                
                if 'pass_yd' in results['base_projections'].columns:
                    top_passers = results['base_projections'].nlargest(5, 'pass_yd')[available_cols]
                    print(top_passers.to_string(index=False))
                else:
                    print(results['base_projections'][available_cols].head().to_string(index=False))
            
            print(f"\n💾 Results saved to data/projections/")
            print(f"   • nfl25_proj_week{week}.csv - Base projections")
            
        else:
            print("❌ No results generated. Check data availability and try again.")
            return False
            
    except Exception as e:
        print(f"❌ Error running projections: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n⏰ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True

def main():
    """Main function to handle command line arguments and run projections."""
    week = 1  # Default week
    n_simulations = 10000  # Default simulations
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print_help()
            return
        elif sys.argv[1] == "--week":
            # Format: python run_projections.py --week 2
            if len(sys.argv) > 2:
                try:
                    week = int(sys.argv[2])
                except ValueError:
                    print(f"❌ Invalid week number: {sys.argv[2]}")
                    print("Usage: python run_projections.py --week <week_number>")
                    return
            else:
                print("❌ Missing week number")
                print("Usage: python run_projections.py --week <week_number>")
                return
        elif sys.argv[1].startswith("--"):
            # Unknown option
            print(f"❌ Unknown option: {sys.argv[1]}")
            print_help()
            return
        else:
            # Assume first argument is week number: python run_projections.py 2
            try:
                week = int(sys.argv[1])
            except ValueError:
                print(f"❌ Invalid week number: {sys.argv[1]}")
                print("Usage: python run_projections.py [week_number]")
                return
    
    # Validate week number
    if week < 1 or week > 18:
        print(f"❌ Invalid week number: {week}")
        print("Week must be between 1 and 18")
        return
    
    # Check if required data files exist
    required_files = [
        "data/game_data/game_data_2025.csv",
        "data/game_data/game_data_2024.csv", 
        "data/nfl-2025-EasternStandardTime.csv",
        "data/roster.xlsx"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease ensure all data files are available before running projections.")
        return
    
    # Run projections
    success = run_projections(week, n_simulations)
    
    if success:
        print("\n🎉 Projection generation complete!")
    else:
        print("\n💥 Projection generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
