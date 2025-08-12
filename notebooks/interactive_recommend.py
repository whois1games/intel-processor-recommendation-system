# %%
# Interactive Intel Processor Recommendation System

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------
# CONFIG
# -------------------
FEATURES_FILE = "../data/intel_processors_features.csv"
TOP_N = 5  # default recommendations count

# -------------------
# LOAD DATA
# -------------------
try:
    df = pd.read_csv(FEATURES_FILE)
    print(f"✅ Loaded features: {df.shape[0]} processors, {df.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Could not find file: {FEATURES_FILE}")

# -------------------
# PREPARE FEATURES
# -------------------
id_cols = ["processor_name", "category"]
if "feat.vertical_segment" in df.columns:
    id_cols.append("feat.vertical_segment")

numeric_cols = [c for c in df.columns if c.startswith("feat.") and c not in ("feat.vertical_segment",)]
numeric_df = df[numeric_cols]

# Normalize features
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_df)

# Compute cosine similarity
similarity_matrix = cosine_similarity(numeric_scaled)

# -------------------
# UTILITY FUNCTIONS
# -------------------
def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    """Display the application header"""
    print("=" * 60)
    print("🚀 INTEL PROCESSOR RECOMMENDATION SYSTEM")
    print("=" * 60)

def get_user_preferences():
    """Get user preferences for processor recommendation"""
    print("\n📝 Please answer a few questions to get personalized recommendations:")
    print("-" * 50)
    
    preferences = {}
    
    # Processor family/type search
    print("\n1. What type of Intel processor are you looking for?")
    print("   Available families: 'Core 3', 'Core 5', 'Core 7', 'Core 9', 'Core Ultra', 'Xeon'")
    print("   Legacy names: 'i3', 'i5', 'i7', 'i9' (will be mapped to new naming)")
    print("   Or enter 'any' to see all processors")
    
    processor_type = input("\n🔍 Enter processor type: ").strip()
    preferences['processor_type'] = processor_type if processor_type.lower() != 'any' else ''
    
    # Budget/Price Range
    print("\n2. What's your budget range?")
    print("   a) Budget ($0 - $300)")
    print("   b) Mid-range ($300 - $600)")
    print("   c) High-end ($600 - $1000)")
    print("   d) Premium ($1000 - $2000)")
    print("   e) Ultra Premium ($2000+)")
    print("   f) Custom range")
    print("   g) No budget constraint")
    
    budget = input("\nEnter your choice (a-g): ").strip().lower()
    
    if budget == 'f':
        try:
            min_price = float(input("Enter minimum price ($): "))
            max_price = float(input("Enter maximum price ($): "))
            preferences['budget'] = (min_price, max_price)
        except ValueError:
            print("Invalid price range, using no constraint")
            preferences['budget'] = (0, float('inf'))
    else:
        budget_map = {
            'a': (0, 300), 'b': (300, 600), 'c': (600, 1000),
            'd': (1000, 2000), 'e': (2000, float('inf')), 'g': (0, float('inf'))
        }
        preferences['budget'] = budget_map.get(budget, (0, float('inf')))
    
    # Usage type
    print("\n3. What will you primarily use this processor for?")
    print("   a) Gaming")
    print("   b) Content Creation (Video editing, 3D rendering)")
    print("   c) Office Work (Documents, browsing, light tasks)")
    print("   d) Programming/Development")
    print("   e) Server/Enterprise applications")
    
    usage = input("\nEnter your choice (a-e): ").strip().lower()
    usage_map = {
        'a': 'Gaming', 'b': 'Content Creation', 'c': 'Office Work',
        'd': 'Programming', 'e': 'Server/Enterprise'
    }
    preferences['usage'] = usage_map.get(usage, 'General Use')
    
    # Performance priority
    print("\n4. What's most important to you?")
    print("   a) Single-core performance (Gaming, general use)")
    print("   b) Multi-core performance (Content creation, programming)")
    print("   c) Power efficiency (Laptops, battery life)")
    print("   d) Balanced performance")
    
    priority = input("\nEnter your choice (a-d): ").strip().lower()
    priority_map = {
        'a': 'single_core', 'b': 'multi_core', 'c': 'efficiency', 'd': 'balanced'
    }
    preferences['priority'] = priority_map.get(priority, 'balanced')
    
    return preferences

def filter_by_preferences(df, preferences):
    """Filter processors based on user preferences"""
    filtered_df = df.copy()
    
    # Filter by processor type if specified
    if preferences.get('processor_type') and preferences['processor_type']:
        search_term = preferences['processor_type']
        
        # Map old Intel naming to new naming
        search_mapping = {
            'i3': 'Core™ 3',
            'i5': 'Core™ 5', 
            'i7': 'Core™ 7',
            'i9': 'Core™ 9',
            'core 3': 'Core™ 3',
            'core 5': 'Core™ 5',
            'core 7': 'Core™ 7', 
            'core 9': 'Core™ 9',
            'core ultra': 'Core™ Ultra',
            'ultra': 'Core™ Ultra'
        }
        
        # Check if we need to map the search term
        mapped_term = search_mapping.get(search_term.lower(), search_term)
        
        # Try both original and mapped terms
        filtered_df = filtered_df[
            filtered_df['processor_name'].str.contains(mapped_term, case=False, na=False) |
            filtered_df['processor_name'].str.contains(search_term, case=False, na=False)
        ]
        
        print(f"🔍 Searching for '{search_term}' (mapped to '{mapped_term}'): {len(filtered_df)} processors found")
    
    # Filter by budget
    if 'budget' in preferences:
        min_price, max_price = preferences['budget']
        filtered_df = filtered_df[
            (filtered_df['feat.price_usd'] >= min_price) & 
            (filtered_df['feat.price_usd'] <= max_price)
        ]
        if max_price == float('inf'):
            print(f"💰 Filtered by budget: ${min_price:,.0f}+ ({len(filtered_df)} processors)")
        else:
            print(f"💰 Filtered by budget: ${min_price:,.0f} - ${max_price:,.0f} ({len(filtered_df)} processors)")
    
    if filtered_df.empty:
        return filtered_df
    
    # Add preference-based scoring
    if preferences.get('priority') == 'single_core':
        filtered_df['pref_score'] = filtered_df['feat.max_turbo_ghz']
    elif preferences.get('priority') == 'multi_core':
        filtered_df['pref_score'] = (filtered_df['feat.total_cores'] * 
                                    filtered_df['feat.max_turbo_ghz'])
    elif preferences.get('priority') == 'efficiency':
        filtered_df['pref_score'] = filtered_df['feat.freq_per_watt']
    else:  # balanced
        filtered_df['pref_score'] = (
            filtered_df['feat.max_turbo_ghz'] * 0.3 +
            filtered_df['feat.total_cores'] * 0.3 +
            filtered_df['feat.freq_per_watt'] * 0.4
        )
    
    return filtered_df.sort_values('pref_score', ascending=False)

def display_recommendations(recommendations, preferences):
    """Display formatted recommendations"""
    print(f"\n🎯 TOP {len(recommendations)} RECOMMENDATIONS FOR YOU")
    print("=" * 80)
    
    # Show search criteria
    if preferences.get('processor_type'):
        print(f"🔍 Processor type: {preferences['processor_type']}")
    
    min_price, max_price = preferences.get('budget', (0, float('inf')))
    if max_price == float('inf'):
        print(f"💰 Budget: ${min_price:,.0f}+")
    else:
        print(f"💰 Budget: ${min_price:,.0f} - ${max_price:,.0f}")
    
    print(f"🎯 Usage: {preferences.get('usage', 'General')}")
    print(f"⚡ Priority: {preferences.get('priority', 'balanced').replace('_', ' ').title()}")
    print("=" * 80)
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"\n🏆 RANK #{idx}")
        print("-" * 40)
        
        # Shorten processor name if too long
        name = row['processor_name']
        if len(name) > 60:
            name = name[:57] + "..."
        
        print(f"📱 Processor: {name}")
        print(f"🏷️  Category: {row['category']}")
        print(f"💰 Price: ${row['feat.price_usd']:,.0f}")
        print(f"⚡ Max Turbo: {row['feat.max_turbo_ghz']:.1f} GHz")
        print(f"🔧 Cores/Threads: {int(row['feat.total_cores'])}/{int(row['feat.total_threads'])}")
        print(f"💾 Cache: {row['feat.cache_mb']:.0f} MB")
        print(f"🔋 Base Power: {row['feat.base_power_w']:.0f}W")
        print(f"🎮 Graphics: {row['feat.gfx_max_dyn_ghz']:.2f} GHz")
        print(f"📊 Performance Score: {row['pref_score']:.2f}")
        
        # Add value rating
        value_score = row['pref_score'] / (row['feat.price_usd'] / 1000)  # Performance per $1000
        print(f"💯 Value Rating: {value_score:.2f}")
        
        # Add usage recommendations
        print(f"✅ Best for: {get_usage_recommendation(row, preferences)}")
        
        # Add price category
        price = row['feat.price_usd']
        if price < 300:
            price_cat = "💚 Budget-friendly"
        elif price < 600:
            price_cat = "💙 Mid-range value"
        elif price < 1000:
            price_cat = "💜 High-performance"
        elif price < 2000:
            price_cat = "🧡 Premium choice"
        else:
            price_cat = "❤️ Ultra-premium"
        
        print(f"🏷️  Price Category: {price_cat}")

def get_usage_recommendation(row, preferences):
    """Get usage recommendation based on processor specs"""
    cores = row['feat.total_cores']
    turbo = row['feat.max_turbo_ghz']
    efficiency = row['feat.freq_per_watt']
    
    recommendations = []
    
    if turbo >= 4.5:
        recommendations.append("Gaming")
    if cores >= 8:
        recommendations.append("Content Creation")
    if efficiency >= 0.2:
        recommendations.append("Power Efficiency")
    if cores >= 6 and turbo >= 4.0:
        recommendations.append("Programming")
    
    return ", ".join(recommendations) if recommendations else "General Use"

def search_by_name():
    """Search for processors by name with better examples"""
    print("\n🔍 SEARCH BY PROCESSOR NAME")
    print("-" * 30)
    
    # Show processor family examples
    print("\n📋 Search Examples:")
    print("   • For Intel Core: 'Core 3', 'Core 5', 'Core 7', 'Core 9'")
    print("   • For older naming: 'i3', 'i5', 'i7', 'i9'")
    print("   • For premium: 'Core Ultra', 'Ultra 5', 'Ultra 7'")
    print("   • For servers: 'Xeon', 'Xeon Max'")
    print("   • For specific models: '13700K', '14900', '12600'")
    
    # Show actual processor samples by category
    print("\n💡 Sample processors in database:")
    categories = df['category'].unique()
    for cat in categories[:3]:  # Show first 3 categories
        sample = df[df['category'] == cat]['processor_name'].iloc[0]
        # Extract key terms from the sample
        if 'Core™ 3' in sample:
            key_term = "Core 3"
        elif 'Core™ 5' in sample:
            key_term = "Core 5"
        elif 'Core™ 7' in sample:
            key_term = "Core 7"
        elif 'Core™ 9' in sample:
            key_term = "Core 9"
        elif 'Xeon' in sample:
            key_term = "Xeon"
        else:
            key_term = sample.split()[1:3]  # Take first 2 words after Intel
            key_term = ' '.join(key_term) if isinstance(key_term, list) else str(key_term)
        
        print(f"   • {cat}: Try '{key_term}'")
    
    search_term = input("\n🔍 Enter search term (short name like 'Core 5' or 'i7'): ").strip()
    
    if not search_term:
        print("❌ Please enter a search term!")
        return
    
    # Search for processors containing the search term
    matching = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
    
    if matching.empty:
        print(f"❌ No processors found containing '{search_term}'")
        print("\n💡 Try these suggestions:")
        print("   • Use shorter terms like 'Core 5' instead of full names")
        print("   • Try 'i5', 'i7', 'i9' for classic naming")
        print("   • Use 'Xeon' for server processors")
        return
    
    print(f"\n✅ Found {len(matching)} processors containing '{search_term}':")
    print("-" * 70)
    
    # Group by price ranges for better organization
    price_ranges = [
        (0, 300, "💚 Budget"),
        (300, 600, "💙 Mid-range"), 
        (600, 1000, "💜 High-end"),
        (1000, 2000, "🧡 Premium"),
        (2000, float('inf'), "❤️ Ultra Premium")
    ]
    
    for min_p, max_p, label in price_ranges:
        range_processors = matching[
            (matching['feat.price_usd'] >= min_p) & 
            (matching['feat.price_usd'] < max_p)
        ].sort_values('feat.price_usd')
        
        if not range_processors.empty:
            print(f"\n{label} (${min_p:,.0f}{'+' if max_p == float('inf') else f' - ${max_p:,.0f}'}):")
            for idx, (_, row) in enumerate(range_processors.head(5).iterrows(), 1):
                cores = int(row['feat.total_cores'])
                threads = int(row['feat.total_threads'])
                turbo = row['feat.max_turbo_ghz']
                price = row['feat.price_usd']
                
                # Shorten processor name for display
                name = row['processor_name']
                if len(name) > 50:
                    name = name[:47] + "..."
                
                print(f"  {idx}. {name}")
                print(f"     💰 ${price:,.0f} | ⚡ {turbo:.1f}GHz | 🔧 {cores}C/{threads}T")
            
            if len(range_processors) > 5:
                print(f"     ... and {len(range_processors) - 5} more in this range")
    
    # Let user select one for detailed view
    try:
        print(f"\nTotal processors found: {len(matching)}")
        choice = input("Enter processor number for detailed specs (or press Enter to skip): ").strip()
        
        if choice:
            choice_num = int(choice) - 1
            all_matching = matching.sort_values('feat.price_usd').reset_index(drop=True)
            if 0 <= choice_num < len(all_matching):
                selected = all_matching.iloc[choice_num]
                show_detailed_specs(selected)
            else:
                print("❌ Invalid selection!")
                
    except ValueError:
        print("❌ Please enter a valid number!")

def show_detailed_specs(processor):
    """Show detailed specifications for a selected processor"""
    print(f"\n📋 DETAILED SPECIFICATIONS")
    print("=" * 50)
    print(f"📱 Processor: {processor['processor_name']}")
    print(f"🏷️  Category: {processor['category']}")
    print(f"🎯 Segment: {processor['feat.vertical_segment']}")
    print(f"💰 Price: ${processor['feat.price_usd']:,.0f}")
    print(f"⚡ Base Frequency: {processor['feat.base_freq_ghz']:.1f} GHz")
    print(f"🚀 Max Turbo: {processor['feat.max_turbo_ghz']:.1f} GHz")
    print(f"🔧 Cores: {int(processor['feat.total_cores'])}")
    print(f"🧵 Threads: {int(processor['feat.total_threads'])}")
    print(f"💾 Cache: {processor['feat.cache_mb']:.0f} MB")
    print(f"🔋 Base Power: {processor['feat.base_power_w']:.0f}W")
    print(f"⚡ Turbo Power: {processor['feat.turbo_power_w']:.0f}W")
    print(f"🧠 Max Memory: {processor['feat.max_mem_gb']:.0f} GB")
    print(f"🎮 Graphics Max: {processor['feat.gfx_max_dyn_ghz']:.2f} GHz")
    print(f"🔲 Execution Units: {int(processor['feat.execution_units'])}")
    print(f"⚡ Freq/Watt: {processor['feat.freq_per_watt']:.3f}")
    print(f"🔧 Cores/Watt: {processor['feat.cores_per_watt']:.3f}")
    print(f"💾 Cache/Core: {processor['feat.cache_per_core']:.2f} MB")

def compare_processors():
    """Compare two processors side by side"""
    print("\n🔄 PROCESSOR COMPARISON")
    print("-" * 30)
    print("💡 Use short search terms like: 'Core 5', 'i7', 'Xeon', '13700', etc.")
    
    processors = []
    for i in range(2):
        print(f"\n🔍 Select Processor #{i+1}:")
        search_term = input(f"Search term (e.g., 'Core 7', 'i9'): ").strip()
        
        if not search_term:
            print("❌ Please enter a search term!")
            return
        
        matching = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
        
        if matching.empty:
            print(f"❌ No processors found containing '{search_term}'")
            print("💡 Try shorter terms like 'Core 5', 'i7', or model numbers")
            return
        
        # Sort by price for better organization
        matching = matching.sort_values('feat.price_usd')
        
        print(f"\n✅ Found {len(matching)} processors with '{search_term}':")
        for idx, (_, row) in enumerate(matching.head(10).iterrows(), 1):
            price = row['feat.price_usd']
            cores = int(row['feat.total_cores'])
            turbo = row['feat.max_turbo_ghz']
            
            # Shorten name
            name = row['processor_name']
            if len(name) > 45:
                name = name[:42] + "..."
            
            print(f"{idx:2}. {name}")
            print(f"    💰 ${price:,.0f} | ⚡ {turbo:.1f}GHz | 🔧 {cores} cores")
        
        if len(matching) > 10:
            print(f"    ... and {len(matching) - 10} more")
        
        try:
            choice = int(input(f"\nSelect processor #{i+1} (1-{min(10, len(matching))}): ")) - 1
            if 0 <= choice < len(matching):
                processors.append(matching.iloc[choice])
            else:
                print("❌ Invalid selection!")
                return
        except ValueError:
            print("❌ Please enter a valid number!")
            return
    
    # Display comparison
    proc1, proc2 = processors[0], processors[1]
    name1 = proc1['processor_name'][:30] + "..." if len(proc1['processor_name']) > 30 else proc1['processor_name']
    name2 = proc2['processor_name'][:30] + "..." if len(proc2['processor_name']) > 30 else proc2['processor_name']
    
    print(f"\n🔄 COMPARISON")
    print("=" * 80)
    print(f"Processor 1: {name1}")
    print(f"Processor 2: {name2}")
    print("=" * 80)
    
    specs = [
        ('💰 Price', 'feat.price_usd', '$', 'lower_better'),
        ('⚡ Max Turbo', 'feat.max_turbo_ghz', ' GHz', 'higher_better'),
        ('🔧 Cores', 'feat.total_cores', '', 'higher_better'),
        ('🧵 Threads', 'feat.total_threads', '', 'higher_better'),
        ('💾 Cache', 'feat.cache_mb', ' MB', 'higher_better'),
        ('🔋 Base Power', 'feat.base_power_w', 'W', 'lower_better'),
        ('🎮 Graphics', 'feat.gfx_max_dyn_ghz', ' GHz', 'higher_better'),
        ('⚡ Efficiency', 'feat.freq_per_watt', '', 'higher_better'),
        ('💯 Value (Score/$1K)', 'calculated', '', 'higher_better'),
    ]
    
    for spec_name, col, unit, better in specs:
        if col == 'calculated':
            # Calculate value score
            val1 = (proc1['feat.max_turbo_ghz'] + proc1['feat.total_cores']) / (proc1['feat.price_usd'] / 1000)
            val2 = (proc2['feat.max_turbo_ghz'] + proc2['feat.total_cores']) / (proc2['feat.price_usd'] / 1000)
        else:
            val1 = proc1[col]
            val2 = proc2[col]
        
        if better == 'lower_better':
            winner1 = "🟢" if val1 < val2 else "🔴" if val1 > val2 else "🟡"
            winner2 = "🔴" if val1 < val2 else "🟢" if val1 > val2 else "🟡"
        else:
            winner1 = "🟢" if val1 > val2 else "🔴" if val1 < val2 else "🟡"
            winner2 = "🔴" if val1 > val2 else "🟢" if val1 < val2 else "🟡"
        
        print(f"{spec_name:20} {winner1} {val1:8.1f}{unit:>5} vs {winner2} {val2:8.1f}{unit:>5}")
    
    # Overall recommendation
    print("\n🏆 RECOMMENDATION:")
    if proc1['feat.price_usd'] < proc2['feat.price_usd']:
        if (proc1['feat.max_turbo_ghz'] * proc1['feat.total_cores']) >= (proc2['feat.max_turbo_ghz'] * proc2['feat.total_cores']) * 0.9:
            print(f"💚 Processor 1 offers better value for money!")
        else:
            print(f"💙 Processor 1 is more budget-friendly, Processor 2 has better performance")
    else:
        if (proc2['feat.max_turbo_ghz'] * proc2['feat.total_cores']) >= (proc1['feat.max_turbo_ghz'] * proc1['feat.total_cores']) * 0.9:
            print(f"💚 Processor 2 offers better value for money!")
        else:
            print(f"💙 Processor 2 is more budget-friendly, Processor 1 has better performance")

def main_menu():
    """Display main menu and handle user choices"""
    while True:
        display_header()
        print("\n📋 MAIN MENU")
        print("-" * 20)
        print("1. 🎯 Get personalized recommendations")
        print("2. 🔍 Search by processor name")
        print("3. 📊 Browse all processors by category")
        print("4. 🔄 Compare two processors")
        print("5. 🚪 Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            clear_screen()
            preferences = get_user_preferences()
            filtered_df = filter_by_preferences(df, preferences)
            
            if filtered_df.empty:
                print("\n❌ No processors found matching your criteria.")
                print("\n💡 Suggestions:")
                print("   • Try a broader price range (e.g., 'No budget constraint')")
                print("   • Use different processor type (e.g., 'Core 5' instead of 'i7')")
                print("   • Check available families: Core 3, Core 5, Core 7, Core 9, Core Ultra, Xeon")
                
                # Show what's available in their budget
                if 'budget' in preferences:
                    min_price, max_price = preferences['budget']
                    budget_processors = df[
                        (df['feat.price_usd'] >= min_price) & 
                        (df['feat.price_usd'] <= max_price)
                    ]
                    if len(budget_processors) > 0:
                        print(f"\n📊 {len(budget_processors)} processors available in your budget:")
                        families_in_budget = set()
                        for name in budget_processors['processor_name']:
                            if 'Core™' in name:
                                try:
                                    family = name.split('Core™')[1].split()[0]
                                    families_in_budget.add(f"Core {family}")
                                except:
                                    continue
                            elif 'Xeon' in name:
                                families_in_budget.add("Xeon")
                        print(f"   Available families: {', '.join(sorted(families_in_budget))}")
            else:
                recommendations = filtered_df.head(TOP_N)
                display_recommendations(recommendations, preferences)
            
            input("\nPress Enter to continue...")
            clear_screen()
            
        elif choice == '2':
            clear_screen()
            search_by_name()
            input("\nPress Enter to continue...")
            clear_screen()
            
        elif choice == '3':
            clear_screen()
            print("\n📊 PROCESSORS BY CATEGORY")
            print("-" * 30)
            categories = df['category'].unique()
            for i, cat in enumerate(categories, 1):
                count = len(df[df['category'] == cat])
                print(f"{i}. {cat} ({count} processors)")
            
            try:
                cat_choice = int(input(f"\nSelect category (1-{len(categories)}): ")) - 1
                if 0 <= cat_choice < len(categories):
                    selected_cat = categories[cat_choice]
                    cat_df = df[df['category'] == selected_cat].sort_values('feat.price_usd')
                    
                    print(f"\n📋 {selected_cat} Processors:")
                    print("-" * 50)
                    for idx, (_, row) in enumerate(cat_df.iterrows(), 1):
                        print(f"{idx:2}. {row['processor_name']} - ${row['feat.price_usd']:,.0f}")
                    
                    try:
                        proc_choice = int(input(f"\nSelect processor for details (1-{len(cat_df)}): ")) - 1
                        if 0 <= proc_choice < len(cat_df):
                            selected = cat_df.iloc[proc_choice]
                            show_detailed_specs(selected)
                    except ValueError:
                        print("❌ Invalid selection!")
                        
            except ValueError:
                print("❌ Invalid selection!")
            
            input("\nPress Enter to continue...")
            clear_screen()
            
        elif choice == '4':
            clear_screen()
            compare_processors()
            input("\nPress Enter to continue...")
            clear_screen()
            
        elif choice == '5':
            print("\n👋 Thank you for using Intel Processor Recommendation System!")
            print("Happy computing! 🚀")
            break
            
        else:
            print("❌ Invalid choice! Please select 1-5.")
            input("Press Enter to continue...")
            clear_screen()

# -------------------
# MAIN EXECUTION
# -------------------
if __name__ == "__main__":
    clear_screen()
    main_menu()
