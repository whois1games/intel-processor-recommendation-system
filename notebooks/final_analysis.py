import pandas as pd

# Analyze the dataset for README updates
df = pd.read_csv('../data/intel_processors_features.csv')

print("=== DATASET ANALYSIS FOR README ===")
print(f"Total processors: {len(df)}")
print(f"Price range: ${df['feat.price_usd'].min():,.0f} - ${df['feat.price_usd'].max():,.0f}")

# Analyze by processor families
print("\n=== PROCESSOR FAMILIES ===")
families = {}
for name in df['processor_name']:
    if 'Core™ 3' in name:
        families.setdefault('Core 3', []).append(name)
    elif 'Core™ 5' in name:
        families.setdefault('Core 5', []).append(name)
    elif 'Core™ 7' in name:
        families.setdefault('Core 7', []).append(name)
    elif 'Core™ 9' in name:
        families.setdefault('Core 9', []).append(name)
    elif 'Core™ Ultra' in name:
        families.setdefault('Core Ultra', []).append(name)
    elif 'Xeon' in name and 'Max' not in name:
        families.setdefault('Xeon', []).append(name)
    elif 'Xeon' in name and 'Max' in name:
        families.setdefault('Xeon Max', []).append(name)

for family, processors in families.items():
    family_df = df[df['processor_name'].isin(processors)]
    print(f"{family}: {len(processors)} processors")
    print(f"  Price range: ${family_df['feat.price_usd'].min():,.0f} - ${family_df['feat.price_usd'].max():,.0f}")

# Analyze by price ranges
print("\n=== PRICE RANGE ANALYSIS ===")
price_ranges = [
    (0, 300, "Budget"),
    (300, 600, "Mid-range"), 
    (600, 1000, "High-end"),
    (1000, 2000, "Premium"),
    (2000, float('inf'), "Ultra Premium")
]

for min_p, max_p, label in price_ranges:
    if max_p == float('inf'):
        range_df = df[df['feat.price_usd'] >= min_p]
        price_label = f"${min_p:,.0f}+"
    else:
        range_df = df[(df['feat.price_usd'] >= min_p) & (df['feat.price_usd'] < max_p)]
        price_label = f"${min_p:,.0f}-${max_p:,.0f}"
    
    print(f"{label} ({price_label}): {len(range_df)} processors")
    
    # Show what families are in this range
    if len(range_df) > 0:
        range_families = set()
        for name in range_df['processor_name']:
            if 'Core™ 3' in name:
                range_families.add('Core 3')
            elif 'Core™ 5' in name:
                range_families.add('Core 5')
            elif 'Core™ 7' in name:
                range_families.add('Core 7')
            elif 'Core™ 9' in name:
                range_families.add('Core 9')
            elif 'Core™ Ultra' in name:
                range_families.add('Core Ultra')
            elif 'Xeon' in name:
                range_families.add('Xeon')
        print(f"  Available families: {', '.join(sorted(range_families))}")

# Categories breakdown
print("\n=== CATEGORIES ===")
for category in df['category'].unique():
    cat_df = df[df['category'] == category]
    print(f"{category}: {len(cat_df)} processors")
    print(f"  Price range: ${cat_df['feat.price_usd'].min():,.0f} - ${cat_df['feat.price_usd'].max():,.0f}")

print("\n=== KEY FEATURES ===")
print("✅ Search mapping: i3→Core 3, i5→Core 5, i7→Core 7, i9→Core 9")
print("✅ Price filtering with 5 budget categories")
print("✅ Usage-based recommendations (Gaming, Content Creation, etc.)")
print("✅ Performance scoring algorithms")
print("✅ Processor comparison feature")
print("✅ Detailed specifications display")
