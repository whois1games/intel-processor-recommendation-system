# %%
# Interactive Intel Processor Recommendation using Co# -------------------
# INTERACTIVE MODE
# -------------------
def interactive_recommend():
    """Interactive recommendation with search functionality"""
    print("\n🎯 Intel Processor Recommendation System")
    print("=" * 50)
    
    # Allow partial search
    search_term = input("\n🔍 Enter processor name or part of it (e.g., 'Core i7', 'Xeon'): ").strip()
    
    if not search_term:
        print("❌ Please enter a search term!")
        return
    
    # Search for processors containing the search term
    matching = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
    
    if matching.empty:
        print(f"❌ No processors found containing '{search_term}'")
        print("\n📋 Here are some example processors:")
        for name in df["processor_name"].head(10):
            print(f" - {name}")
        return
    
    print(f"\n✅ Found {len(matching)} processors containing '{search_term}':")
    print("-" * 60)
    
    # Show matching processors with key specs
    for idx, (_, row) in enumerate(matching.head(15).iterrows(), 1):
        print(f"{idx:2}. {row['processor_name']}")
        print(f"    💰 ${row['feat.price_usd']:,.0f} | ⚡ {row['feat.max_turbo_ghz']:.1f}GHz | 🔧 {int(row['feat.total_cores'])}cores")
    
    if len(matching) > 15:
        print(f"    ... and {len(matching) - 15} more")
    
    try:
        choice = int(input(f"\nSelect a processor (1-{min(15, len(matching))}): ")) - 1
        if 0 <= choice < len(matching):
            selected_cpu = matching.iloc[choice]['processor_name']
            
            print(f"\n🔍 Finding recommendations for: {selected_cpu}")
            print("-" * 60)
            
            results = recommend(selected_cpu, top_n=TOP_N)
            
            print(f"\n🏆 TOP {TOP_N} SIMILAR PROCESSORS:")
            print("=" * 80)
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                print(f"\n{idx}. {row['processor_name']}")
                print(f"   🏷️  Category: {row['category']}")
                print(f"   💰 Price: ${row['feat.price_usd']:,.0f}")
                print(f"   ⚡ Max Turbo: {row['feat.max_turbo_ghz']:.1f} GHz")
                print(f"   🔧 Cores/Threads: {int(row['feat.total_cores'])}/{int(row['feat.total_threads'])}")
                print(f"   💾 Cache: {row['feat.cache_mb']:.0f} MB")
                print(f"   🔋 Power: {row['feat.base_power_w']:.0f}W")
                print(f"   📊 Similarity: {row['similarity_score']:.3f}")
                
        else:
            print("❌ Invalid selection!")
    except ValueError:
        print("❌ Please enter a valid number!")

if __name__ == "__main__":
    while True:
        interactive_recommend()
        
        again = input("\n🔄 Would you like to search for another processor? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("\n👋 Thank you for using the recommendation system!")
            break
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
# RECOMMEND FUNCTION
# -------------------
def recommend(cpu_name: str, top_n: int = TOP_N):
    if cpu_name not in df["processor_name"].values:
        raise ValueError(f"❌ CPU '{cpu_name}' not found in dataset.")

    idx = df.index[df["processor_name"] == cpu_name][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    recs = []
    for i, score in sim_scores:
        row = df.iloc[i][id_cols + numeric_cols].to_dict()
        row["similarity_score"] = round(score, 3)
        recs.append(row)

    return pd.DataFrame(recs)

# -------------------
# INTERACTIVE MODE
# -------------------
if __name__ == "__main__":
    # Show first 10 CPU names so user knows what’s available
    print("\n📋 Example CPUs:")
    for name in df["processor_name"].head(10):
        print(f" - {name}")

    user_input = input("\nEnter the exact processor name from the list above: ").strip()

    try:
        results = recommend(user_input, top_n=TOP_N)
        print(f"\n🔍 Recommendations for '{user_input}':\n")
        print(results.to_string(index=False))
    except ValueError as e:
        print(e)
