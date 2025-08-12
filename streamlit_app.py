import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# CONFIG
# -------------------
FEATURES_FILE = "data/intel_processors_features.csv"
TOP_N = 5

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="Intel Processor Recommendation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# LOAD DATA & CACHE
# -------------------
@st.cache_data
def load_data():
    """Load and cache the processor data"""
    try:
        df = pd.read_csv(FEATURES_FILE)
        return df
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {FEATURES_FILE}")
        st.info("📁 Please make sure your data file is at: `data/intel_processors_features.csv`")
        st.stop()

@st.cache_data
def prepare_similarity_matrix(df):
    """Prepare and cache similarity matrix"""
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
    
    return similarity_matrix, scaler

# Load data
df = load_data()
similarity_matrix, scaler = prepare_similarity_matrix(df)

# -------------------
# UTILITY FUNCTIONS
# -------------------
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
        
        mapped_term = search_mapping.get(search_term.lower(), search_term)
        
        filtered_df = filtered_df[
            filtered_df['processor_name'].str.contains(mapped_term, case=False, na=False) |
            filtered_df['processor_name'].str.contains(search_term, case=False, na=False)
        ]
    
    # Filter by budget
    if 'budget' in preferences:
        min_price, max_price = preferences['budget']
        filtered_df = filtered_df[
            (filtered_df['feat.price_usd'] >= min_price) & 
            (filtered_df['feat.price_usd'] <= max_price)
        ]
    
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

def get_usage_recommendation(row):
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

def display_processor_card(row, rank=None):
    """Display a processor as a card"""
    # Container for the processor card
    with st.container():
        # Header with rank and name
        if rank:
            st.subheader(f"🏆 #{rank} - {row['processor_name'][:60]}")
        else:
            st.subheader(f"📱 {row['processor_name'][:60]}")
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📱 Total Processors", f"{len(df)}")
        
        with col2:
            st.metric("💰 Avg Price", f"${df['feat.price_usd'].mean():,.0f}")
        
        with col3:
            st.metric("⚡ Avg Turbo", f"{df['feat.max_turbo_ghz'].mean():.1f} GHz")
        
        with col4:
            st.metric("🔧 Avg Cores", f"{df['feat.total_cores'].mean():.1f}")
        
        # Price distribution
        st.subheader("💰 Price Distribution")
        
        price_ranges = [
            ("Under $300", 0, 300),
            ("$300-600", 300, 600),
            ("$600-1000", 600, 1000),
            ("$1000-2000", 1000, 2000),
            ("Over $2000", 2000, float('inf'))
        ]
        
        price_dist_data = []
        for label, min_p, max_p in price_ranges:
            count = len(df[(df['feat.price_usd'] >= min_p) & 
                          (df['feat.price_usd'] < max_p)])
            price_dist_data.append({"Price Range": label, "Count": count})
        
        price_dist_df = pd.DataFrame(price_dist_data)
        st.bar_chart(price_dist_df.set_index('Price Range'))
        
        # Performance vs Price analysis
        st.subheader("⚡ Performance vs Price Analysis")
        
        # Create performance score
        df['performance_score'] = (df['feat.max_turbo_ghz'] * 0.4 + 
                                 df['feat.total_cores'] * 0.3 + 
                                 df['feat.freq_per_watt'] * 0.3)
        
        # Create price vs performance chart
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color by category
            categories = df['category'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            for i, category in enumerate(categories):
                cat_data = df[df['category'] == category]
                scatter = ax.scatter(cat_data['feat.price_usd'], 
                                   cat_data['performance_score'],
                                   label=category,
                                   alpha=0.7,
                                   s=cat_data['feat.total_cores']*5,
                                   c=[colors[i]])
            
            ax.set_xlabel('Price (USD)')
            ax.set_ylabel('Performance Score')
            ax.set_title('Performance Score vs Price by Category')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add trend line
            z = np.polyfit(df['feat.price_usd'], df['performance_score'], 1)
            p = np.poly1d(z)
            ax.plot(df['feat.price_usd'].sort_values(), 
                   p(df['feat.price_usd'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2, label='Trend')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.info("Performance vs Price chart skipped - continuing with other analytics.")
        
        # Category breakdown
        st.subheader("🏷️ Category Statistics")
        
        category_stats = df.groupby('category').agg({
            'feat.price_usd': ['mean', 'min', 'max', 'count'],
            'feat.max_turbo_ghz': 'mean',
            'feat.total_cores': 'mean',
            'feat.cache_mb': 'mean'
        }).round(2)
        
        category_stats.columns = ['Avg Price ($)', 'Min Price ($)', 'Max Price ($)', 
                                'Count', 'Avg Turbo (GHz)', 'Avg Cores', 'Avg Cache (MB)']
        st.dataframe(category_stats, use_container_width=True)
        
        # Top performers by different metrics
        st.subheader("🏆 Top Performers")
        
        col_top1, col_top2, col_top3 = st.columns(3)
        
        with col_top1:
            st.write("**⚡ Highest Turbo Frequency**")
            top_turbo = df.nlargest(5, 'feat.max_turbo_ghz')[['processor_name', 'feat.max_turbo_ghz', 'feat.price_usd']]
            for _, row in top_turbo.iterrows():
                st.write(f"• {row['processor_name'][:30]}...")
                st.write(f"  {row['feat.max_turbo_ghz']:.1f} GHz - ${row['feat.price_usd']:,.0f}")
        
        with col_top2:
            st.write("**🔧 Most Cores**")
            top_cores = df.nlargest(5, 'feat.total_cores')[['processor_name', 'feat.total_cores', 'feat.price_usd']]
            for _, row in top_cores.iterrows():
                st.write(f"• {row['processor_name'][:30]}...")
                st.write(f"  {int(row['feat.total_cores'])} cores - ${row['feat.price_usd']:,.0f}")
        
        with col_top3:
            st.write("**⚡ Best Efficiency (Freq/Watt)**")
            top_efficiency = df.nlargest(5, 'feat.freq_per_watt')[['processor_name', 'feat.freq_per_watt', 'feat.price_usd']]
            for _, row in top_efficiency.iterrows():
                st.write(f"• {row['processor_name'][:30]}...")
                st.write(f"  {row['feat.freq_per_watt']:.3f} - ${row['feat.price_usd']:,.0f}")
        
        # Value analysis
        st.subheader("💎 Best Value Processors")
        
        # Calculate value score (performance per dollar)
        df['value_score'] = df['performance_score'] / (df['feat.price_usd'] / 1000)
        
        best_value = df.nlargest(10, 'value_score')[['processor_name', 'feat.price_usd', 
                                                     'feat.max_turbo_ghz', 'feat.total_cores', 'value_score']]
        
        st.write("**Top 10 processors by value (performance per $1000):**")
        
        for idx, (_, row) in enumerate(best_value.iterrows(), 1):
            col_val1, col_val2 = st.columns([3, 1])
            
            with col_val1:
                st.write(f"**{idx}. {row['processor_name'][:50]}**")
                st.write(f"💰 ${row['feat.price_usd']:,.0f} | ⚡ {row['feat.max_turbo_ghz']:.1f}GHz | 🔧 {int(row['feat.total_cores'])} cores")
            
            with col_val2:
                st.metric("Value Score", f"{row['value_score']:.2f}")
        
        # Market insights
        st.subheader("🧠 Market Insights")
        
        insights = []
        
        # Price insights
        budget_count = len(df[df['feat.price_usd'] < 500])
        premium_count = len(df[df['feat.price_usd'] > 1000])
        insights.append(f"📊 {budget_count} processors ({budget_count/len(df)*100:.1f}%) are budget-friendly (under $500)")
        insights.append(f"💎 {premium_count} processors ({premium_count/len(df)*100:.1f}%) are premium (over $1000)")
        
        # Performance insights
        high_perf_count = len(df[df['feat.max_turbo_ghz'] > 4.5])
        many_cores_count = len(df[df['feat.total_cores'] >= 8])
        insights.append(f"🚀 {high_perf_count} processors ({high_perf_count/len(df)*100:.1f}%) have turbo frequency above 4.5 GHz")
        insights.append(f"🔧 {many_cores_count} processors ({many_cores_count/len(df)*100:.1f}%) have 8 or more cores")
        
        # Efficiency insights
        efficient_count = len(df[df['feat.freq_per_watt'] > 0.1])
        insights.append(f"⚡ {efficient_count} processors ({efficient_count/len(df)*100:.1f}%) are highly efficient (>0.1 GHz/Watt)")
        
        for insight in insights:
            st.info(insight)
        
        # Data download
        st.subheader("📥 Export Data")
        
        # Prepare summary data for download
        summary_data = df[['processor_name', 'category', 'feat.price_usd', 'feat.max_turbo_ghz', 
                          'feat.total_cores', 'feat.total_threads', 'feat.cache_mb', 
                          'feat.base_power_w', 'feat.freq_per_watt']].copy()
        
        summary_data.columns = ['Processor Name', 'Category', 'Price (USD)', 'Max Turbo (GHz)', 
                               'Cores', 'Threads', 'Cache (MB)', 'Base Power (W)', 'Efficiency']
        
        csv = summary_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Processor Data (CSV)",
            data=csv,
            file_name="intel_processors_summary.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
        
        with col1:
            st.metric("💰 Price", f"${row['feat.price_usd']:,.0f}")
        
        with col2:
            st.metric("⚡ Max Turbo", f"{row['feat.max_turbo_ghz']:.1f} GHz")
        
        with col3:
            st.metric("🔧 Cores/Threads", f"{int(row['feat.total_cores'])}/{int(row['feat.total_threads'])}")
        
        with col4:
            st.metric("💾 Cache", f"{row['feat.cache_mb']:.0f} MB")
        
        # Additional info in two columns
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"🏷️ **Category:** {row['category']}")
            st.write(f"🔋 **Base Power:** {row['feat.base_power_w']:.0f}W")
            st.write(f"🎮 **Graphics:** {row['feat.gfx_max_dyn_ghz']:.2f} GHz")
        
        with info_col2:
            st.write(f"✅ **Best for:** {get_usage_recommendation(row)}")
            
            # Price category
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
            
            st.write(f"🏷️ **Price Category:** {price_cat}")
            
            # Performance score if available
            if 'pref_score' in row:
                st.write(f"📊 **Performance Score:** {row['pref_score']:.2f}")

def create_simple_comparison_table(proc1, proc2):
    """Create a simple comparison table"""
    specs = {
        'Specification': ['Price ($)', 'Max Turbo (GHz)', 'Cores', 'Threads', 'Cache (MB)', 
                         'Base Power (W)', 'Graphics (GHz)', 'Efficiency (Freq/Watt)'],
        'Processor 1': [
            f"${proc1['feat.price_usd']:,.0f}",
            f"{proc1['feat.max_turbo_ghz']:.1f}",
            f"{int(proc1['feat.total_cores'])}",
            f"{int(proc1['feat.total_threads'])}",
            f"{proc1['feat.cache_mb']:.0f}",
            f"{proc1['feat.base_power_w']:.0f}",
            f"{proc1['feat.gfx_max_dyn_ghz']:.2f}",
            f"{proc1['feat.freq_per_watt']:.3f}"
        ],
        'Processor 2': [
            f"${proc2['feat.price_usd']:,.0f}",
            f"{proc2['feat.max_turbo_ghz']:.1f}",
            f"{int(proc2['feat.total_cores'])}",
            f"{int(proc2['feat.total_threads'])}",
            f"{proc2['feat.cache_mb']:.0f}",
            f"{proc2['feat.base_power_w']:.0f}",
            f"{proc2['feat.gfx_max_dyn_ghz']:.2f}",
            f"{proc2['feat.freq_per_watt']:.3f}"
        ]
    }
    
    return pd.DataFrame(specs)

def create_price_performance_chart(category_df, category_name):
    """Create a simple matplotlib chart for price vs performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(category_df['feat.price_usd'], 
                        category_df['feat.max_turbo_ghz'],
                        s=category_df['feat.total_cores']*10,
                        alpha=0.6,
                        c=category_df['feat.cache_mb'],
                        cmap='viridis')
    
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Max Turbo (GHz)')
    ax.set_title(f'{category_name} - Price vs Performance')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cache (MB)')
    
    # Add some annotations for outliers
    for idx, row in category_df.iterrows():
        if (row['feat.price_usd'] > category_df['feat.price_usd'].quantile(0.8) or 
            row['feat.max_turbo_ghz'] > category_df['feat.max_turbo_ghz'].quantile(0.8)):
            ax.annotate(row['processor_name'][:20], 
                       (row['feat.price_usd'], row['feat.max_turbo_ghz']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig

# -------------------
# MAIN APP
# -------------------
def main():
    st.title("🚀 Intel Processor Recommendation System")
    st.markdown("Find the perfect Intel processor for your needs!")
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["🎯 Get Recommendations", "🔍 Search Processors", "📊 Browse by Category", "🔄 Compare Processors", "📈 Analytics"]
    )
    
    if page == "🎯 Get Recommendations":
        st.header("🎯 Personalized Recommendations")
        
        with st.form("preferences_form"):
            st.subheader("Tell us about your needs:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                processor_type = st.selectbox(
                    "🔍 Processor Type:",
                    ["Any", "Core 3", "Core 5", "Core 7", "Core 9", "Core Ultra", "Xeon", "i3", "i5", "i7", "i9"]
                )
                
                budget_range = st.selectbox(
                    "💰 Budget Range:",
                    ["No constraint", "Budget ($0 - $300)", "Mid-range ($300 - $600)", 
                     "High-end ($600 - $1000)", "Premium ($1000 - $2000)", "Ultra Premium ($2000+)"]
                )
                
                if budget_range == "No constraint":
                    budget = (0, float('inf'))
                else:
                    budget_map = {
                        "Budget ($0 - $300)": (0, 300),
                        "Mid-range ($300 - $600)": (300, 600),
                        "High-end ($600 - $1000)": (600, 1000),
                        "Premium ($1000 - $2000)": (1000, 2000),
                        "Ultra Premium ($2000+)": (2000, float('inf'))
                    }
                    budget = budget_map[budget_range]
            
            with col2:
                usage = st.selectbox(
                    "🎯 Primary Usage:",
                    ["Gaming", "Content Creation", "Office Work", "Programming", "Server/Enterprise"]
                )
                
                priority = st.selectbox(
                    "⚡ Performance Priority:",
                    ["Balanced", "Single-core Performance", "Multi-core Performance", "Power Efficiency"]
                )
                
                priority_map = {
                    "Balanced": "balanced",
                    "Single-core Performance": "single_core",
                    "Multi-core Performance": "multi_core",
                    "Power Efficiency": "efficiency"
                }
                priority = priority_map[priority]
            
            submitted = st.form_submit_button("🔍 Get Recommendations")
            
            if submitted:
                preferences = {
                    'processor_type': processor_type if processor_type != "Any" else "",
                    'budget': budget,
                    'usage': usage,
                    'priority': priority
                }
                
                filtered_df = filter_by_preferences(df, preferences)
                
                if filtered_df.empty:
                    st.error("❌ No processors found matching your criteria.")
                    
                    # Show helpful suggestions
                    st.info("💡 **Suggestions to broaden your search:**")
                    st.write("• Try a wider budget range")
                    st.write("• Select 'Any' for processor type")
                    st.write("• Check if your search terms are correct")
                    
                    # Show what's available in budget
                    if budget != (0, float('inf')):
                        budget_processors = df[
                            (df['feat.price_usd'] >= budget[0]) & 
                            (df['feat.price_usd'] <= budget[1])
                        ]
                        if len(budget_processors) > 0:
                            st.write(f"📊 **{len(budget_processors)} processors available in your budget range**")
                            families = set()
                            for name in budget_processors['processor_name'].head(10):
                                if 'Core™' in name:
                                    try:
                                        family = name.split('Core™')[1].split()[0]
                                        families.add(f"Core {family}")
                                    except:
                                        continue
                                elif 'Xeon' in name:
                                    families.add("Xeon")
                            if families:
                                st.write(f"Available families: {', '.join(sorted(families))}")
                else:
                    st.success(f"✅ Found {len(filtered_df)} processors matching your criteria!")
                    recommendations = filtered_df.head(TOP_N)
                    
                    st.subheader(f"🎯 Top {len(recommendations)} Recommendations")
                    
                    # Show search summary
                    st.info(f"**Search Summary:** {preferences.get('usage', 'General')} usage, "
                           f"{priority.replace('_', ' ').title()} priority, "
                           f"Budget: ${budget[0]:,.0f}" + 
                           (f" - ${budget[1]:,.0f}" if budget[1] != float('inf') else "+"))
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            display_processor_card(row, rank=idx)
                            st.divider()
    
    elif page == "🔍 Search Processors":
        st.header("🔍 Search Processors")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input(
                "Search for processors:",
                placeholder="Try: 'Core 5', 'i7', 'Xeon', '13700', etc."
            )
        
        with col2:
            st.write("**💡 Search Examples:**")
            st.write("• Core 3, Core 5, Core 7")
            st.write("• i5, i7, i9")
            st.write("• Xeon, Ultra")
        
        if search_term:
            matching = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
            
            if matching.empty:
                st.error(f"❌ No processors found containing '{search_term}'")
                st.info("💡 **Try these suggestions:**")
                st.write("• Use shorter terms like 'Core 5' instead of full names")
                st.write("• Try legacy names like 'i5', 'i7', 'i9'")
                st.write("• Use 'Xeon' for server processors")
            else:
                st.success(f"✅ Found {len(matching)} processors")
                
                # Filters
                col_filter1, col_filter2 = st.columns(2)
                
                with col_filter1:
                    # Price range filter
                    price_min = int(matching['feat.price_usd'].min())
                    price_max = int(matching['feat.price_usd'].max())
                    price_range = st.slider(
                        "💰 Price Range ($)",
                        min_value=price_min,
                        max_value=price_max,
                        value=(price_min, price_max)
                    )
                
                with col_filter2:
                    # Sort options
                    sort_by = st.selectbox(
                        "📊 Sort by:",
                        ["Price (Low to High)", "Price (High to Low)", 
                         "Performance (High to Low)", "Name (A-Z)"]
                    )
                
                # Apply filters
                filtered_matching = matching[
                    (matching['feat.price_usd'] >= price_range[0]) & 
                    (matching['feat.price_usd'] <= price_range[1])
                ]
                
                # Apply sorting
                if sort_by == "Price (Low to High)":
                    filtered_matching = filtered_matching.sort_values('feat.price_usd')
                elif sort_by == "Price (High to Low)":
                    filtered_matching = filtered_matching.sort_values('feat.price_usd', ascending=False)
                elif sort_by == "Performance (High to Low)":
                    filtered_matching = filtered_matching.sort_values('feat.max_turbo_ghz', ascending=False)
                else:  # Name A-Z
                    filtered_matching = filtered_matching.sort_values('processor_name')
                
                if not filtered_matching.empty:
                    st.write(f"**Showing {len(filtered_matching)} processors:**")
                    
                    for _, row in filtered_matching.iterrows():
                        with st.container():
                            display_processor_card(row)
                            
                            with st.expander("🔍 Detailed Specifications"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Core Specifications:**")
                                    st.write(f"Base Frequency: {row['feat.base_freq_ghz']:.1f} GHz")
                                    st.write(f"Turbo Power: {row['feat.turbo_power_w']:.0f}W")
                                    st.write(f"Max Memory: {row['feat.max_mem_gb']:.0f} GB")
                                
                                with col2:
                                    st.write("**Graphics & Processing:**")
                                    st.write(f"Execution Units: {int(row['feat.execution_units'])}")
                                    st.write(f"Cache/Core: {row['feat.cache_per_core']:.2f} MB")
                                    st.write(f"Cores/Watt: {row['feat.cores_per_watt']:.3f}")
                                
                                with col3:
                                    st.write("**Additional Info:**")
                                    st.write(f"Segment: {row['feat.vertical_segment']}")
                                    st.write(f"Freq/Watt: {row['feat.freq_per_watt']:.3f}")
                                    # Value score
                                    value_score = (row['feat.max_turbo_ghz'] + row['feat.total_cores']) / (row['feat.price_usd'] / 1000)
                                    st.write(f"Value Score: {value_score:.2f}")
                            
                            st.divider()
                else:
                    st.warning("No processors match your current filters.")
    
    elif page == "📊 Browse by Category":
        st.header("📊 Browse by Category")
        
        categories = df['category'].unique()
        selected_category = st.selectbox("Select a category:", categories)
        
        if selected_category:
            category_df = df[df['category'] == selected_category].sort_values('feat.price_usd')
            
            st.subheader(f"{selected_category} ({len(category_df)} processors)")
            
            # Category statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Avg Price", f"${category_df['feat.price_usd'].mean():,.0f}")
            
            with col2:
                st.metric("⚡ Avg Turbo", f"{category_df['feat.max_turbo_ghz'].mean():.1f} GHz")
            
            with col3:
                st.metric("🔧 Avg Cores", f"{category_df['feat.total_cores'].mean():.1f}")
            
            with col4:
                st.metric("💾 Avg Cache", f"{category_df['feat.cache_mb'].mean():.0f} MB")
            
            # Create and display chart
            try:
                fig = create_price_performance_chart(category_df, selected_category)
                st.pyplot(fig)
            except Exception as e:
                st.info("Chart generation skipped - continuing with processor list.")
            
            # Price ranges within category
            st.subheader("💰 Price Ranges")
            price_ranges = [
                ("💚 Budget", 0, 300),
                ("💙 Mid-range", 300, 600),
                ("💜 High-end", 600, 1000),
                ("🧡 Premium", 1000, 2000),
                ("❤️ Ultra Premium", 2000, float('inf'))
            ]
            
            for label, min_p, max_p in price_ranges:
                range_processors = category_df[
                    (category_df['feat.price_usd'] >= min_p) & 
                    (category_df['feat.price_usd'] < max_p)
                ]
                
                if not range_processors.empty:
                    with st.expander(f"{label} ({len(range_processors)} processors)"):
                        for _, row in range_processors.head(5).iterrows():
                            st.write(f"**{row['processor_name'][:50]}**")
                            st.write(f"💰 ${row['feat.price_usd']:,.0f} | ⚡ {row['feat.max_turbo_ghz']:.1f}GHz | 🔧 {int(row['feat.total_cores'])}C/{int(row['feat.total_threads'])}T")
                            st.write("---")
                        
                        if len(range_processors) > 5:
                            st.write(f"... and {len(range_processors) - 5} more processors")
            
            # Detailed processor selection
            st.subheader("🔍 Detailed View")
            selected_processor = st.selectbox(
                "Select a processor for detailed specifications:",
                options=range(len(category_df)),
                format_func=lambda x: f"{category_df.iloc[x]['processor_name'][:50]} - ${category_df.iloc[x]['feat.price_usd']:,.0f}"
            )
            
            if selected_processor is not None:
                processor = category_df.iloc[selected_processor]
                display_processor_card(processor)
    
    elif page == "🔄 Compare Processors":
        st.header("🔄 Compare Processors")
        
        st.info("💡 **How to compare:** Search for processors using short terms like 'Core 5', 'i7', 'Xeon', or model numbers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processor 1")
            search1 = st.text_input("Search Processor 1:", key="search1", 
                                   placeholder="e.g., 'Core 7', 'i9'")
            
            if search1:
                matching1 = df[df['processor_name'].str.contains(search1, case=False, na=False)]
                if not matching1.empty:
                    # Show count
                    st.write(f"Found {len(matching1)} processors")
                    
                    processor1 = st.selectbox(
                        "Select Processor 1:",
                        options=range(len(matching1)),
                        format_func=lambda x: f"{matching1.iloc[x]['processor_name'][:40]} - ${matching1.iloc[x]['feat.price_usd']:,.0f}",
                        key="proc1"
                    )
                    proc1 = matching1.iloc[processor1] if processor1 is not None else None
                else:
                    proc1 = None
                    st.error("No processors found")
            else:
                proc1 = None
        
        with col2:
            st.subheader("Processor 2")
            search2 = st.text_input("Search Processor 2:", key="search2",
                                   placeholder="e.g., 'Core 5', 'i5'")
            
            if search2:
                matching2 = df[df['processor_name'].str.contains(search2, case=False, na=False)]
                if not matching2.empty:
                    # Show count
                    st.write(f"Found {len(matching2)} processors")
                    
                    processor2 = st.selectbox(
                        "Select Processor 2:",
                        options=range(len(matching2)),
                        format_func=lambda x: f"{matching2.iloc[x]['processor_name'][:40]} - ${matching2.iloc[x]['feat.price_usd']:,.0f}",
                        key="proc2"
                    )
                    proc2 = matching2.iloc[processor2] if processor2 is not None else None
                else:
                    proc2 = None
                    st.error("No processors found")
            else:
                proc2 = None
        
        if proc1 is not None and proc2 is not None:
            st.subheader("🔄 Comparison Results")
            
            # Side-by-side comparison
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.subheader("Processor 1")
                display_processor_card(proc1)
            
            with comp_col2:
                st.subheader("Processor 2")
                display_processor_card(proc2)
            
            # Detailed comparison table
            st.subheader("📊 Detailed Comparison")
            comparison_df = create_simple_comparison_table(proc1, proc2)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Winner analysis
            st.subheader("🏆 Analysis")
            
            # Price comparison
            price_winner = "Processor 1" if proc1['feat.price_usd'] < proc2['feat.price_usd'] else "Processor 2"
            perf_winner = "Processor 1" if proc1['feat.max_turbo_ghz'] > proc2['feat.max_turbo_ghz'] else "Processor 2"
            cores_winner = "Processor 1" if proc1['feat.total_cores'] > proc2['feat.total_cores'] else "Processor 2"
            
            col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
            
            with col_analysis1:
                st.metric("💰 Better Price", price_winner)
            
            with col_analysis2:
                st.metric("⚡ Higher Performance", perf_winner)
            
            with col_analysis3:
                st.metric("🔧 More Cores", cores_winner)
            
            # Overall recommendation
            st.write("**🎯 Recommendation:**")
            
            # Calculate value scores
            value1 = (proc1['feat.max_turbo_ghz'] + proc1['feat.total_cores']) / (proc1['feat.price_usd'] / 1000)
            value2 = (proc2['feat.max_turbo_ghz'] + proc2['feat.total_cores']) / (proc2['feat.price_usd'] / 1000)
            
            if value1 > value2 * 1.1:  # 10% better value
                st.success("🏆 **Processor 1** offers significantly better value for money!")
            elif value2 > value1 * 1.1:
                st.success("🏆 **Processor 2** offers significantly better value for money!")
            else:
                st.info("⚖️ Both processors offer similar value. Choose based on your specific needs.")
            
            # Usage recommendations
            st.write("**✅ Usage Recommendations:**")
            st.write(f"• **Processor 1:** {get_usage_recommendation(proc1)}")
            st.write(f"• **Processor 2:** {get_usage_recommendation(proc2)}")
    
    elif page == "📈 Analytics":
        st.header("📈 Market Analytics")
        
        # Overall statistics
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
