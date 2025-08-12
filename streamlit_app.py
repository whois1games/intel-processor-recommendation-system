import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if rank:
            st.subheader(f"🏆 #{rank} - {row['processor_name'][:50]}...")
        else:
            st.subheader(f"📱 {row['processor_name'][:50]}...")
        
        # Create metrics in columns
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("💰 Price", f"${row['feat.price_usd']:,.0f}")
        
        with metric_cols[1]:
            st.metric("⚡ Max Turbo", f"{row['feat.max_turbo_ghz']:.1f} GHz")
        
        with metric_cols[2]:
            st.metric("🔧 Cores/Threads", f"{int(row['feat.total_cores'])}/{int(row['feat.total_threads'])}")
        
        with metric_cols[3]:
            st.metric("💾 Cache", f"{row['feat.cache_mb']:.0f} MB")
        
        # Additional details
        st.write(f"🏷️ **Category:** {row['category']}")
        st.write(f"🔋 **Base Power:** {row['feat.base_power_w']:.0f}W")
        st.write(f"🎮 **Graphics:** {row['feat.gfx_max_dyn_ghz']:.2f} GHz")
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
    
    with col2:
        # Performance score gauge
        if 'pref_score' in row:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = row['pref_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Performance Score"},
                gauge = {
                    'axis': {'range': [None, row['pref_score'] * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, row['pref_score'] * 0.5], 'color': "lightgray"},
                        {'range': [row['pref_score'] * 0.5, row['pref_score']], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': row['pref_score']}}))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

def create_comparison_chart(proc1, proc2):
    """Create comparison chart for two processors"""
    specs = [
        ('Max Turbo (GHz)', 'feat.max_turbo_ghz'),
        ('Cores', 'feat.total_cores'),
        ('Cache (MB)', 'feat.cache_mb'),
        ('Graphics (GHz)', 'feat.gfx_max_dyn_ghz'),
        ('Efficiency', 'feat.freq_per_watt'),
    ]
    
    categories = [spec[0] for spec in specs]
    proc1_values = [proc1[spec[1]] for spec in specs]
    proc2_values = [proc2[spec[1]] for spec in specs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=proc1_values,
        theta=categories,
        fill='toself',
        name=f"Processor 1",
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=proc2_values,
        theta=categories,
        fill='toself',
        name=f"Processor 2",
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(proc1_values), max(proc2_values)) * 1.1]
            )),
        showlegend=True,
        title="Performance Comparison"
    )
    
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
                    st.info("💡 Try broadening your search criteria.")
                else:
                    st.success(f"✅ Found {len(filtered_df)} processors matching your criteria!")
                    recommendations = filtered_df.head(TOP_N)
                    
                    st.subheader(f"🎯 Top {len(recommendations)} Recommendations")
                    
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            display_processor_card(row, rank=idx)
                            st.divider()
    
    elif page == "🔍 Search Processors":
        st.header("🔍 Search Processors")
        
        search_term = st.text_input(
            "Search for processors:",
            placeholder="Try: 'Core 5', 'i7', 'Xeon', '13700', etc."
        )
        
        if search_term:
            matching = df[df['processor_name'].str.contains(search_term, case=False, na=False)]
            
            if matching.empty:
                st.error(f"❌ No processors found containing '{search_term}'")
            else:
                st.success(f"✅ Found {len(matching)} processors")
                
                # Price range filter
                price_range = st.slider(
                    "💰 Price Range ($)",
                    min_value=int(df['feat.price_usd'].min()),
                    max_value=int(df['feat.price_usd'].max()),
                    value=(int(df['feat.price_usd'].min()), int(df['feat.price_usd'].max()))
                )
                
                filtered_matching = matching[
                    (matching['feat.price_usd'] >= price_range[0]) & 
                    (matching['feat.price_usd'] <= price_range[1])
                ].sort_values('feat.price_usd')
                
                if not filtered_matching.empty:
                    for _, row in filtered_matching.iterrows():
                        with st.container():
                            display_processor_card(row)
                            
                            with st.expander("🔍 Detailed Specifications"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"**Base Frequency:** {row['feat.base_freq_ghz']:.1f} GHz")
                                    st.write(f"**Turbo Power:** {row['feat.turbo_power_w']:.0f}W")
                                    st.write(f"**Max Memory:** {row['feat.max_mem_gb']:.0f} GB")
                                
                                with col2:
                                    st.write(f"**Execution Units:** {int(row['feat.execution_units'])}")
                                    st.write(f"**Cache/Core:** {row['feat.cache_per_core']:.2f} MB")
                                    st.write(f"**Cores/Watt:** {row['feat.cores_per_watt']:.3f}")
                                
                                with col3:
                                    st.write(f"**Segment:** {row['feat.vertical_segment']}")
                                    st.write(f"**Freq/Watt:** {row['feat.freq_per_watt']:.3f}")
                            
                            st.divider()
    
    elif page == "📊 Browse by Category":
        st.header("📊 Browse by Category")
        
        categories = df['category'].unique()
        selected_category = st.selectbox("Select a category:", categories)
        
        if selected_category:
            category_df = df[df['category'] == selected_category].sort_values('feat.price_usd')
            
            st.subheader(f"{selected_category} ({len(category_df)} processors)")
            
            # Create a summary chart
            fig = px.scatter(category_df, 
                           x='feat.price_usd', 
                           y='feat.max_turbo_ghz',
                           size='feat.total_cores',
                           color='feat.cache_mb',
                           hover_data=['processor_name'],
                           title=f"{selected_category} - Price vs Performance",
                           labels={
                               'feat.price_usd': 'Price (USD)',
                               'feat.max_turbo_ghz': 'Max Turbo (GHz)',
                               'feat.total_cores': 'Cores',
                               'feat.cache_mb': 'Cache (MB)'
                           })
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show processors
            selected_processor = st.selectbox(
                "Select a processor for details:",
                options=range(len(category_df)),
                format_func=lambda x: f"{category_df.iloc[x]['processor_name']} - ${category_df.iloc[x]['feat.price_usd']:,.0f}"
            )
            
            if selected_processor is not None:
                processor = category_df.iloc[selected_processor]
                display_processor_card(processor)
    
    elif page == "🔄 Compare Processors":
        st.header("🔄 Compare Processors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processor 1")
            search1 = st.text_input("Search Processor 1:", key="search1")
            
            if search1:
                matching1 = df[df['processor_name'].str.contains(search1, case=False, na=False)]
                if not matching1.empty:
                    processor1 = st.selectbox(
                        "Select Processor 1:",
                        options=range(len(matching1)),
                        format_func=lambda x: matching1.iloc[x]['processor_name'][:50],
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
            search2 = st.text_input("Search Processor 2:", key="search2")
            
            if search2:
                matching2 = df[df['processor_name'].str.contains(search2, case=False, na=False)]
                if not matching2.empty:
                    processor2 = st.selectbox(
                        "Select Processor 2:",
                        options=range(len(matching2)),
                        format_func=lambda x: matching2.iloc[x]['processor_name'][:50],
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
            
            # Comparison chart
            comparison_fig = create_comparison_chart(proc1, proc2)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Side-by-side comparison
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.subheader("Processor 1")
                display_processor_card(proc1)
            
            with comp_col2:
                st.subheader("Processor 2")
                display_processor_card(proc2)
            
            # Winner analysis
            st.subheader("🏆 Analysis")
            
            metrics = {
                "Price": ("feat.price_usd", "lower_better"),
                "Max Turbo": ("feat.max_turbo_ghz", "higher_better"),
                "Cores": ("feat.total_cores", "higher_better"),
                "Cache": ("feat.cache_mb", "higher_better"),
                "Power Efficiency": ("feat.freq_per_watt", "higher_better"),
                "Value Score": ("calculated", "higher_better")
            }
            
            comparison_data = []
            
            for metric_name, (col, better) in metrics.items():
                if col == "calculated":
                    val1 = (proc1['feat.max_turbo_ghz'] + proc1['feat.total_cores']) / (proc1['feat.price_usd'] / 1000)
                    val2 = (proc2['feat.max_turbo_ghz'] + proc2['feat.total_cores']) / (proc2['feat.price_usd'] / 1000)
                else:
                    val1 = proc1[col]
                    val2 = proc2[col]
                
                if better == "lower_better":
                    winner = "Processor 1" if val1 < val2 else "Processor 2" if val1 > val2 else "Tie"
                else:
                    winner = "Processor 1" if val1 > val2 else "Processor 2" if val1 < val2 else "Tie"
                
                comparison_data.append({
                    "Metric": metric_name,
                    "Processor 1": f"{val1:.2f}",
                    "Processor 2": f"{val2:.2f}",
                    "Winner": winner
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    elif page == "📈 Analytics":
        st.header("📈 Market Analytics")
        
        # Price distribution
        fig_price = px.histogram(df, x='feat.price_usd', nbins=30, 
                               title="Price Distribution of Intel Processors")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Performance vs Price scatter
        fig_perf = px.scatter(df, x='feat.price_usd', y='feat.max_turbo_ghz',
                             size='feat.total_cores', color='category',
                             title="Performance vs Price Analysis",
                             labels={
                                 'feat.price_usd': 'Price (USD)',
                                 'feat.max_turbo_ghz': 'Max Turbo (GHz)'
                             })
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Category statistics
        st.subheader("📊 Category Statistics")
        category_stats = df.groupby('category').agg({
            'feat.price_usd': ['mean', 'min', 'max', 'count'],
            'feat.max_turbo_ghz': 'mean',
            'feat.total_cores': 'mean'
        }).round(2)
        
        category_stats.columns = ['Avg Price', 'Min Price', 'Max Price', 'Count', 'Avg Turbo', 'Avg Cores']
        st.dataframe(category_stats, use_container_width=True)

if __name__ == "__main__":
    main()
