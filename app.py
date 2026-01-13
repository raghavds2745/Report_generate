import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="LLM Replay Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">LLM Replay Comparison Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare evaluation metrics across different replay runs</p>', unsafe_allow_html=True)


    
with st.sidebar:
    st.header("📁 Data Source Selection")
    
    # Add radio button for data source selection
    data_source = st.radio(
        "Choose data source:",
        options=["Upload CSV Files", "Use Sample Data"],
        index=0,
        help="Select whether to upload your own files or use sample data for demo"
    )
    
    st.markdown("---")
    
    prev_file = None
    new_file = None
    
    if data_source == "Upload CSV Files":
        st.subheader("📤 Upload Your CSV Files")
        
        prev_file = st.file_uploader(
            "Previous Run CSV",
            type=['csv'],
            key='prev',
            help="Upload the CSV from your previous replay run"
        )
        
        new_file = st.file_uploader(
            "New Run CSV",
            type=['csv'],
            key='new',
            help="Upload the CSV from your new replay run"
        )
        
        if prev_file and new_file:
            st.success("✅ Both files uploaded successfully!")
        elif prev_file:
            st.warning("⚠️ Please upload the New Run CSV")
        elif new_file:
            st.warning("⚠️ Please upload the Previous Run CSV")
        else:
            st.info("👆 Upload both CSV files to start comparison")
    
    else:  # Use Sample Data
        st.subheader("📊 Sample Data")
        
        import os
        
        # Check if sample files exist
        sample_prev_path = 'for_report_generation_community_303.csv'
        sample_new_path = '05-01-2026-16-17community_304_filtered_metrics_ragas.csv'
        
        if os.path.exists(sample_prev_path) and os.path.exists(sample_new_path):
            prev_file = sample_prev_path
            new_file = sample_new_path
            
            st.success("✅ Sample data loaded!")
            
            st.info("""
            **About Sample Data:**
            - Contains 30 example questions
            - Demonstrates all comparison features
            - Perfect for testing the tool
            """)
            
            # Show sample data info
            with st.expander("View Sample Data Info"):
                sample_df_prev = pd.read_csv(sample_prev_path)
                sample_df_new = pd.read_csv(sample_new_path)
                
                st.write(f"**Previous Run:** {len(sample_df_prev)} rows")
                st.write(f"**New Run:** {len(sample_df_new)} rows")
                st.write(f"**Columns:** {', '.join(sample_df_prev.columns.tolist())}")
        else:
            st.error("❌ Sample data files not found in repository")
            st.write("""
            **Expected files:**
            - `sample_data/sample_prev.csv`
            - `sample_data/sample_new.csv`
            
            Please ensure these files exist in your repository.
            """)
            prev_file = None
            new_file = None
    
    st.markdown("---")
    
    # Threshold slider (only show if files are loaded)
    if prev_file and new_file:
        st.subheader("⚙️ Comparison Settings")
        threshold = st.slider(
            "Change Threshold (%)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Highlight rows where metrics changed by this percentage or more"
        )
    else:
        threshold = 10  # Default value
    # st.markdown("---")
    
    # if prev_file and new_file:
    #     st.success("Both files uploaded successfully")
    # else:
    #     st.info("Please upload both CSV files to begin comparison")
    st.markdown("---")
    

# Main content
# if prev_file and new_file:
#     # Load data
#     try:
#         df_prev = pd.read_csv(prev_file)
#         df_new = pd.read_csv(new_file)
        
#         # Verify required columns exist
#         required_cols = ['user_input', 'answer_relevancy', 'answer_correctness', 
#                         'Precision', 'Recall', 'Reranker Score', 'F1 Score']
        
#         missing_prev = [col for col in required_cols if col not in df_prev.columns]
#         missing_new = [col for col in required_cols if col not in df_new.columns]
        
#         if missing_prev or missing_new:
#             st.error("Missing required columns in uploaded files")
#             if missing_prev:
#                 st.write("Previous CSV missing:", missing_prev)
#             if missing_new:
#                 st.write("New CSV missing:", missing_new)
#         else:
#             # Numeric columns for comparison
#             numeric_cols = ['answer_relevancy', 'answer_correctness', 
#                            'Precision', 'Recall', 'Reranker Score', 'F1 Score']

if prev_file and new_file:
    try:
        # Check if files are the same
        if data_source == "Select from Available Files":
            if os.path.basename(prev_file) == os.path.basename(new_file):
                st.error("❌ Cannot compare the same file with itself!")
                st.stop()
        
        # Determine if we're using stored files or uploaded files
        if isinstance(prev_file, str):  # Stored files (file paths)
            df_prev = pd.read_csv(prev_file)
            df_new = pd.read_csv(new_file)
            data_source_type = "stored"
        else:  # Uploaded files
            df_prev = pd.read_csv(prev_file)
            df_new = pd.read_csv(new_file)
            data_source_type = "uploaded"
        
        # Verify required columns exist
        required_cols = ['user_input', 'answer_relevancy', 'answer_correctness', 
                        'Precision', 'Recall', 'Reranker Score', 'F1 Score']
        
        missing_prev = [col for col in required_cols if col not in df_prev.columns]
        missing_new = [col for col in required_cols if col not in df_new.columns]
        
        if missing_prev or missing_new:
            st.error("❌ Missing required columns in CSV files")
            if missing_prev:
                st.write("**Previous CSV missing:**", missing_prev)
            if missing_new:
                st.write("**New CSV missing:**", missing_new)
            
            with st.expander("ℹ️ Required CSV Format"):
                st.markdown("""
                **Required Columns:**
                - `user_input` - The question/query
                - `answer_relevancy` - Semantic relevancy score (0-1)
                - `answer_correctness` - Factual correctness score (0-1)
                - `Precision` - Retrieval precision (0-1)
                - `Recall` - Retrieval recall (0 or 1)
                - `Reranker Score` - Reranker confidence (0-1)
                - `F1 Score` - Harmonic mean of precision and recall (0-1)
                
                **Optional Columns:**
                - `Expected Source` - Expected source document
                - `Actual Sources` - Retrieved source documents
                - `response` - Generated response
                - `reference` - Ground truth reference
                
                **Note:** Both CSV files must have the same column structure.
                """)
            st.stop()
        
        # Numeric columns for comparison
        numeric_cols = ['answer_relevancy', 'answer_correctness', 
                       'Precision', 'Recall', 'Reranker Score', 'F1 Score']
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview", 
            "Combined Data", 
            "Detailed Comparison",
            "Significant Changes",
            "Hallucination Analysis",
            "Retrieval Metrics Changes"
        ])        
           
  

            # TAB 1: Overview
        with tab1:
                st.header("Comparison Overview")
                
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Previous Run",
                        f"{len(df_prev)} rows",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "New Run",
                        f"{len(df_new)} rows",
                        delta=f"{len(df_new) - len(df_prev)} rows"
                    )
                
                with col3:
                    common_rows = len(set(df_prev['user_input']).intersection(set(df_new['user_input'])))
                    st.metric(
                        "Common Questions",
                        f"{common_rows}",
                        delta=None
                    )
                
                st.markdown("---")
                
                # Average metrics comparison
                st.subheader("Average Metrics Comparison")
                
                comparison_data = []
                for col in numeric_cols:
                    prev_mean = df_prev[col].mean()
                    new_mean = df_new[col].mean()
                    diff = new_mean - prev_mean
                    pct_change = (diff / prev_mean * 100) if prev_mean != 0 else 0
                    
                    comparison_data.append({
                        'Metric': col,
                        'Previous': round(prev_mean, 4),
                        'New': round(new_mean, 4),
                        'Difference': round(diff, 4),
                        'Change (%)': round(pct_change, 2)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Style the dataframe
                def highlight_change(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return 'background-color: #d4edda; color: #155724'
                        elif val < 0:
                            return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_df = comparison_df.style.applymap(
                    highlight_change,
                    subset=['Difference', 'Change (%)']
                )
                
                st.dataframe(styled_df, use_container_width=True)
            
            # TAB 2: Combined Data
        with tab2:
                st.header("Combined Data View")
                
                # Merge on user_input
                merged_df = df_prev.merge(
                    df_new,
                    on='user_input',
                    how='outer',
                    suffixes=('_prev', '_new')
                )
                
                # Reorder columns to show prev and new side by side
                ordered_cols = ['user_input']
                for col in df_prev.columns:
                    if col != 'user_input':
                        if f'{col}_prev' in merged_df.columns:
                            ordered_cols.append(f'{col}_prev')
                        if f'{col}_new' in merged_df.columns:
                            ordered_cols.append(f'{col}_new')
                
                # Filter to only include columns that exist
                ordered_cols = [col for col in ordered_cols if col in merged_df.columns]
                merged_df_ordered = merged_df[ordered_cols]
                
                st.info(f"Showing {len(merged_df_ordered)} rows (union of both datasets)")
                
                # Download button
                csv = merged_df_ordered.to_csv(index=False)
                st.download_button(
                    label="Download Combined Data as CSV",
                    data=csv,
                    file_name="combined_replay_comparison.csv",
                    mime="text/csv"
                )
                
                # Display dataframe with search
                st.dataframe(
                    merged_df_ordered,
                    use_container_width=True,
                    height=600
                )
            
            # TAB 3: Detailed Comparison
            
        with tab3:
                st.header("Detailed Metric Comparison")
                
                # Create plots for each metric
                for col in numeric_cols:
                    st.subheader(f"{col} Comparison")
                    
                    # Prepare data
                    prev_avg = df_prev[col].mean()
                    new_avg = df_new[col].mean()
                    prev_median = df_prev[col].median()
                    new_median = df_new[col].median()
                    
                    # Create side-by-side plots
                    col_left, col_right = st.columns(2)
                    
                    # Average comparison
                    with col_left:
                        fig_avg = go.Figure()
                        fig_avg.add_trace(go.Bar(
                            name='Average',
                            x=['Previous', 'New'],
                            y=[prev_avg, new_avg],
                            text=[round(prev_avg, 4), round(new_avg, 4)],
                            textposition='auto',
                            marker_color=['#e74c3c', '#27ae60']
                        ))
                        fig_avg.update_layout(
                            title=f"{col} - Average Comparison",
                            yaxis_title="Average Value",
                            height=350,
                            showlegend=False
                        )
                        st.plotly_chart(fig_avg, use_container_width=True)
                    
                    # Median comparison
                    with col_right:
                        fig_median = go.Figure()
                        fig_median.add_trace(go.Bar(
                            name='Median',
                            x=['Previous', 'New'],
                            y=[prev_median, new_median],
                            text=[round(prev_median, 4), round(new_median, 4)],
                            textposition='auto',
                            marker_color=['#e67e22', '#3498db']
                        ))
                        fig_median.update_layout(
                            title=f"{col} - Median Comparison",
                            yaxis_title="Median Value",
                            height=350,
                            showlegend=False
                        )
                        st.plotly_chart(fig_median, use_container_width=True)
                    
                    # Distribution comparison (side by side)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_hist_prev = px.histogram(
                            df_prev,
                            x=col,
                            nbins=30,
                            title=f"Previous - {col} Distribution",
                            color_discrete_sequence=['#e74c3c']
                        )
                        fig_hist_prev.update_layout(height=300)
                        st.plotly_chart(fig_hist_prev, use_container_width=True)
                    
                    with col2:
                        fig_hist_new = px.histogram(
                            df_new,
                            x=col,
                            nbins=30,
                            title=f"New - {col} Distribution",
                            color_discrete_sequence=['#27ae60']
                        )
                        fig_hist_new.update_layout(height=300)
                        st.plotly_chart(fig_hist_new, use_container_width=True)
                    
                    st.markdown("---")


            # TAB 4: Significant Changes
        with tab4:
                st.header("Significant Changes Analysis")
                
                st.info(f"Showing rows where answer_relevancy or answer_correctness changed by ≥ {threshold}%")
                
                # Merge dataframes on user_input
                merged_for_analysis = df_prev.merge(
                    df_new,
                    on='user_input',
                    how='inner',
                    suffixes=('_prev', '_new')
                )
                
                # Calculate percentage changes for relevancy and correctness
                merged_for_analysis['relevancy_change_pct'] = (
                    (merged_for_analysis['answer_relevancy_new'] - merged_for_analysis['answer_relevancy_prev']) 
                    / merged_for_analysis['answer_relevancy_prev'] * 100
                )
                
                merged_for_analysis['correctness_change_pct'] = (
                    (merged_for_analysis['answer_correctness_new'] - merged_for_analysis['answer_correctness_prev']) 
                    / merged_for_analysis['answer_correctness_prev'] * 100
                )
                
                # Filter rows with significant changes
                significant_changes = merged_for_analysis[
                    (abs(merged_for_analysis['relevancy_change_pct']) >= threshold) |
                    (abs(merged_for_analysis['correctness_change_pct']) >= threshold)
                ].copy()
                
                if len(significant_changes) > 0:
                    # Add improvement indicator
                    def get_indicator(rel_change, corr_change, threshold):
                        rel_improved = rel_change >= threshold
                        corr_improved = corr_change >= threshold
                        rel_degraded = rel_change <= -threshold
                        corr_degraded = corr_change <= -threshold
                        
                        if (rel_improved or corr_improved) and not (rel_degraded or corr_degraded):
                            return "Improved"
                        elif (rel_degraded or corr_degraded) and not (rel_improved or corr_improved):
                            return "Degraded"
                        else:
                            return "Mixed"
                    
                    significant_changes['Status'] = significant_changes.apply(
                        lambda row: get_indicator(
                            row['relevancy_change_pct'], 
                            row['correctness_change_pct'],
                            threshold
                        ),
                        axis=1
                    )
                    
                    # Select columns to display
                    display_cols = [
                        'user_input',
                        'answer_relevancy_prev',
                        'answer_relevancy_new',
                        'relevancy_change_pct',
                        'answer_correctness_prev',
                        'answer_correctness_new',
                        'correctness_change_pct',
                        'Status'
                    ]
                    
                    significant_display = significant_changes[display_cols].copy()
                    
                    # Round numeric columns
                    significant_display['answer_relevancy_prev'] = significant_display['answer_relevancy_prev'].round(4)
                    significant_display['answer_relevancy_new'] = significant_display['answer_relevancy_new'].round(4)
                    significant_display['relevancy_change_pct'] = significant_display['relevancy_change_pct'].round(2)
                    significant_display['answer_correctness_prev'] = significant_display['answer_correctness_prev'].round(4)
                    significant_display['answer_correctness_new'] = significant_display['answer_correctness_new'].round(4)
                    significant_display['correctness_change_pct'] = significant_display['correctness_change_pct'].round(2)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        improved = len(significant_changes[significant_changes['Status'] == 'Improved'])
                        st.metric("Improved", improved, delta=improved, delta_color="normal")
                    
                    with col2:
                        degraded = len(significant_changes[significant_changes['Status'] == 'Degraded'])
                        st.metric("Degraded", degraded, delta=-degraded, delta_color="inverse")
                    
                    with col3:
                        mixed = len(significant_changes[significant_changes['Status'] == 'Mixed'])
                        st.metric("Mixed", mixed)
                    
                    st.markdown("---")

                    # Add filter dropdown for metric selection
                    st.subheader("Filter by Metric")
                    metric_filter = st.selectbox(
                        "Select which metric changes to display:",
                        options=['Both (Relevancy & Correctness)', 'Relevancy Only', 'Correctness Only'],
                        index=0
                    )
                    
                    # Filter based on selection
                    if metric_filter == 'Relevancy Only':
                        significant_display = significant_display[
                            abs(significant_display['relevancy_change_pct']) >= threshold
                        ]
                    elif metric_filter == 'Correctness Only':
                        significant_display = significant_display[
                            abs(significant_display['correctness_change_pct']) >= threshold
                        ]
                    # If 'Both', no additional filtering needed (already filtered above)
                    
                    if len(significant_display) == 0:
                        st.info(f"No changes found for the selected filter: {metric_filter}")
                    # Function to highlight rows
                    def highlight_status(row):
                        if row['Status'] == 'Improved':
                            return ['background-color: #d4edda'] * len(row)
                        elif row['Status'] == 'Degraded':
                            return ['background-color: #f8d7da'] * len(row)
                        else:
                            return ['background-color: #fff3cd'] * len(row)
                    
                    # Display styled dataframe
                    styled_significant = significant_display.style.apply(highlight_status, axis=1)
                    st.dataframe(styled_significant, use_container_width=True, height=600)
                    
                    # Download button
                    csv_sig = significant_display.to_csv(index=False)
                    st.download_button(
                        label="Download Significant Changes as CSV",
                        data=csv_sig,
                        file_name="significant_changes.csv",
                        mime="text/csv"
                    )
                else:
                    st.success(f"No significant changes found (threshold: {threshold}%)")

            # TAB 5: Hallucination Analysis
        with tab5:
                st.header("Hallucination Cases Comparison")
                
                st.markdown("""
                **Hallucination Definition:** High answer relevancy (>0.8) but low answer correctness (<0.4)
                
                These cases indicate the model stays on-topic but provides factually incorrect information.
                """)
                
                # Merge dataframes to get both prev and new values
                merged_hall = df_prev.merge(
                    df_new,
                    on='user_input',
                    how='inner',
                    suffixes=('_prev', '_new')
                )
                
                # Identify hallucinations in both runs (handle NaN values)
                merged_hall['is_hallucination_prev'] = (
                    (merged_hall['answer_relevancy_prev'].notna()) &
                    (merged_hall['answer_correctness_prev'].notna()) &
                    (merged_hall['answer_relevancy_prev'] > 0.8) & 
                    (merged_hall['answer_correctness_prev'] < 0.4)
                )
                
                merged_hall['is_hallucination_new'] = (
                    (merged_hall['answer_relevancy_new'].notna()) &
                    (merged_hall['answer_correctness_new'].notna()) &
                    (merged_hall['answer_relevancy_new'] > 0.8) & 
                    (merged_hall['answer_correctness_new'] < 0.4)
                )
                
                # Check if new run has good quality (both metrics good)
                merged_hall['is_good_quality_new'] = (
                    (merged_hall['answer_relevancy_new'].notna()) &
                    (merged_hall['answer_correctness_new'].notna()) &
                    (merged_hall['answer_relevancy_new'] > 0.8) & 
                    (merged_hall['answer_correctness_new'] >= 0.4)
                )
                
                # Calculate changes
                merged_hall['relevancy_change'] = (
                    merged_hall['answer_relevancy_new'] - merged_hall['answer_relevancy_prev']
                )
                merged_hall['correctness_change'] = (
                    merged_hall['answer_correctness_new'] - merged_hall['answer_correctness_prev']
                )
                
                # Categorize cases with STRICT criteria
                # RESOLVED: Was hallucination AND now is good quality (both metrics good)
                resolved_cases = merged_hall[
                    (merged_hall['is_hallucination_prev'] == True) & 
                    (merged_hall['is_good_quality_new'] == True)
                ].copy()
                
                # NEW CASES: Wasn't hallucination before AND is hallucination now
                new_cases = merged_hall[
                    (merged_hall['is_hallucination_prev'] == False) & 
                    (merged_hall['is_hallucination_new'] == True)
                ].copy()
                
                # PERSISTENT: Was hallucination AND still is hallucination
                persistent_cases = merged_hall[
                    (merged_hall['is_hallucination_prev'] == True) & 
                    (merged_hall['is_hallucination_new'] == True)
                ].copy()
                
                # Count total hallucinations (including those not in common questions, handle NaN)
                total_prev = len(df_prev[
                    (df_prev['answer_relevancy'].notna()) &
                    (df_prev['answer_correctness'].notna()) &
                    (df_prev['answer_relevancy'] > 0.8) & 
                    (df_prev['answer_correctness'] < 0.4)
                ])
                total_new = len(df_new[
                    (df_new['answer_relevancy'].notna()) &
                    (df_new['answer_correctness'].notna()) &
                    (df_new['answer_relevancy'] > 0.8) & 
                    (df_new['answer_correctness'] < 0.4)
                ])
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Previous Run",
                        total_prev,
                        delta=None
                    )
                
                with col2:
                    diff = total_new - total_prev
                    st.metric(
                        "New Run",
                        total_new,
                        delta=diff,
                        delta_color="inverse"  # More hallucinations is bad
                    )
                
                with col3:
                    pct_change = (diff / total_prev * 100) if total_prev > 0 else 0
                    st.metric(
                        "Change",
                        f"{pct_change:.1f}%",
                        delta=diff,
                        delta_color="inverse"
                    )
                
                st.markdown("---")
                
                # Display categories
                tab_resolved, tab_new, tab_persistent = st.tabs([
                    f"Resolved ({len(resolved_cases)})",
                    f"New Cases ({len(new_cases)})",
                    f"Persistent ({len(persistent_cases)})"
                ])
                
                # Helper function to prepare display dataframe
                def prepare_display_df(df):
                    display_df = df[[
                        'user_input',
                        'answer_relevancy_prev',
                        'answer_relevancy_new',
                        'relevancy_change',
                        'answer_correctness_prev',
                        'answer_correctness_new',
                        'correctness_change',
                        'response_prev',
                        'response_new'
                    ]].copy()
                    
                    # Round numeric columns
                    display_df['answer_relevancy_prev'] = display_df['answer_relevancy_prev'].round(4)
                    display_df['answer_relevancy_new'] = display_df['answer_relevancy_new'].round(4)
                    display_df['relevancy_change'] = display_df['relevancy_change'].round(4)
                    display_df['answer_correctness_prev'] = display_df['answer_correctness_prev'].round(4)
                    display_df['answer_correctness_new'] = display_df['answer_correctness_new'].round(4)
                    display_df['correctness_change'] = display_df['correctness_change'].round(4)
                    
                    # Rename columns for clarity
                    display_df.columns = [
                        'Question',
                        'Relevancy (Prev)',
                        'Relevancy (New)',
                        'Relevancy Change',
                        'Correctness (Prev)',
                        'Correctness (New)',
                        'Correctness Change',
                        'Response (Prev)',
                        'Response (New)'
                    ]
                    
                    return display_df
                
                # Resolved cases
                with tab_resolved:
                    if len(resolved_cases) > 0:
                        st.success(f"{len(resolved_cases)} hallucination cases were TRULY resolved in the new run")
                        
                        st.markdown("""
                        **These questions were hallucinations in the previous run and are now high-quality responses.**
                        
                        ✅ **Criteria for Resolved:**
                        - Previous: Relevancy > 0.8 AND Correctness < 0.4 (hallucination)
                        - New: Relevancy > 0.8 AND Correctness ≥ 0.4 (good quality)
                        
                        Look for:
                        - Increased correctness (positive change)
                        - Maintained high relevancy (> 0.8)
                        """)
                        
                        resolved_display = prepare_display_df(resolved_cases)
                        
                        # Highlight improvements
                        def highlight_resolved(row):
                            colors = [''] * len(row)
                            # Highlight correctness improvement in green
                            if 'Correctness Change' in row.index:
                                idx = list(row.index).index('Correctness Change')
                                if row['Correctness Change'] > 0:
                                    colors[idx] = 'background-color: #d4edda; font-weight: bold'
                            # Highlight relevancy if maintained high
                            if 'Relevancy (New)' in row.index:
                                idx = list(row.index).index('Relevancy (New)')
                                if row['Relevancy (New)'] > 0.8:
                                    colors[idx] = 'background-color: #d4edda'
                            return colors
                        
                        styled_resolved = resolved_display.style.apply(highlight_resolved, axis=1)
                        st.dataframe(styled_resolved, use_container_width=True, height=400)
                        
                        # Download button
                        csv_resolved = resolved_display.to_csv(index=False)
                        st.download_button(
                            label="Download Resolved Cases",
                            data=csv_resolved,
                            file_name="resolved_hallucinations.csv",
                            mime="text/csv",
                            key="download_resolved"
                        )
                    else:
                        st.info("No hallucinations were truly resolved (both metrics good in new run)")
                
                # New cases
                with tab_new:
                    if len(new_cases) > 0:
                        st.warning(f"{len(new_cases)} new hallucination cases appeared in the new run")
                        
                        st.markdown("""
                        **These questions became hallucinations in the new run (regression).**
                        
                        ⚠️ **Criteria for New Cases:**
                        - Previous: NOT a hallucination (either relevancy ≤ 0.8 OR correctness ≥ 0.4)
                        - New: Relevancy > 0.8 AND Correctness < 0.4 (hallucination)
                        
                        Look for:
                        - Decreased correctness (negative change)
                        - High relevancy maintained but accuracy dropped
                        """)
                        
                        new_display = prepare_display_df(new_cases)
                        
                        # Highlight degradations
                        def highlight_new(row):
                            colors = [''] * len(row)
                            # Highlight correctness decrease in red
                            if 'Correctness Change' in row.index:
                                idx = list(row.index).index('Correctness Change')
                                if row['Correctness Change'] < 0:
                                    colors[idx] = 'background-color: #f8d7da; font-weight: bold'
                            # Highlight high relevancy in yellow (on-topic but wrong)
                            if 'Relevancy (New)' in row.index:
                                idx = list(row.index).index('Relevancy (New)')
                                if row['Relevancy (New)'] > 0.8:
                                    colors[idx] = 'background-color: #fff3cd'
                            return colors
                        
                        styled_new = new_display.style.apply(highlight_new, axis=1)
                        st.dataframe(styled_new, use_container_width=True, height=400)
                        
                        # Download button
                        csv_new = new_display.to_csv(index=False)
                        st.download_button(
                            label="Download New Cases",
                            data=csv_new,
                            file_name="new_hallucinations.csv",
                            mime="text/csv",
                            key="download_new"
                        )
                    else:
                        st.success("No new hallucinations appeared")
                
                # Persistent cases
                with tab_persistent:
                    if len(persistent_cases) > 0:
                        st.error(f"{len(persistent_cases)} hallucination cases persist across both runs")
                        
                        st.markdown("""
                        **These questions remain hallucinations in both runs.**
                        
                        🔴 **Criteria for Persistent:**
                        - Previous: Relevancy > 0.8 AND Correctness < 0.4 (hallucination)
                        - New: Relevancy > 0.8 AND Correctness < 0.4 (still hallucination)
                        
                        Look for:
                        - Whether correctness is improving or worsening (even if still below 0.4)
                        - Cases where correctness improved but still below threshold
                        - Consistently problematic questions that need attention
                        """)
                        
                        persistent_display = prepare_display_df(persistent_cases)
                        
                        # Highlight trend
                        def highlight_persistent(row):
                            colors = [''] * len(row)
                            if 'Correctness Change' in row.index:
                                idx = list(row.index).index('Correctness Change')
                                if row['Correctness Change'] > 0.1:
                                    colors[idx] = 'background-color: #fff3cd; font-weight: bold'  # Yellow for improving
                                elif row['Correctness Change'] < -0.1:
                                    colors[idx] = 'background-color: #f8d7da; font-weight: bold'  # Red for worsening
                            # Keep relevancy highlighted to show it's still high
                            if 'Relevancy (New)' in row.index:
                                idx = list(row.index).index('Relevancy (New)')
                                if row['Relevancy (New)'] > 0.8:
                                    colors[idx] = 'background-color: #fff3cd'
                            return colors
                        
                        styled_persistent = persistent_display.style.apply(highlight_persistent, axis=1)
                        st.dataframe(styled_persistent, use_container_width=True, height=400)
                        
                        # Show trend analysis
                        improving = len(persistent_cases[persistent_cases['correctness_change'] > 0.1])
                        worsening = len(persistent_cases[persistent_cases['correctness_change'] < -0.1])
                        stable = len(persistent_cases[abs(persistent_cases['correctness_change']) <= 0.1])
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Improving (but still hallucination)", improving, delta=improving, delta_color="normal")
                        with col_b:
                            st.metric("Worsening", worsening, delta=-worsening, delta_color="inverse")
                        with col_c:
                            st.metric("Stable", stable)
                        
                        # Download button
                        csv_persistent = persistent_display.to_csv(index=False)
                        st.download_button(
                            label="Download Persistent Cases",
                            data=csv_persistent,
                            file_name="persistent_hallucinations.csv",
                            mime="text/csv",
                            key="download_persistent"
                        )
                    else:
                        st.success("No persistent hallucinations")
                
                st.markdown("---")
                
                # Visualization
                st.subheader("Hallucination Trend")
                
                fig_hall = go.Figure()
                
                # Add bars for each category
                fig_hall.add_trace(go.Bar(
                    name='Total',
                    x=['Previous Run', 'New Run'],
                    y=[total_prev, total_new],
                    text=[total_prev, total_new],
                    textposition='auto',
                    marker_color=['#3498db', '#2ecc71' if total_new < total_prev else '#e74c3c']
                ))
                
                fig_hall.update_layout(
                    title="Hallucination Count Comparison",
                    yaxis_title="Number of Cases",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_hall, use_container_width=True)
                
                # Breakdown pie chart
                if len(resolved_cases) + len(new_cases) + len(persistent_cases) > 0:
                    st.subheader("Case Breakdown (Common Questions)")
                    
                    breakdown_data = pd.DataFrame({
                        'Category': ['Resolved', 'New', 'Persistent'],
                        'Count': [len(resolved_cases), len(new_cases), len(persistent_cases)]
                    })
                    
                    # Filter out zero values for cleaner pie chart
                    breakdown_data = breakdown_data[breakdown_data['Count'] > 0]
                    
                    if len(breakdown_data) > 0:
                        fig_pie = px.pie(
                            breakdown_data,
                            values='Count',
                            names='Category',
                            color='Category',
                            color_discrete_map={
                                'Resolved': '#2ecc71',
                                'New': '#e74c3c',
                                'Persistent': '#f39c12'
                            }
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
                        st.plotly_chart(fig_pie, use_container_width=True)

            # TAB 6: Retrieval Metrics 
        with tab6:
                st.header("Retrieval Metrics Changes Analysis")
                
                st.info(f"Showing rows where Precision, Recall, or Reranker Score changed (excluding identical values)")
                
                # Merge dataframes on user_input
                merged_retrieval = df_prev.merge(
                    df_new,
                    on='user_input',
                    how='inner',
                    suffixes=('_prev', '_new')
                )
                
                # Calculate changes
                merged_retrieval['precision_change'] = (
                    merged_retrieval['Precision_new'] - merged_retrieval['Precision_prev']
                )
                merged_retrieval['recall_change'] = (
                    merged_retrieval['Recall_new'] - merged_retrieval['Recall_prev']
                )
                merged_retrieval['reranker_change'] = (
                    merged_retrieval['Reranker Score_new'] - merged_retrieval['Reranker Score_prev']
                )
                
                # Filter rows where at least one metric changed (not zero change)
                retrieval_changes = merged_retrieval[
                    (merged_retrieval['precision_change'] != 0) |
                    (merged_retrieval['recall_change'] != 0) |
                    (merged_retrieval['reranker_change'] != 0)
                ].copy()
                
                if len(retrieval_changes) > 0:

                    # Categorize overall change status
                    def get_retrieval_status(prec_change, rec_change, rerank_change):
                        # Count improvements and degradations
                        improvements = sum([
                            prec_change > 0,
                            rec_change > 0,
                            rerank_change > 0
                        ])
                        degradations = sum([
                            prec_change < 0,
                            rec_change < 0,
                            rerank_change < 0
                        ])
                        
                        if improvements > 0 and degradations == 0:
                            return "Improved"
                        elif degradations > 0 and improvements == 0:
                            return "Degraded"
                        else:
                            return "Mixed"
                    
                    retrieval_changes['Status'] = retrieval_changes.apply(
                        lambda row: get_retrieval_status(
                            row['precision_change'],
                            row['recall_change'],
                            row['reranker_change']
                        ),
                        axis=1
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Changes",
                            len(retrieval_changes)
                        )
                    
                    with col2:
                        improved = len(retrieval_changes[retrieval_changes['Status'] == 'Improved'])
                        st.metric("Improved", improved, delta=improved, delta_color="normal")
                    
                    with col3:
                        degraded = len(retrieval_changes[retrieval_changes['Status'] == 'Degraded'])
                        st.metric("Degraded", degraded, delta=-degraded, delta_color="inverse")
                    
                    with col4:
                        mixed = len(retrieval_changes[retrieval_changes['Status'] == 'Mixed'])
                        st.metric("Mixed", mixed)
                    
                    st.markdown("---")
                    
                    # Prepare display dataframe with source columns
                    display_cols = [
                        'user_input',
                        'Expected Source_prev',
                        'Expected Source_new',
                        'Actual Sources_prev',
                        'Actual Sources_new',
                        'Precision_prev',
                        'Precision_new',
                        'precision_change',
                        'Recall_prev',
                        'Recall_new',
                        'recall_change',
                        'Reranker Score_prev',
                        'Reranker Score_new',
                        'reranker_change',
                        'Status'
                    ]
                    
                    # Check which source columns exist (they might have different names)
                    available_cols = []
                    for col in display_cols:
                        if col in retrieval_changes.columns:
                            available_cols.append(col)
                        # Handle case where column might not exist
                        elif col == 'Expected Source_prev' and 'Expected Source' in df_prev.columns:
                            available_cols.append('Expected Source_prev')
                        elif col == 'Expected Source_new' and 'Expected Source' in df_new.columns:
                            available_cols.append('Expected Source_new')
                        elif col == 'Actual Sources_prev' and 'Actual Sources' in df_prev.columns:
                            available_cols.append('Actual Sources_prev')
                        elif col == 'Actual Sources_new' and 'Actual Sources' in df_new.columns:
                            available_cols.append('Actual Sources_new')
                    
                    # If source columns don't exist, use the original display_cols without them
                    if 'Expected Source_prev' not in retrieval_changes.columns:
                        available_cols = [col for col in display_cols if 'Source' not in col]
                    
                    retrieval_display = retrieval_changes[available_cols].copy()
                    
                    # Round numeric columns
                    for col in ['Precision_prev', 'Precision_new', 'precision_change',
                               'Recall_prev', 'Recall_new', 'recall_change',
                               'Reranker Score_prev', 'Reranker Score_new', 'reranker_change']:
                        if col in retrieval_display.columns:
                            retrieval_display[col] = retrieval_display[col].round(4)
                    
                    # Rename columns for better display
                    column_rename_map = {
                        'user_input': 'Question',
                        'Expected Source_prev': 'Expected Source (Prev)',
                        'Expected Source_new': 'Expected Source (New)',
                        'Actual Sources_prev': 'Actual Sources (Prev)',
                        'Actual Sources_new': 'Actual Sources (New)',
                        'Precision_prev': 'Precision (Prev)',
                        'Precision_new': 'Precision (New)',
                        'precision_change': 'Precision Change',
                        'Recall_prev': 'Recall (Prev)',
                        'Recall_new': 'Recall (New)',
                        'recall_change': 'Recall Change',
                        'Reranker Score_prev': 'Reranker (Prev)',
                        'Reranker Score_new': 'Reranker (New)',
                        'reranker_change': 'Reranker Change',
                        'Status': 'Status'
                    }
                    
                    # Only rename columns that exist
                    rename_dict = {k: v for k, v in column_rename_map.items() if k in retrieval_display.columns}
                    retrieval_display.columns = [rename_dict.get(col, col) for col in retrieval_display.columns]
                    
                    # Function to highlight changes - cell by cell
                    def highlight_retrieval_changes(val, col_name):
                        if 'Change' in col_name and col_name != 'Status':
                            if isinstance(val, (int, float)):
                                if val > 0:
                                    return 'background-color: #d4edda; font-weight: bold'
                                elif val < 0:
                                    return 'background-color: #f8d7da; font-weight: bold'
                        elif col_name == 'Status':
                            if val == 'Improved':
                                return 'background-color: #d4edda; font-weight: bold'
                            elif val == 'Degraded':
                                return 'background-color: #f8d7da; font-weight: bold'
                            elif val == 'Mixed':
                                return 'background-color: #fff3cd'
                        return ''
                    
                    # Apply styling
                    styled_retrieval = retrieval_display.style.applymap(
                        lambda val: highlight_retrieval_changes(val, retrieval_display.columns[retrieval_display.columns.get_loc(val) if val in retrieval_display.columns else 0]),
                        subset=['Precision Change', 'Recall Change', 'Reranker Change', 'Status']
                    )
                    
                    # Simpler styling approach
                    def highlight_row_by_status(row):
                        colors = [''] * len(row)
                        status_idx = list(row.index).index('Status')
                        
                        # Color the change columns based on their values
                        for idx, col in enumerate(row.index):
                            if 'Change' in col and col != 'Status':
                                if row[col] > 0:
                                    colors[idx] = 'background-color: #d4edda; font-weight: bold'
                                elif row[col] < 0:
                                    colors[idx] = 'background-color: #f8d7da; font-weight: bold'
                            elif col == 'Status':
                                if row[col] == 'Improved':
                                    colors[idx] = 'background-color: #d4edda; font-weight: bold'
                                elif row[col] == 'Degraded':
                                    colors[idx] = 'background-color: #f8d7da; font-weight: bold'
                                elif row[col] == 'Mixed':
                                    colors[idx] = 'background-color: #fff3cd'
                        
                        return colors
                    
                    styled_retrieval = retrieval_display.style.apply(highlight_row_by_status, axis=1)
                    
                    # Display with filters
                    st.subheader("Filter by Status")
                    status_filter = st.multiselect(
                        "Select status to display:",
                        options=['Improved', 'Degraded', 'Mixed'],
                        default=['Improved', 'Degraded', 'Mixed']
                    )
                    
                    if status_filter:
                        filtered_display = retrieval_display[retrieval_display['Status'].isin(status_filter)]
                        styled_filtered = filtered_display.style.apply(highlight_row_by_status, axis=1)
                        st.dataframe(styled_filtered, use_container_width=True, height=600)
                    else:
                        st.warning("Please select at least one status filter")
                    
                    # Download button
                    csv_retrieval = retrieval_display.to_csv(index=False)
                    st.download_button(
                        label="Download Retrieval Changes as CSV",
                        data=csv_retrieval,
                        file_name="retrieval_metrics_changes.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("---")
                    
                    # Detailed breakdown by metric
                    st.subheader("Metric-wise Breakdown")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**Precision Changes**")
                        prec_improved = len(retrieval_changes[retrieval_changes['precision_change'] > 0])
                        prec_degraded = len(retrieval_changes[retrieval_changes['precision_change'] < 0])
                        prec_same = len(retrieval_changes[retrieval_changes['precision_change'] == 0])
                        
                        st.write(f"✅ Improved: {prec_improved}")
                        st.write(f"❌ Degraded: {prec_degraded}")
                        st.write(f"➖ Unchanged: {prec_same}")
                    
                    with col_b:
                        st.markdown("**Recall Changes**")
                        rec_improved = len(retrieval_changes[retrieval_changes['recall_change'] > 0])
                        rec_degraded = len(retrieval_changes[retrieval_changes['recall_change'] < 0])
                        rec_same = len(retrieval_changes[retrieval_changes['recall_change'] == 0])
                        
                        st.write(f"✅ Improved: {rec_improved}")
                        st.write(f"❌ Degraded: {rec_degraded}")
                        st.write(f"➖ Unchanged: {rec_same}")
                    
                    with col_c:
                        st.markdown("**Reranker Score Changes**")
                        rerank_improved = len(retrieval_changes[retrieval_changes['reranker_change'] > 0])
                        rerank_degraded = len(retrieval_changes[retrieval_changes['reranker_change'] < 0])
                        rerank_same = len(retrieval_changes[retrieval_changes['reranker_change'] == 0])
                        
                        st.write(f"✅ Improved: {rerank_improved}")
                        st.write(f"❌ Degraded: {rerank_degraded}")
                        st.write(f"➖ Unchanged: {rerank_same}")
                    
                    # Visualization - Change distribution
                    st.markdown("---")
                    st.subheader("Change Distribution")
                    
                    change_data = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'Reranker Score'],
                        'Improved': [prec_improved, rec_improved, rerank_improved],
                        'Degraded': [prec_degraded, rec_degraded, rerank_degraded],
                        'Unchanged': [prec_same, rec_same, rerank_same]
                    })
                    
                    fig_changes = go.Figure()
                    
                    fig_changes.add_trace(go.Bar(
                        name='Improved',
                        x=change_data['Metric'],
                        y=change_data['Improved'],
                        marker_color='#2ecc71',
                        text=change_data['Improved'],
                        textposition='auto'
                    ))
                    
                    fig_changes.add_trace(go.Bar(
                        name='Degraded',
                        x=change_data['Metric'],
                        y=change_data['Degraded'],
                        marker_color='#e74c3c',
                        text=change_data['Degraded'],
                        textposition='auto'
                    ))
                    
                    fig_changes.add_trace(go.Bar(
                        name='Unchanged',
                        x=change_data['Metric'],
                        y=change_data['Unchanged'],
                        marker_color='#95a5a6',
                        text=change_data['Unchanged'],
                        textposition='auto'
                    ))
                    
                    fig_changes.update_layout(
                        barmode='group',
                        title="Retrieval Metrics Change Distribution",
                        xaxis_title="Metric",
                        yaxis_title="Number of Questions",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_changes, use_container_width=True)
                    
                else:
                    st.success("No changes detected in Precision, Recall, or Reranker Score across common questions")
            
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("Please ensure the CSV files exist in the 'stored_csvs' folder")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("❌ Error: One or both files are empty or corrupted")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.info("Please ensure your CSV files are properly formatted")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.exception(e)
        st.stop() 
    
else:
    # Display placeholder when no files loaded
    st.info("👈 Please select a data source from the sidebar to begin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📤 Upload CSV Files
        
        **Use this option if:**
        - You have CSV files on your local machine
        - You want to compare new replay runs
        - Files are not stored in the repository
        
        **Steps:**
        1. Select "Upload CSV Files"
        2. Upload Previous Run CSV
        3. Upload New Run CSV
        4. Start comparing!
        """)
    
    with col2:
        st.markdown("""
        ### 📂 Select from Available Files
        
        **Use this option if:**
        - CSV files are already in the repository
        - You want quick access to stored runs
        - Comparing historical replay data
        
        **Steps:**
        1. Select "Select from Available Files"
        2. Choose Previous Run from dropdown
        3. Choose New Run from dropdown
        4. Start comparing!
        """)
    
    with st.expander("📋 Required CSV Format"):
        st.markdown("""
        **Required Columns:**
        - `user_input` - The question/query
        - `answer_relevancy` - Semantic relevancy score (0-1)
        - `answer_correctness` - Factual correctness score (0-1)
        - `Precision` - Retrieval precision (0-1)
        - `Recall` - Retrieval recall (0 or 1)
        - `Reranker Score` - Reranker confidence (0-1)
        - `F1 Score` - Harmonic mean of precision and recall (0-1)

        """)

    

