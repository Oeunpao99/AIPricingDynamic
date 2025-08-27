import sys
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_chat import message
from datetime import datetime

# --- Path Handling ---
def get_file_path(filename):
    """
    Get the absolute path to a file in the same directory as this script.
    Works for both development and bundled/PyInstaller environments.
    """
    if getattr(sys, 'frozen', False):
        # The application is frozen (bundled)
        application_path = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(application_path, filename)

# --- File Paths ---
ICON_PATH = get_file_path("price_icon.png")
MODEL_PATH = get_file_path("dynamic_price_model.pkl")
COLUMNS_PATH = get_file_path("model_column.pkl")
DATA_PATH = get_file_path("sales_project_training_data_remove_columns.csv")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #166312;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background-color: #166312;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color:#166316;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .category-button {
        width: 100%;
        margin-bottom: 10px;
        border-radius: 8px;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Pricing Assistant",
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else "💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cached Data Loading ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Data file not found at: {DATA_PATH}")
            return pd.DataFrame()  # Return empty DataFrame to allow app to continue
        
        data = pd.read_csv(DATA_PATH)
        # Data quality checks
        required_columns = ["quote_number", "selling", "costing", "project_status"]
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"Critical columns missing: {', '.join(missing_cols)}")
            return pd.DataFrame()  # Return empty DataFrame to allow app to continue
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame to allow app to continue

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
            st.error("Model files not found! Please ensure dynamic_price_model.pkl and model_column.pkl are in the correct directory.")
            return None, None
            
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLUMNS_PATH)
        return model, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize components
data = load_data()
model, feature_columns = load_model()

def show_pricing_tool():
    """Main pricing tool interface"""
    st.markdown('<h1 class="main-header">📊 AI Pricing & Business Strategy Assistant</h1>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.markdown("""
        <div class="highlight">
        1. Enter a quote number to load existing data<br>
        2. Adjust the parameters as needed<br>
        3. Click 'Generate Pricing & Advice'<br>
        4. Review the recommendations<br>
        5. Ask the advisor any follow-up questions
        </div>
        """, unsafe_allow_html=True)

    quote_number = st.text_input("🔍 Enter Quote Number (e.g., QT-4351)", key="quote_input")
    selected_row = data[data["quote_number"] == quote_number].iloc[0] if quote_number in data["quote_number"].values else None

    if selected_row is not None:
        st.success(f" Quote found: {quote_number}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sub-header">Cost & Demand Factors</div>', unsafe_allow_html=True)
            costing = st.number_input("Our Product Cost ($)", value=float(selected_row["costing"]), key="cost_input")
            demand_level = st.selectbox("Demand Level", ["Low", "Medium", "High"], 
                                      index=["Low", "Medium", "High"].index(selected_row["Demand_Level"]), key="demand_input")
            stock_availability = st.selectbox("Stock Availability", ["In Stock", "Low Stock", "Out of Stock"], 
                                            index=["In Stock", "Low Stock", "Out of Stock"].index(selected_row["Stock_Availability"]), key="stock_input")
        
        with col2:
            st.markdown('<div class="sub-header">Pricing & Customer Factors</div>', unsafe_allow_html=True)
            installation_cost = st.number_input("Installation Cost ($)", value=float(selected_row["Installation_Cost"]), key="install_input")
            customer_type = st.selectbox("Customer Type", ["Corporate", "Enterprise", "Government", "SME"], 
                                        index=["Corporate", "Enterprise", "Government", "SME"].index(selected_row["Customer_Type"]), key="customer_input")
            est_competitor_cost = st.number_input("Competitor's Cost ($)", value=float(selected_row["est_competitor_cost"]), key="competitor_input")
        
        st.markdown('<div class="sub-header">Pricing Strategy</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            comp_markup = st.slider("Competitor's Typical Markup (%)", 10, 50, 25, key="comp_markup_slider")
        with col4:
            our_min_margin = st.slider("Our Minimum Acceptable Margin (%)", 5, 30, 15, key="min_margin_slider")

        if st.button("💡 Generate Pricing & Advice", key="generate_button"):
            with st.spinner("Analyzing market conditions and generating recommendations..."):
                # Prepare input
                new_row = pd.DataFrame([{
                    "costing": costing,
                    "Demand_Level": demand_level,
                    "Stock_Availability": stock_availability,
                    "Installation_Cost": installation_cost,
                    "Customer_Type": customer_type,
                    "est_competitor_cost": est_competitor_cost
                }])
                
                new_row_encoded = pd.get_dummies(new_row, drop_first=True)
                new_row_encoded = new_row_encoded.reindex(columns=feature_columns, fill_value=0)

                # Generate predictions
                base_price = model.predict(new_row_encoded)[0]
                competitor_price = est_competitor_cost * (1 + comp_markup / 100)
                our_min_price = costing * (1 + our_min_margin / 100)
                competitive_price = min(base_price, competitor_price * 0.95)
                competitive_price = max(competitive_price, our_min_price)
                competitive_margin = (competitive_price - costing) / competitive_price * 100

                # Store context for advisor
                st.session_state.context_data = {
                    "base_price": base_price,
                    "competitor_price": competitor_price,
                    "competitive_price": competitive_price,
                    "costing": costing,
                    "competitive_margin": competitive_margin
                }

                # Display results in metric cards
                st.markdown('<div class="sub-header">Pricing Recommendations</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Model Suggested Price</div>
                        <div class="metric-value">${base_price:,.2f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Recommended Competitive Price</div>
                        <div class="metric-value">${competitive_price:,.2f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Projected Margin</div>
                        <div class="metric-value">{competitive_margin:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Price Comparison Bar Chart", "Pie Chart", "Line Chart"])
                
                with tab1:
                    # Price comparison chart
                    chart_df = pd.DataFrame({
                        "Category": ["Our Cost", "Competitor Price", "Our Suggested Price"],
                        "Value": [costing, competitor_price, competitive_price]
                    })
                    
                    fig = px.bar(chart_df, x="Category", y="Value", text="Value",
                                title="Price Comparison Analysis",
                                color="Category",
                                color_discrete_map={
                                    "Our Cost": "#ed0a0a",
                                    "Competitor Price": "#ff7700",
                                    "Our Suggested Price": "#166312"
                                })
                    fig.update_traces(texttemplate='$%{text:,.2f}', textposition='outside')
                    fig.update_layout(
                        uniformtext_minsize=8, 
                        uniformtext_mode='hide',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Pie chart version
                    fig_pie = px.pie(
                        chart_df,
                        names="Category",
                        values="Value",
                        title="Price Distribution",
                        color="Category",
                        color_discrete_map={
                            "Our Cost": "#ed0a0a",
                            "Competitor Price": "#ff7700",
                            "Our Suggested Price": "#166312"
                        },
                        hole=0.4  # donut style
                    )
                    fig_pie.update_traces(
                        textinfo="label+percent", 
                        pull=[0, 0.05, 0],  # slight separation for competitor
                        textfont_size=14
                    )
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
                    # Line chart
                    fig_line = px.line(
                        chart_df,
                        x="Category",
                        y="Value",
                        markers=True,
                        title="Price Comparison Trend",
                        color="Category",
                        color_discrete_map={
                            "Our Cost": "#ed0a0a",
                            "Competitor Price": "#ff7700",
                            "Our Suggested Price": "#0ab30a"
                        }
                    )
                    fig_line.update_traces(
                        text=chart_df["Value"], 
                        textposition="top center",
                        line=dict(width=4)
                    )
                    fig_line.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14),
                        yaxis_title="Price ($)",
                        xaxis_title=""
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                # Advisor insights
                st.markdown('<div class="sub-header">📋 Business Advisor Insights</div>', unsafe_allow_html=True)
                
                # Create expandable sections for different insights
                with st.expander("💰 Pricing Strategy", expanded=True):
                    st.info(f"""
                    - Our AI model suggests a base price of **${base_price:,.2f}**
                    - After considering competitor pricing and your minimum margin, we recommend **${competitive_price:,.2f}**
                    - This gives you a healthy profit margin of **{competitive_margin:.1f}%**
                    """)
                
                with st.expander("📈 Market Position"):
                    price_difference = competitive_price - competitor_price
                    if price_difference < 0:
                        st.success(f"**Competitive Advantage:** You're priced ${abs(price_difference):.2f} below your competitor")
                    else:
                        st.warning(f"**Price Premium:** You're priced ${price_difference:.2f} above your competitor")
                
                with st.expander("🎯 Actionable Recommendations"):
                    if competitive_margin < our_min_margin:
                        st.error("**Warning:** Your projected margin is below your minimum acceptable margin. Consider:")
                        st.write("- Negotiating better supplier prices")
                        st.write("- Reducing installation costs")
                        st.write("- Adding value to justify a higher price")
                    else:
                        st.success("**Good News:** Your pricing strategy meets your margin requirements!")
                
                context_str = f"""
                Base model price: ${base_price:,.2f}
                Competitor price: ${competitor_price:,.2f}
                Recommended selling price: ${competitive_price:,.2f}
                Our cost: ${costing:,.2f}
                Profit margin: {competitive_margin:.1f}%
                """

                # Download button
                csv = pd.DataFrame({
                    "Metric": ["Base Price", "Competitor Price", "Recommended Price", "Cost", "Margin"],
                    "Value": [base_price, competitor_price, competitive_price, costing, competitive_margin]
                }).to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Pricing Analysis",
                    data=csv,
                    file_name=f"pricing_analysis_{quote_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_button"
                )

                # Initialize chat history if not exists
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

    elif quote_number:
        st.error("Quote number not found in data. Please check the number and try again.")

def show_data_insights():
    st.markdown('<h1 class="main-header">📊 Your Business Insights</h1>', unsafe_allow_html=True)

    data = load_data()
    if data.empty:
        st.warning("Please upload a CSV file to view insights.")
        return

    question_categories = {
        "Revenue Insights": [
            "Which product line earns the most revenue?",
            "What is the total revenue by customer type?",
            "What are the top projects by revenue?",
            "Who are the top customers by revenue?",
            "Which customer types generate the most revenue?",
        ],
        "Win/Loss Analysis": [
            "What is the overall win rate?",
            "How many quotes resulted in Won vs Lost?",
            "Which product lines have the highest loss rate?",
            "How much revenue was lost due to lost projects?",
        ],
        "Customer & Market Analysis": [
            "What is the average customer budget by segment?",
            "What is the average competitor cost vs our selling price?",
            "Are we pricing competitively?",
        ],
        "Profitability": [
            "What is the average profit margin per product?",
        ],
        "Demand & Stock": [
            "What is the demand level distribution?",
            "How does stock availability affect win rate?",
        ],
        "Cost Analysis": [
            "Is installation cost impacting the winning rate?",
        ],
        "Pipeline & Trends": [
            "What is the pipeline status distribution?",
            "How many quotes are active, lost, or won per month?",
            "Which projects are pending and likely to convert?",
        ],
        "Vendor & Product Insights": [
            "Which brands/vendors have the most successful quotes?",
            "How often do different product versions sell?",
            "What is the average quantity sold per product/customer?",
        ]
    }

    # Initialize session state for selected category if not present
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
        
    # Display categories as clickable cards (buttons)
    st.markdown('<div class="sub-header">Select an Analysis Category</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    cat_list = list(question_categories.keys())
    for i, cat in enumerate(cat_list):
        col = cols[i % 4]
        if col.button(cat, key=f"cat_{i}", use_container_width=True):
            st.session_state.selected_category = cat

    if st.session_state.selected_category:
        st.markdown(f'<div class="sub-header">{st.session_state.selected_category}</div>', unsafe_allow_html=True)
        questions = question_categories[st.session_state.selected_category]
        question_choice = st.selectbox("Choose a question:", ["Select a question"] + questions, key="question_select")

        if question_choice != "Select a question":
            # -------------------------
            # INSIGHT LOGIC (your existing code)
            # -------------------------
            if question_choice == "Which product line earns the most revenue?":
                revenue_by_product = data.groupby("product_line")["selling"].sum()
                top_product = revenue_by_product.idxmax()
                top_revenue = revenue_by_product.max()
                st.write(f"**Top Product Line by Revenue:** {top_product} with total revenue ${top_revenue:,.2f}")
                fig = px.pie(revenue_by_product.reset_index(), values="selling", names="product_line",
                             title="Revenue Distribution by Product Line",
                             color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig)

            elif question_choice == "What is the total revenue by customer type?":
                revenue_by_customer = data.groupby("Customer_Type")["selling"].sum()
                st.write("**Total Revenue by Customer Type:**")
                st.dataframe(revenue_by_customer)
                fig = px.bar(revenue_by_customer.reset_index(), x="Customer_Type", y="selling", 
                             title="Revenue by Customer Type",
                             color="Customer_Type",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig)

            elif question_choice == "What are the top projects by revenue?":
                revenue_by_project = data.groupby("project_name")["selling"].sum().sort_values(ascending=False).head(5)
                st.write("**Top 5 Projects by Revenue:**")
                st.dataframe(revenue_by_project)
                fig = px.bar(revenue_by_project.reset_index(), x="project_name", y="selling", 
                             title="Top 5 Projects by Revenue",
                             color="project_name",
                             color_discrete_sequence=[
                                 "rgba(31, 119, 180, 1)", 
                                 "rgba(255, 127, 14, 1)", 
                                 "rgba(44, 160, 44, 1)", 
                                 "rgba(214, 39, 40, 1)", 
                                 "rgba(148, 103, 189, 1)"
                             ])
                st.plotly_chart(fig)

            elif question_choice == "What is the overall win rate?":
                win_count = data[data["project_status"] == "Won"].shape[0]
                total = data.shape[0]
                win_rate = win_count / total * 100
                st.write(f"**Overall Win Rate:** {win_rate:.2f}%")

            elif question_choice == "How many quotes resulted in Won vs Lost?":
                status_counts = data["project_status"].value_counts()
                st.write("**Quotes Status Counts:**")
                st.dataframe(status_counts)
                df_status = status_counts.reset_index()
                df_status.columns = ['project_status', 'count']
                fig = px.pie(df_status, values="count", names="project_status", 
                             title="Quotes Won vs Lost",
                             color="project_status",
                             color_discrete_map={"Won": "green", "Lost": "red"})
                st.plotly_chart(fig)

            elif question_choice == "Which product lines have the highest loss rate?":
                grouped = data.groupby(["product_line", "project_status"]).size().unstack(fill_value=0)
                grouped["loss_rate"] = grouped.get("Lost", 0) / grouped.sum(axis=1)
                highest_loss = grouped["loss_rate"].idxmax()
                st.write(f"**Product Line with Highest Loss Rate:** {highest_loss}")
                st.dataframe(grouped)

            elif question_choice == "How much revenue was lost due to lost projects?":
                lost_revenue = data[data["project_status"] == "Lost"]["selling"].sum()
                st.write(f"**Total Revenue Lost due to Lost Projects:** ${lost_revenue:,.2f}")

            elif question_choice == "Who are the top customers by revenue?":
                revenue_by_customer = data.groupby("Customer_Type")["selling"].sum().sort_values(ascending=False).head(10)
                st.write("**Top 10 Customers by Revenue:**")
                st.dataframe(revenue_by_customer)
                fig = px.bar(revenue_by_customer.reset_index(), x="Customer_Type", y="selling", 
                             title="Top Customers by Revenue",
                             color="Customer_Type",
                             color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig)

            elif question_choice == "Which customer types generate the most revenue?":
                revenue_by_custype = data.groupby("Customer_Type")["selling"].sum()
                st.write("**Revenue by Customer Type:**")
                st.dataframe(revenue_by_custype)
                fig = px.bar(revenue_by_custype.reset_index(), x="Customer_Type", y="selling", 
                             title="Revenue by Customer Type",
                             color="Customer_Type",
                             color_discrete_sequence=px.colors.qualitative.Pastel1)
                st.plotly_chart(fig)

            elif question_choice == "What is the average customer budget by segment?":
                avg_budget = data.groupby("Customer_Type")["customer_budget"].mean()
                st.write("**Average Customer Budget by Segment:**")
                st.dataframe(avg_budget)

            elif question_choice == "What is the average competitor cost vs our selling price?":
                avg_competitor_cost = data["est_competitor_cost"].mean()
                avg_selling_price = data["selling"].mean()
                st.write(f"**Average Competitor Cost:** ${avg_competitor_cost:,.2f}")
                st.write(f"**Average Selling Price:** ${avg_selling_price:,.2f}")

            elif question_choice == "Are we pricing competitively?":
                competitively_priced = (data["selling"] < data["est_competitor_cost"]).mean() * 100
                st.write(f"**Percentage of quotes priced below competitor:** {competitively_priced:.2f}%")

            elif question_choice == "What is the average profit margin per product?":
                data["profit_margin"] = (data["selling"] - data["costing"]) / data["selling"] * 100
                avg_margin = data.groupby("product_line")["profit_margin"].mean()
                st.write("**Average Profit Margin by Product Line:**")
                st.dataframe(avg_margin)

            elif question_choice == "What is the demand level distribution?":
                demand_counts = data["Demand_Level"].value_counts()
                st.write("**Demand Level Distribution:**")
                st.dataframe(demand_counts)
                fig = px.pie(demand_counts.reset_index(), values="Demand_Level", names="index", 
                             title="Demand Level Distribution",
                             color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig)

            elif question_choice == "How does stock availability affect win rate?":
                grouped = data.groupby(["Stock_Availability", "project_status"]).size().unstack(fill_value=0)
                grouped["win_rate"] = grouped.get("Won", 0) / grouped.sum(axis=1)
                st.write("**Win Rate by Stock Availability:**")
                st.dataframe(grouped)
                fig = px.bar(grouped.reset_index(), x="Stock_Availability", y="win_rate", 
                             title="Win Rate by Stock Availability",
                             color="Stock_Availability",
                             color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig)

            elif question_choice == "Is installation cost impacting the winning rate?":
                avg_install_won = data[data["project_status"] == "Won"]["Installation_Cost"].mean()
                avg_install_lost = data[data["project_status"] == "Lost"]["Installation_Cost"].mean()
                st.write(f"**Average Installation Cost (Won):** ${avg_install_won:.2f}")
                st.write(f"**Average Installation Cost (Lost):** ${avg_install_lost:.2f}")

            elif question_choice == "What is the pipeline status distribution?":
                pipeline_counts = data["project_status"].value_counts()
                st.write("**Pipeline Status Distribution:**")
                st.dataframe(pipeline_counts)
                df_pipeline = pipeline_counts.reset_index()
                df_pipeline.columns = ['project_status', 'count']
                fig = px.pie(df_pipeline, values="count", names="project_status", 
                             title="Pipeline Status Distribution",
                             color="project_status",
                             color_discrete_map={"Won": "green", "Lost": "red", "Pending": "orange"})
                st.plotly_chart(fig)

            elif question_choice == "How many quotes are active, lost, or won per month?":
                if "quote_date" in data.columns:
                    data["quote_date"] = pd.to_datetime(data["quote_date"])
                    monthly_status = data.groupby([data["quote_date"].dt.to_period("M"), "project_status"]).size().unstack(fill_value=0)
                    st.write("**Monthly Quotes Status:**")
                    st.dataframe(monthly_status)
                    fig = px.line(monthly_status, x=monthly_status.index.astype(str), y=monthly_status.columns,
                                  title="Quotes Status Over Time",
                                  markers=True)
                    st.plotly_chart(fig)
                else:
                    st.warning("Column 'quote_date' not found in data.")

            elif question_choice == "Which projects are pending and likely to convert?":
                pending_projects = data[data["project_status"].str.lower() == "pending"]
                if not pending_projects.empty:
                    st.write("**Pending Projects:**")
                    st.dataframe(pending_projects)
                else:
                    st.write("No pending projects found.")

            elif question_choice == "Which brands/vendors have the most successful quotes?":
                vendor_success = data[data["project_status"] == "Won"].groupby("brand").size().sort_values(ascending=False)
                st.write("**Brands by Number of Successful Quotes:**")
                st.dataframe(vendor_success)
                fig = px.bar(vendor_success.reset_index(), x="brand", y=0, 
                             title="Brands by Successful Quotes",
                             color="brand",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig)

            elif question_choice == "How often do different product versions sell?":
                product_version_counts = data["product_version"].value_counts()
                st.write("**Product Versions Sales Count:**")
                st.dataframe(product_version_counts)
                fig = px.bar(product_version_counts.reset_index(), x="product_version", y="count", 
                             title="Product Versions Sales",
                             color="product_version",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig)

            elif question_choice == "What is the average quantity sold per product/customer?":
                qty_by_product = data.groupby("product_line")["qty"].mean()
                qty_by_customer = data.groupby("Customer_Type")["qty"].mean()
                st.write("**Average Quantity Sold by Product Line:**")
                st.dataframe(qty_by_product)
                st.write("**Average Quantity Sold by Customer Type:**")
                st.dataframe(qty_by_customer)

def show_about():
    """About page"""
    st.markdown('<h1 class="main-header">About This Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="highlight">
        - Generate optimal prices using machine learning
        - Analyze historical data for insights
        - Get AI recommendations for pricing strategy

        **Key Features:**
        - Dynamic price prediction based on market conditions
        - Competitive pricing analysis
        - Business advisor with natural language interface
        - Comprehensive data visualization
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
            ***
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h3 style="color: #1f77b4;">Developed by Proseth Developer Team</h3>
                <p>
                    Visit our website: <a href="https://prosethsolutions.com/" target="_blank">https://prosethsolutions.com/</a>
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- Main App Structure ---
def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
 
    with st.sidebar:
        page_options = {
            "Pricing Tool": "💰",
            "Data Insights": "📊", 
            "About": "ℹ️"
        }
        
        page = st.radio("Go to", list(page_options.keys()), 
                        format_func=lambda x: f"{page_options[x]} {x}")
        
        st.markdown("***")
        st.markdown("### Quick Stats")
        if not data.empty:
            total_revenue = data["selling"].sum()
            win_rate = (data["project_status"] == "Won").mean() * 100
            avg_deal_size = data["selling"].mean()
            
            st.markdown(f"""
            - Total Revenue: **${total_revenue:,.2f}**
            - Win Rate: **{win_rate:.1f}%**
            - Avg Deal Size: **${avg_deal_size:,.2f}**
            """)
        else:
            st.info("Upload data to see statistics")
    
    # Main content area
    if page == "Pricing Tool":
        show_pricing_tool()
        
        # Chat interface (only show if we have context)
        if "context_data" in st.session_state:
            st.markdown("---")
            st.markdown('<div class="sub-header">💬 Pricing Advisor Chat</div>', unsafe_allow_html=True)
            
            # Display chat history
            for i, (speaker, msg) in enumerate(st.session_state.chat_history):
                message(msg, is_user=(speaker == "You"), key=f"chat_{i}")
            
            # User input
            user_question = st.chat_input("Ask the advisor about this pricing...")
            if user_question:
                with st.spinner("Thinking..."):
                    # Add your chat logic here
                    context_str = f"""
                    Base model price: ${st.session_state.context_data['base_price']:,.2f}
                    Competitor price: ${st.session_state.context_data['competitor_price']:,.2f}
                    Recommended selling price: ${st.session_state.context_data['competitive_price']:,.2f}
                    Our cost: ${st.session_state.context_data['costing']:,.2f}
                    Profit margin: {st.session_state.context_data['competitive_margin']:.1f}%
                    """
                    # For now, just echo the question
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Advisor", f"I received your question: '{user_question}'. I would analyze this based on the pricing context."))
                st.rerun()
                
    elif page == "Data Insights":
        show_data_insights()
    else:
        show_about()

if __name__ == "__main__":
    main()
