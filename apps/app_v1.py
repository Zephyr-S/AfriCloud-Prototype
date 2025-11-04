import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import contextlib
import textwrap
import re
import ast
import time
from typing import List, Dict, Any
import random

# Set page config with premium aesthetics
st.set_page_config(
    page_title="DataVision AI Analyst",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

class SmartAnalysisGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def detect_request_type(self, user_request: str) -> str:
        """Intelligently detect what type of analysis is requested"""
        request_lower = user_request.lower()
        
        # Check for specific analysis types
        if any(word in request_lower for word in ['correlation', 'relationship', 'scatter', 'correlate']):
            return 'correlation'
        elif any(word in request_lower for word in ['distribution', 'histogram', 'frequency', 'density']):
            return 'distribution'
        elif any(word in request_lower for word in ['category', 'count', 'bar chart', 'pie chart', 'categorical']):
            return 'categorical'
        elif any(word in request_lower for word in ['trend', 'time', 'date', 'over time', 'timeline']):
            return 'temporal'
        elif any(word in request_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return 'comparison'
        elif any(word in request_lower for word in ['outlier', 'anomaly', 'extreme']):
            return 'outlier'
        elif any(word in request_lower for word in ['graph', 'chart', 'visualize', 'plot', 'visualization']):
            return 'visualization'
        elif any(word in request_lower for word in ['overview', 'summary', 'comprehensive', 'everything']):
            return 'comprehensive'
        else:
            return 'smart'
    
    def extract_mentioned_columns(self, user_request: str) -> List[str]:
        """Extract column names mentioned in the user request"""
        mentioned_cols = []
        request_lower = user_request.lower()
        
        for col in self.df.columns:
            col_lower = col.lower()
            # Check if column name is mentioned in the request
            if col_lower in request_lower:
                mentioned_cols.append(col)
            # Also check for partial matches for common column names
            elif any(word in col_lower for word in ['age', 'salary', 'price', 'cost', 'value', 'amount']) and any(word in request_lower for word in ['age', 'salary', 'price', 'cost', 'value', 'amount']):
                mentioned_cols.append(col)
        
        return list(set(mentioned_cols))  # Remove duplicates
    
    def generate_smart_analysis(self, user_request: str) -> str:
        """Generate analysis code based on intelligent detection of user intent"""
        analysis_type = self.detect_request_type(user_request)
        mentioned_cols = self.extract_mentioned_columns(user_request)
        
        # Map analysis type to appropriate template
        templates = {
            'correlation': self._generate_correlation_analysis(mentioned_cols),
            'distribution': self._generate_distribution_analysis(mentioned_cols),
            'categorical': self._generate_categorical_analysis(mentioned_cols),
            'temporal': self._generate_temporal_analysis(mentioned_cols),
            'comparison': self._generate_comparison_analysis(mentioned_cols),
            'outlier': self._generate_outlier_analysis(mentioned_cols),
            'visualization': self._generate_visualization_analysis(mentioned_cols),
            'comprehensive': self._generate_comprehensive_analysis(),
            'smart': self._generate_smart_fallback_analysis(user_request, mentioned_cols)
        }
        
        return templates[analysis_type]
    
    def _generate_correlation_analysis(self, mentioned_cols: List[str]) -> str:
        if len(mentioned_cols) >= 2:
            cols = mentioned_cols[:2]
        elif len(self.numeric_cols) >= 2:
            cols = self.numeric_cols[:2]
        else:
            cols = self.df.columns[:2] if len(self.df.columns) >= 2 else self.df.columns
        
        return f"""
import matplotlib.pyplot as plt
import numpy as np

print("📈 Correlation Analysis")
print("=" * 30)

col1, col2 = '{cols[0]}', '{cols[1]}'
correlation = df[col1].corr(df[col2])
print(f"Correlation between '{{col1}}' and '{{col2}}': {{correlation:.3f}}")

plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(df[col1], df[col2], alpha=0.6, color='#667eea', s=50)
plt.title(f'Relationship between {{col1}} and {{col2}}')
plt.xlabel(col1)
plt.ylabel(col2)
plt.grid(True, alpha=0.3)

# Add trendline
z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
p = np.poly1d(z)
plt.plot(df[col1], p(df[col1]), "r--", alpha=0.8, linewidth=2)

# Correlation matrix if multiple numeric columns
plt.subplot(1, 2, 2)
if len(df.select_dtypes(include=[np.number]).columns) > 2:
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Matrix')
else:
    # Bar chart showing correlation strength
    correlations = []
    other_numeric = [col for col in self.numeric_cols if col != col1][:4]
    for other_col in other_numeric:
        corr_val = df[col1].corr(df[other_col])
        correlations.append((other_col, corr_val))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    plt.bar([x[0] for x in correlations], [x[1] for x in correlations], color='#764ba2')
    plt.title(f'Correlation with {{col1}}')
    plt.xticks(rotation=45)
    plt.ylabel('Correlation Coefficient')

plt.tight_layout()
plt.show()
"""
    
    def _generate_distribution_analysis(self, mentioned_cols: List[str]) -> str:
        if mentioned_cols:
            col = mentioned_cols[0]
        elif self.numeric_cols:
            col = self.numeric_cols[0]
        else:
            col = self.df.columns[0]
        
        return f"""
import matplotlib.pyplot as plt
import numpy as np

print("📊 Distribution Analysis for '{col}'")
print("=" * 40)

stats = df['{col}'].describe()
print(f"Mean: {{stats['mean']:.2f}}")
print(f"Median: {{df['{col}'].median():.2f}}")
print(f"Std Dev: {{stats['std']:.2f}}")
print(f"Range: {{stats['min']:.2f}} to {{stats['max']:.2f}}")
print(f"IQR: {{stats['25%']:.2f}} - {{stats['75%']:.2f}}")

plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(1, 3, 1)
plt.hist(df['{col}'].dropna(), bins=30, alpha=0.7, color='#667eea', edgecolor='black')
plt.title('Histogram')
plt.xlabel('{col}')
plt.ylabel('Frequency')

# Boxplot
plt.subplot(1, 3, 2)
plt.boxplot(df['{col}'].dropna())
plt.title('Boxplot')
plt.ylabel('{col}')

# Density plot
plt.subplot(1, 3, 3)
sns.kdeplot(df['{col}'].dropna(), fill=True, color='#764ba2')
plt.title('Density Plot')
plt.xlabel('{col}')

plt.tight_layout()
plt.show()
"""
    
    def _generate_categorical_analysis(self, mentioned_cols: List[str]) -> str:
        if mentioned_cols:
            col = mentioned_cols[0]
        elif self.categorical_cols:
            col = self.categorical_cols[0]
        else:
            # Try to find any column with reasonable cardinality
            for col in self.df.columns:
                if 2 <= len(self.df[col].unique()) <= 20:
                    break
            else:
                col = self.df.columns[0]
        
        return f"""
import matplotlib.pyplot as plt

print("📊 Categorical Analysis for '{col}'")
print("=" * 40)

value_counts = df['{col}'].value_counts()
print("Value Counts:")
print(value_counts)
print(f"\\nUnique values: {{len(value_counts)}}")
print(f"Most common: {{value_counts.index[0]}} ({{value_counts.values[0]}})")
print(f"Least common: {{value_counts.index[-1]}} ({{value_counts.values[-1]}})")

plt.figure(figsize=(15, 6))

# Bar chart
plt.subplot(1, 2, 1)
bars = plt.bar(range(len(value_counts)), value_counts.values, 
               color=['#667eea', '#764ba2', '#6b7a8f', '#f7cac9', '#92a8d1'])
plt.title('Value Counts')
plt.xlabel('{col}')
plt.ylabel('Count')
plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)

# Add value labels
for i, count in enumerate(value_counts.values):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
        colors=['#667eea', '#764ba2', '#6b7a8f', '#f7cac9', '#92a8d1'])
plt.title('Percentage Distribution')

plt.tight_layout()
plt.show()
"""
    
    def _generate_visualization_analysis(self, mentioned_cols: List[str]) -> str:
        return """
import matplotlib.pyplot as plt
import numpy as np

print("📈 Automatic Visualization Dashboard")
print("=" * 40)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Available numeric columns: {numeric_cols}")
print(f"Available categorical columns: {categorical_cols}")

# Create a comprehensive visualization dashboard
fig = plt.figure(figsize=(18, 12))

# 1. Numeric distributions (if available)
if numeric_cols:
    for i, col in enumerate(numeric_cols[:2], 1):
        plt.subplot(2, 3, i)
        plt.hist(df[col].dropna(), bins=20, alpha=0.7, color='#667eea')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

# 2. Categorical analysis (if available)
if categorical_cols:
    for i, col in enumerate(categorical_cols[:2], 3):
        if i > 6:
            break
        plt.subplot(2, 3, i)
        value_counts = df[col].value_counts().head(6)
        bars = plt.bar(range(len(value_counts)), value_counts.values, 
                      color=['#667eea', '#764ba2', '#6b7a8f', '#f7cac9', '#92a8d1'])
        plt.title(f'Top values in {col}')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        plt.ylabel('Count')
        
        # Add value labels
        for j, count in enumerate(value_counts.values):
            plt.text(j, count + 0.1, str(count), ha='center', va='bottom')

# 3. Correlation heatmap (if multiple numeric columns)
if len(numeric_cols) > 2:
    plt.subplot(2, 3, 5)
    corr_matrix = df[numeric_cols].corr()
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Correlation Matrix')

# 4. Scatter plot of first two numeric columns
if len(numeric_cols) >= 2:
    plt.subplot(2, 3, 6)
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, color='#667eea')
    plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])

plt.tight_layout()
plt.show()
"""
    
    def _generate_comprehensive_analysis(self) -> str:
        return f"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("🔍 Comprehensive Data Analysis Report")
print("=" * 50)
print(f"Dataset Shape: {{df.shape}}")
print(f"Total Records: {{len(df)}}")
print(f"Columns: {{len(df.columns)}}")

print(f"\\n📋 Column Overview:")
for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"  {{col}}: {{df[col].dtype}} ({{null_count}} missing, {{null_pct:.1f}}%)")

print(f"\\n📊 Summary Statistics:")
print(df.describe())

# Create visualizations
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(15, 5 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        # Histogram
        axes[i, 0].hist(df[col].dropna(), bins=30, alpha=0.7, color='#667eea')
        axes[i, 0].set_title(f'Distribution of {{col}}')
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel('Frequency')
        
        # Boxplot
        axes[i, 1].boxplot(df[col].dropna())
        axes[i, 1].set_title(f'Boxplot of {{col}}')
        axes[i, 1].set_ylabel(col)
    
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found for detailed visualization")
"""
    
    def _generate_smart_fallback_analysis(self, user_request: str, mentioned_cols: List[str]) -> str:
        """Generate analysis for unspecified requests"""
        # Try to understand the intent and provide relevant analysis
        if any(word in user_request.lower() for word in ['show', 'display', 'tell', 'what']):
            return self._generate_comprehensive_analysis()
        elif any(word in user_request.lower() for word in ['find', 'discover', 'reveal', 'insight']):
            return self._generate_visualization_analysis(mentioned_cols)
        else:
            return self._generate_comprehensive_analysis()
    
    def _generate_temporal_analysis(self, mentioned_cols: List[str]) -> str:
        # Simplified temporal analysis
        return self._generate_comprehensive_analysis()
    
    def _generate_comparison_analysis(self, mentioned_cols: List[str]) -> str:
        return self._generate_correlation_analysis(mentioned_cols)
    
    def _generate_outlier_analysis(self, mentioned_cols: List[str]) -> str:
        return self._generate_distribution_analysis(mentioned_cols)

# Sidebar
with st.sidebar:
    st.markdown("### 🔮 DataVision AI")
    st.markdown("---")
    st.markdown("### ⚡ Quick Analysis")
    
    quick_buttons = st.columns(2)
    with quick_buttons[0]:
        if st.button("📊 Overview", use_container_width=True):
            st.session_state.quick_analysis = "overview"
    with quick_buttons[1]:
        if st.button("📈 Visualize", use_container_width=True):
            st.session_state.quick_analysis = "visualization"

# Main content
st.title("📊 DataVision AI Analyst")
st.markdown("Intelligent analysis for any data request")

# File upload
uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Initialize smart analyzer
    analyzer = SmartAnalysisGenerator(df)
    
    # Data overview
    with st.expander("📋 Dataset Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Rows", df.shape[0])
        with col2: st.metric("Columns", df.shape[1])
        with col3: st.metric("Complete", f"{(1 - df.isnull().mean().mean()) * 100:.1f}%")
        
        st.write("**Columns:**", list(df.columns))
    
    # Analysis request
    user_question = st.text_area(
        "💬 Ask anything about your data:",
        height=100,
        placeholder="Examples: 'Show me sales trends', 'Find correlations', 'Analyze customer demographics', 'Create visualizations', 'Tell me about this data'"
    )
    
    # Quick analysis from sidebar
    if 'quick_analysis' in st.session_state:
        if st.session_state.quick_analysis == "overview":
            user_question = "comprehensive overview"
        elif st.session_state.quick_analysis == "visualization":
            user_question = "create visualizations and charts"
        del st.session_state.quick_analysis
    
    if st.button("🚀 Generate Analysis", type="primary") and user_question:
        with st.spinner("🤔 Analyzing your request..."):
            start_time = time.time()
            
            # Generate smart analysis based on request
            ai_code = analyzer.generate_smart_analysis(user_question)
            generation_time = time.time() - start_time
            
            st.success(f"✅ Analysis ready in {generation_time:.2f}s!")
            
            # Execute code
            try:
                plt.close('all')
                output_buffer = io.StringIO()
                
                with contextlib.redirect_stdout(output_buffer):
                    with contextlib.redirect_stderr(io.StringIO()):
                        exec_globals = {
                            'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'np': np,
                            'print': print, 'len': len, 'str': str, 'list': list, 'dict': dict
                        }
                        exec(ai_code, exec_globals)
                
                output_text = output_buffer.getvalue()
                
                # Display insights
                if output_text.strip():
                    st.markdown("#### 📝 Analysis Insights")
                    st.text_area("", output_text, height=200, label_visibility="collapsed")
                
                # Display visualizations
                if plt.get_fignums():
                    st.markdown("#### 📈 Visualizations")
                    for fig_num in plt.get_fignums():
                        plt.figure(fig_num)
                        st.pyplot(plt)
                
            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
            
            # Show code
            with st.expander("🔧 View Analysis Code"):
                st.code(ai_code, language='python')

else:
    # Welcome state
    st.markdown("<div style='text-align: center; padding: 40px 0;'>", unsafe_allow_html=True)
    st.header("🎯 Welcome to DataVision AI!")
    st.subheader("Ask anything about your data - get instant insights")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature examples
    examples = st.columns(2)
    with examples[0]:
        st.info("""
        **💬 Example Requests:**
        - "Show sales trends"
        - "Find customer patterns"  
        - "Analyze product performance"
        - "Create correlation matrix"
        """)
    with examples[1]:
        st.success("""
        **🎯 Smart Features:**
        - Understands natural language
        - Automatic visualization
        - Statistical analysis
        - Instant results
        """)

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ • Smart Analysis Engine • ⚡ Instant Results**")
