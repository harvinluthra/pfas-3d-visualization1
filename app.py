import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="PFAS 3D Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #0066CC;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #0052A3;
        color: white;
    }
    .stDownloadButton > button {
        background-color: #28A745;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #218838;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def validate_data(df):
    """
    Validate the uploaded data to ensure it contains required columns
    and proper data types.
    """
    required_columns = ['PFAS_Class', 'PFAS_Compd', 'Carbon_Number', 'Concentration']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if PFAS_Class contains text data
    if not df['PFAS_Class'].dtype == 'object':
        return False, "PFAS_Class column must contain text/categorical data"
    
    # Check if PFAS_Compd contains text data
    if not df['PFAS_Compd'].dtype == 'object':
        return False, "PFAS_Compd column must contain text/categorical data"
    
    # Check if Carbon_Number contains numeric data
    try:
        pd.to_numeric(df['Carbon_Number'], errors='raise')
    except (ValueError, TypeError):
        return False, "Carbon_Number column must contain numeric data"
    
    # Check if Concentration contains numeric data
    try:
        concentrations = pd.to_numeric(df['Concentration'], errors='raise')
        if (concentrations < 0).any():
            return False, "Concentration must contain non-negative values (zero or positive)"
    except (ValueError, TypeError):
        return False, "Concentration column must contain numeric data"
    
    # Check for empty data
    if df.empty:
        return False, "The uploaded file contains no data"
    
    return True, "Data validation successful"

def clean_and_prepare_data(df):
    """
    Clean and prepare the data for visualization.
    """
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Remove rows with missing values
    df_clean = df_clean.dropna(subset=['PFAS_Class', 'PFAS_Compd', 'Carbon_Number', 'Concentration'])
    
    # Convert data types
    df_clean['Carbon_Number'] = pd.to_numeric(df_clean['Carbon_Number'], errors='coerce')
    df_clean['Concentration'] = pd.to_numeric(df_clean['Concentration'], errors='coerce')
    
    # Remove any rows that couldn't be converted to numeric
    df_clean = df_clean.dropna()
    
    # Ensure concentrations are non-negative (allow zero values)
    df_clean = df_clean[df_clean['Concentration'] >= 0]
    
    return df_clean

def create_3d_bar_chart(df):
    """
    Create an interactive 3D bar chart using Plotly with gradient coloring based on concentration.
    """
    fig = go.Figure()
    
    # Define the specific order for PFAS classes (reversed for right to left display)
    pfas_class_order = ['PFEAs3', 'PFEAs2', 'PFEAs1', 'FTSs', 'FTCAs', 'PFCAs', 'PFSAs', 'FASAs', 'FASAAs', 'FASEs']
    
    # Get unique classes from data and arrange them in the specified order
    data_classes = df['PFAS_Class'].unique()
    
    # Create ordered list: first the classes that appear in our data in the specified order,
    # then any additional classes not in the predefined order
    ordered_classes = []
    for cls in pfas_class_order:
        if cls in data_classes:
            ordered_classes.append(cls)
    
    # Add any classes from data that weren't in the predefined order
    for cls in data_classes:
        if cls not in ordered_classes:
            ordered_classes.append(cls)
    
    unique_carbons = sorted(df['Carbon_Number'].unique())
    
    # Create a mapping for x and y positions
    class_positions = {cls: i for i, cls in enumerate(ordered_classes)}
    carbon_positions = {carbon: i for i, carbon in enumerate(unique_carbons)}
    
    # Prepare data for 3D bars
    x_pos = []
    y_pos = []
    z_pos = []
    colors = []
    hover_text = []
    
    for idx, row in df.iterrows():
        x = class_positions[row['PFAS_Class']]
        y = carbon_positions[row['Carbon_Number']]
        z = row['Concentration']
        
        x_pos.append(x)
        y_pos.append(y)
        z_pos.append(z)
        colors.append(z)
        hover_text.append(f"Class: {row['PFAS_Class']}<br>Compound: {row['PFAS_Compd']}<br>Carbon #: {row['Carbon_Number']}<br>Concentration: {z:.2e}")
    
    # Create 3D bars using mesh3d for better bar appearance
    for i, (x, y, z, color, text) in enumerate(zip(x_pos, y_pos, z_pos, colors, hover_text)):
        # Create a rectangular bar from (x,y,0) to (x+0.8, y+0.8, z)
        vertices = [
            [x-0.4, y-0.4, 0], [x+0.4, y-0.4, 0], [x+0.4, y+0.4, 0], [x-0.4, y+0.4, 0],  # bottom face
            [x-0.4, y-0.4, z], [x+0.4, y-0.4, z], [x+0.4, y+0.4, z], [x-0.4, y+0.4, z]   # top face
        ]
        
        # Define faces of the rectangular bar
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ]
        
        # Extract vertices for plotly format
        x_vertices = [v[0] for v in vertices]
        y_vertices = [v[1] for v in vertices]
        z_vertices = [v[2] for v in vertices]
        
        # Extract face indices
        i_faces = [f[0] for f in faces]
        j_faces = [f[1] for f in faces]
        k_faces = [f[2] for f in faces]
        
        fig.add_trace(go.Mesh3d(
            x=x_vertices,
            y=y_vertices,
            z=z_vertices,
            i=i_faces,
            j=j_faces,
            k=k_faces,
            intensity=[color] * len(vertices),
            colorscale=[[0, 'white'], [0.001, 'lightblue'], [0.2, 'blue'], [0.4, 'green'], [0.6, 'yellow'], [0.8, 'orange'], [1.0, 'red']],  # Custom colorscale with white for zero
            cmin=0,  # Set color scale minimum to 0
            cmax=df['Concentration'].max(),  # Set color scale maximum to data max
            showscale=True if i == 0 else False,  # Show colorbar only for first trace
            colorbar=dict(
                title=dict(
                    text="Concentration",
                    side="right",
                    font=dict(size=16, family='Arial, sans-serif', color='#2E3440')
                ),
                tickmode="array",
                tickvals=[0, df['Concentration'].max()/4, df['Concentration'].max()/2, 3*df['Concentration'].max()/4, df['Concentration'].max()],
                ticktext=[f"0", f"{df['Concentration'].max()/4:.1e}", f"{df['Concentration'].max()/2:.1e}", f"{3*df['Concentration'].max()/4:.1e}", f"{df['Concentration'].max():.1e}"],
                tickfont=dict(size=14, family='Arial, sans-serif', color='#2E3440'),
                len=0.8,  # Longer colorbar
                thickness=25,  # Thicker colorbar
                x=1.02,  # Better positioning
                bordercolor='#6C757D',
                borderwidth=1,
                outlinecolor='#6C757D',
                outlinewidth=1
            ) if i == 0 else None,
            hovertemplate=f"<b>{text}</b><extra></extra>",
            name=f"Bar {i+1}",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive 3D PFAS Bar Chart',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': '#2E3440'}
        },
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='PFAS Class',
                    font=dict(size=16, family='Arial, sans-serif', color='#2E3440')
                ),
                tickmode='array',
                tickvals=list(range(len(ordered_classes))),
                ticktext=ordered_classes,
                tickangle=45,
                tickfont=dict(size=12, family='Arial, sans-serif', color='#2E3440'),
                backgroundcolor='#F8F9FA',
                gridcolor='#E9ECEF',
                linecolor='#6C757D',
                range=[-0.8, len(ordered_classes) - 0.2]  # Expand X-axis range
            ),
            yaxis=dict(
                title=dict(
                    text='Carbon Number',
                    font=dict(size=16, family='Arial, sans-serif', color='#2E3440')
                ),
                tickmode='array',
                tickvals=list(range(len(unique_carbons))),
                ticktext=[str(c) for c in unique_carbons],
                tickfont=dict(size=12, family='Arial, sans-serif', color='#2E3440'),
                backgroundcolor='#F8F9FA',
                gridcolor='#E9ECEF',
                linecolor='#6C757D'
            ),
            zaxis=dict(
                visible=False  # Hide the Z-axis
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2)  # Better viewing angle
            ),
            bgcolor='#FFFFFF'  # Clean white background
        ),
        width=1200,  # Increased width
        height=900,  # Increased height
        margin=dict(l=20, r=120, t=80, b=20),  # Better margins for colorbar
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family='Arial, sans-serif', size=12, color='#2E3440')
    )
    
    return fig

def generate_sample_template():
    """
    Generate a sample CSV template for users to understand the required format.
    """
    sample_data = {
        'PFAS_Class': ['PFCAs', 'PFSAs', 'FASEs', 'PFCAs', 'PFSAs', 'FTSs', 'PFEAs1', 'PFEAs2'],
        'PFAS_Compd': ['C8-PFOA', 'C8-PFOS', 'C9-FASE', 'C8-PFOA-2', 'C8-PFOS-2', 'C6-FTS', 'C10-PFEA1', 'C10-PFEA2'],
        'Carbon_Number': [8, 8, 9, 8, 8, 6, 10, 10],
        'Concentration': [1.5e-6, 2.3e-5, 0.0, 3.2e-6, 1.8e-5, 5.4e-7, 2.1e-6, 0.0]
    }
    return pd.DataFrame(sample_data)

def main():
    """
    Main application function.
    """
    # Title and description
    st.title("📊 PFAS 3D Bar Chart")
    st.markdown("""
    This application creates interactive 3D bar charts to visualize PFAS (Per- and polyfluoroalkyl substances) data.
    Upload your CSV file containing PFAS Class, PFAS Compound, Carbon Number, and Concentration data.
    Each bar height represents concentration values with gradient coloring.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with columns: PFAS_Class, PFAS_Compd, Carbon_Number, Concentration"
        )
        
        st.markdown("---")
        
        # Download sample template
        st.header("📝 Sample Template")
        st.markdown("Download a sample CSV template to understand the required format:")
        
        sample_df = generate_sample_template()
        csv_template = sample_df.to_csv(index=False)
        
        st.download_button(
            label="Download Sample Template",
            data=csv_template,
            file_name="pfas_sample_template.csv",
            mime="text/csv"
        )
        
        # Display sample data
        st.subheader("Sample Data Format:")
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("---")
        
        # Data requirements
        st.header("📋 Data Requirements")
        st.markdown("""
        **Required Columns:**
        - `PFAS_Class`: Text categories (e.g., PFOA, PFOS, PFNA)
        - `PFAS_Compd`: Specific compound names (e.g., C8-PFOA, C8-PFOS)
        - `Carbon_Number`: Numeric values (carbon chain length)
        - `Concentration`: Positive numeric values (raw concentration data)
        
        **Important Notes:**
        - Concentration values are plotted as-is without any transformation
        - Concentration values must be non-negative (zero or positive values allowed)
        - Missing values will be automatically removed
        - File must be in CSV format
        """)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data info
            st.success(f"✅ File uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            # Validate data
            is_valid, validation_message = validate_data(df)
            
            if not is_valid:
                st.error(f"❌ Data validation failed: {validation_message}")
                st.info("Please check your data format and try again. Use the sample template as a reference.")
                return
            
            st.success(f"✅ {validation_message}")
            
            # Clean and prepare data
            df_clean = clean_and_prepare_data(df)
            
            if df_clean.empty:
                st.error("❌ No valid data remains after cleaning. Please check your data quality.")
                return
            
            # Create and display the 3D visualization
            try:
                fig = create_3d_bar_chart(df_clean)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export functionality
                st.subheader("💾 Export Interactive Chart")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export cleaned data
                    cleaned_csv = df_clean.to_csv(index=False)
                    st.download_button(
                        label="Download Data (CSV)",
                        data=cleaned_csv,
                        file_name="pfas_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export interactive plot as HTML
                    html_string = fig.to_html(
                        include_plotlyjs='cdn',
                        config={'displayModeBar': True, 'responsive': True}
                    )
                    st.download_button(
                        label="Download Interactive Chart (HTML)",
                        data=html_string,
                        file_name="pfas_3d_chart.html",
                        mime="text/html"
                    )
                
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
                logger.error(f"Visualization error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format with the required columns.")
            logger.error(f"File processing error: {str(e)}")
    
    else:
        # Display instructions when no file is uploaded
        st.info("👆 Please upload a CSV file using the sidebar to get started.")
        
        # Display additional information
        st.header("🔍 About This Tool")
        st.markdown("""
        This 3D bar chart tool visualizes PFAS (Per- and polyfluoroalkyl substances) data with the following features:
        
        **📈 3D Bar Chart Features:**
        - **X-axis:** PFAS Class categories
        - **Y-axis:** Carbon numbers  
        - **Z-axis:** Concentration values (bar height)
        - **Color gradient:** Bar color intensity based on concentration
        - **Interactive controls:** Rotate, zoom, and hover for detailed information
        
        **🛠️ Technical Features:**
        - Data validation and cleaning
        - Error handling for invalid data
        - Export functionality for both data and visualizations
        - Responsive design for different screen sizes
        - Detailed data insights and statistics
        
        **📋 Getting Started:**
        1. Download the sample template from the sidebar
        2. Prepare your data in the required format
        3. Upload your CSV file
        4. Explore your interactive 3D visualization!
        """)

if __name__ == "__main__":
    main()
