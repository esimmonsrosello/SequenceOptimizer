"""
HOOF - DNA Codon Optimization Tool
Last updated: 20251022

This tool helps optimize DNA sequences to reduce ribosomal frameshifting while maintaining
proper codon usage. It handles both single sequences and batch processing with multiple
optimization strategies.
"""

# Core libraries for the web interface
import streamlit as st
import streamlit.components.v1 as components

# Data handling and numerical operations
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# System and utility imports
import os
import logging
from collections import defaultdict, Counter
import io
import requests
import time
import re
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime

# BioPython for sequence manipulation
from Bio.Seq import Seq
from Bio.Data import CodonTable


# Set up the Streamlit page - this has to be the first Streamlit command
st.set_page_config(
    page_title="HOOF",
    page_icon=":horse:",  # Shows a horse emoji in the browser tab
    layout="wide",  # Use full width of the browser
    initial_sidebar_state="expanded"  # Start with sidebar open
)

# Application-wide constants
BIAS_WEIGHT_DEFAULT = 0.5  # Default weight for balancing codon bias vs other factors
FRAME_OFFSET = 1  # Offset for reading frame analysis
VALID_DNA_BASES = 'ATGC'  # Valid nucleotides (no U since we're working with DNA)
CONFIG_FILE = "codon_optimizer_config.json"
DEFAULT_CONFIG = {
    "codon_file_path": "HumanCodons.xlsx",  # Where we load codon usage data from
    "bias_weight": BIAS_WEIGHT_DEFAULT,
    "auto_open_files": True,
    "default_output_dir": "."
}

# Theme definitions for data visualization
# Each theme provides different color palettes for various chart types
# This allows users to customize the look or choose colorblind-friendly options
THEMES = {
    "Default": {
        "info": "Default color scheme with vibrant, high-contrast colors.",
        "colors": {
            "utr5": "#1900FF",  # 5' UTR regions
            "cds": "#4ECDC4",  # Coding sequence
            "utr3": "#FF6B6B",  # 3' UTR regions
            "signal_peptide": "#8A2BE2",  # Signal peptide sequences
            "optimization": {'original': '#FF8A80', 'optimized': '#4ECDC4'},  # Before/after colors
            "analysis": ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'],
            "gradient": ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2']
        }
    },
    "Oceanic": {
        "info": "A cool-toned theme inspired by the ocean.",
        "colors": {
            "utr5": "#006994",
            "cds": "#00A5AD",
            "utr3": "#88D8B0",
            "signal_peptide": "#58A4B0",
            "optimization": {'original': '#F9A825', 'optimized': '#00A5AD'},
            "analysis": ['#00A5AD', '#58A4B0', '#88D8B0', '#B3E5FC', '#4DD0E1', '#26C6DA', '#00BCD4', '#00ACC1'],
            "gradient": ['#E0F7FA', '#B2EBF2', '#80DEEA', '#4DD0E1', '#26C6DA', '#00BCD4', '#00ACC1', '#0097A7']
        }
    },
    "Sunset": {
        "info": "A warm-toned theme reminiscent of a sunset.",
        "colors": {
            "utr5": "#D9534F",
            "cds": "#F0AD4E",
            "utr3": "#5CB85C",
            "signal_peptide": "#E57373",
            "optimization": {'original': '#D9534F', 'optimized': '#F0AD4E'},
            "analysis": ['#F0AD4E', '#E57373', '#FF8A65', '#FFB74D', '#FFD54F', '#FFF176', '#DCE775', '#AED581'],
            "gradient": ['#FFF3E0', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726', '#FF9800', '#FB8C00', '#F57C00']
        }
    },
    # --- COLOR-BLIND FRIENDLY THEMES ---
    # These themes are designed to be distinguishable by people with color vision deficiencies
    "Colorblind Safe": {
        "info": "High contrast colors optimized for colorblind users (deuteranopia/protanopia safe).",
        "colors": {
            "utr5": "#000000",      # Black - always distinguishable
            "cds": "#E69F00",       # Orange - safe choice
            "utr3": "#56B4E9",      # Sky Blue - works for most types of colorblindness
            "signal_peptide": "#009E73",  # Bluish Green
            "optimization": {'original': '#CC79A7', 'optimized': '#E69F00'},  # Pink to Orange
            "analysis": ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'],
            "gradient": ['#FFF2CC', '#FFE699', '#FFD966', '#FFCC33', '#E69F00', '#CC8F00', '#B37F00', '#996F00']
        }
    },
    "High Contrast": {
        "info": "Maximum contrast theme for accessibility.",
        "colors": {
            "utr5": "#000000",      # Pure black
            "cds": "#FFFFFF",       # Pure white
            "utr3": "#FF0000",      # Pure red
            "signal_peptide": "#00FF00",  # Pure green
            "optimization": {'original': '#FF0000', 'optimized': '#00FF00'},
            "analysis": ['#000000', '#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
            "gradient": ['#CCCCCC', '#AAAAAA', '#888888', '#666666', '#444444', '#222222', '#111111', '#000000']
        }
    },
    "Viridis": {
        "info": "Perceptually uniform colormap, excellent for colorblind users.",
        "colors": {
            "utr5": "#440154",      # Dark purple
            "cds": "#31688E",       # Blue
            "utr3": "#35B779",      # Green
            "signal_peptide": "#FDE725",  # Yellow
            "optimization": {'original': '#440154', 'optimized': '#35B779'},
            "analysis": ['#440154', '#482777', '#3F4A8A', '#31688E', '#26838F', '#1F9D8A', '#6CCE5A', '#B6DE2B'],
            "gradient": ['#440154', '#482777', '#3F4A8A', '#31688E', '#26838F', '#1F9D8A', '#6CCE5A', '#B6DE2B']
        }
    }
}
# CSS styling for the Streamlit app itself (not the charts)
# These styles change the background colors and text colors of the interface
APP_THEMES_CSS = {
    "Default": "",  # No custom CSS - uses Streamlit's default styling
    "Oceanic": """
        <style>
            /* Light blue background for main app area */
            [data-testid="stAppViewContainer"] {
                background-color: #F0F8FF;
            }
            /* Slightly darker blue for sidebar */
            [data-testid="stSidebar"] {
                background-color: #E0F7FA;
            }
            /* Dark teal text for better contrast */
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #004D40;
            }
        </style>
    """,
    "Sunset": """
        <style>
            /* Warm cream background */
            [data-testid="stAppViewContainer"] {
                background-color: #FFF3E0;
            }
            /* Light orange sidebar */
            [data-testid="stSidebar"] {
                background-color: #FFE0B2;
            }
            /* Dark brown text for warmth */
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #5D4037;
            }
        </style>
    """
}

def inject_app_theme():
    """Applies the selected theme's CSS to the Streamlit interface"""
    theme_css = APP_THEMES_CSS.get(st.session_state.active_theme, "")
    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)


# Initialize Streamlit session state variables
# Session state persists data across reruns of the app (when user interacts with widgets)
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()
if 'active_theme' not in st.session_state:
    st.session_state.active_theme = "Default"
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = []  # Stores results from multiple single-sequence runs
if 'batch_accumulated_results' not in st.session_state:
    st.session_state.batch_accumulated_results = []  # Stores results from multiple batch runs
if 'run_counter' not in st.session_state:
    st.session_state.run_counter = 0  # Tracks how many optimizations we've done
if 'genetic_code' not in st.session_state:
    st.session_state.genetic_code = {}  # Codon to amino acid mapping
if 'codon_weights' not in st.session_state:
    st.session_state.codon_weights = {}  # Usage frequencies for each codon
if 'preferred_codons' not in st.session_state:
    st.session_state.preferred_codons = {}  # Most common codon for each amino acid
if 'human_codon_usage' not in st.session_state:
    st.session_state.human_codon_usage = {}  # Human-specific codon usage data
if 'aa_to_codons' not in st.session_state:
    st.session_state.aa_to_codons = defaultdict(list)  # Maps amino acid to its possible codons


# Load any environment variables from .env file (if it exists)
load_dotenv()

# Configure logging to track what the app is doing
# Logs go both to a file and to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codon_optimizer.log'),  # Save to file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Biological constants for frame analysis
Slippery_Motifs = {"TTTT", "TTTC"}  # Sequences that can cause ribosomal slipping
PLUS1_STOP_CODONS = {"TAA", "TAG"}  # Stop codons in the +1 reading frame
PLUS1_STOP_MOTIFS = {"TAATAA", "TAGTAG", "TAGTAA", "TAATAG"}  # Double stops in +1 frame

# The standard genetic code - maps each 3-letter codon to its amino acid
# '*' represents stop codons
STANDARD_GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S',
    'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}

# Create a dictionary mapping amino acids to their synonymous codons
# This is used throughout the optimization algorithms
synonymous_codons = defaultdict(list)
for codon_val, aa_val in STANDARD_GENETIC_CODE.items(): 
    synonymous_codons[aa_val].append(codon_val)

# Alternative name for the same thing (kept for compatibility)
NC_synonymous_codons = defaultdict(list)
for codon, aa in STANDARD_GENETIC_CODE.items():
    NC_synonymous_codons[aa].append(codon)
    
# Amino acids that commonly appear at slippery sites
# These are used in frame-shift detection
FIRST_AA_CANDIDATES = ['L', 'I', 'V']
SECOND_AA_CANDIDATES = ['V', 'I']

# --- UTILITY FUNCTIONS ---

def calculate_gc_window(sequence, position, window_size=25):
    """
    Calculate GC content for a sliding window around a position.
    
    This helps us see if there are GC-rich or GC-poor regions in the sequence.
    High GC content can affect RNA stability and translation.
    
    Args:
        sequence: DNA sequence string
        position: Amino acid position (1-based indexing)
        window_size: Size of the window in base pairs (default 25bp)
    
    Returns:
        GC percentage as a float (0-100)
    """
    # Convert position from 1-based to 0-based indexing
    center_pos = (position - 1) * 3  # Convert amino acid position to nucleotide position
    
    # Figure out the window boundaries, making sure we don't go past the sequence ends
    start = max(0, center_pos - window_size // 2)
    end = min(len(sequence), center_pos + window_size // 2 + 1)
    
    # Grab the sequence in this window
    window_seq = sequence[start:end]
    
    if len(window_seq) == 0:
        return 0.0
    
    # Count G's and C's, calculate percentage
    gc_count = sum(1 for base in window_seq.upper() if base in 'GC')
    return (gc_count / len(window_seq)) * 100

@st.cache_data
def load_immunogenic_peptides(file_path="epitope_table_export.xlsx"):
    """
    Load immunogenic peptides from an Excel file.
    
    This data is used to check if our optimized sequences might contain peptides
    that could trigger immune responses. Cached so we only load it once.
    
    Args:
        file_path: Path to the Excel file with epitope data
    
    Returns:
        Cleaned DataFrame with peptide sequences, or empty DataFrame if file not found
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            
            # Clean up column names - Excel files can be messy
            df.columns = df.columns.str.strip()
            
            # Handle duplicate column names by numbering them
            # (sometimes Excel exports create duplicate headers)
            seen_columns = {}
            new_columns = []
            for col in df.columns:
                if col in seen_columns:
                    seen_columns[col] += 1
                    new_columns.append(f"{col}_{seen_columns[col]}")
                else:
                    seen_columns[col] = 0
                    new_columns.append(col)
            
            df.columns = new_columns
            
            # Try to find the column that contains the actual peptide sequences
            # Different files might call it different things
            name_column = None
            possible_name_columns = ['Name', 'Name_1', 'Name_2', 'Name_3']
            
            for col in possible_name_columns:
                if col in df.columns:
                    name_column = col
                    break
            
            # If we still haven't found it, just use the 3rd column (common format)
            if name_column is None and len(df.columns) >= 3:
                name_column = df.columns[2]
                
            if name_column is None:
                st.error(f"Could not find Name column. Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            # Clean up the peptide data
            df_clean = df.dropna(subset=[name_column])
            df_clean = df_clean[df_clean[name_column].notna()]
            df_clean[name_column] = df_clean[name_column].astype(str).str.upper().str.strip()
            
            # Filter out junk - sequences that are too short or invalid
            df_clean = df_clean[df_clean[name_column].str.len() >= 3]
            df_clean = df_clean[df_clean[name_column] != 'NAN']
            df_clean = df_clean[df_clean[name_column] != '']
            
            # Remember which column has the sequences for later
            df_clean.attrs['epitope_column'] = name_column
            
            return df_clean
        else:
            st.warning(f"Epitope file {file_path} not found. Immunogenic peptide scanning disabled.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading epitope file {file_path}: {str(e)}")
        st.write(f"**Debug - Exception details:** {e}")
        return pd.DataFrame()

def get_consistent_color_palette(n_colors, palette_type="optimization"):
    """
    Generate color palettes that match the currently selected theme.
    Keeps charts visually consistent with the selected theme.
    
    Args:
        n_colors: How many colors we need
        palette_type: Type of palette - "optimization", "analysis", or "gradient"
    
    Returns:
        List of color hex codes or dict for optimization type
    """
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    
    if palette_type == "optimization":
        return theme_colors["optimization"]
    elif palette_type == "analysis":
        # Cycle through the analysis colors if we need more than available
        base_colors = theme_colors["analysis"]
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    elif palette_type == "gradient":
        return theme_colors["gradient"]

def display_copyable_sequence(sequence, label, key_suffix=""):
    """
    Show a DNA/RNA sequence in a text box that's easy to copy.
    Users can select all with Ctrl+A and copy with Ctrl+C.
    
    Args:
        sequence: The DNA/RNA sequence to display
        label: Text to show above the box
        key_suffix: Unique identifier for this text area (needed by Streamlit)
    """
    st.text_area(
        label,
        sequence,
        height=120,
        key=f"copy_{key_suffix}",
        help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
    )



def find_coding_sequence_bounds(dna_seq):
    """
    Find where the actual coding sequence starts and ends in a DNA sequence.
    
    We prioritize finding the Kozak consensus sequence (ACCATG) because that's usually
    where translation really starts in eukaryotes. If we can't find that, we fall back
    to just finding the first ATG.
    
    Args:
        dna_seq: The full DNA sequence (may include UTRs)
    
    Returns:
        tuple: (start_position, end_position) of the CDS, or (None, None) if not found
    """
    dna_seq_upper = dna_seq.upper().replace('U', 'T')  # Handle RNA input too
    stop_codons = {"TAA", "TAG", "TGA"}
    
    start_pos = None
    
    # Look for the Kozak sequence first - ACCATG
    # This is the most common translation initiation context in mammals
    accatg_pos = dna_seq_upper.find('ACCATG')
    if accatg_pos != -1:
        # ATG starts 3 bases into ACCATG
        start_pos = accatg_pos + 3
    else:
        # No Kozak? Just find the first ATG then
        atg_pos = dna_seq_upper.find('ATG')
        if atg_pos != -1:
            start_pos = atg_pos
            
    if start_pos is None:
        # No start codon at all - can't do anything
        return None, None
    
    # Now find the end - first in-frame stop codon after our start
    end_pos = None
    for i in range(start_pos, len(dna_seq_upper) - 2, 3):  # Step by 3 to stay in frame
        codon = dna_seq_upper[i:i+3]
        if len(codon) == 3 and codon in stop_codons:
            end_pos = i
            break
            
    return start_pos, end_pos


def create_interactive_cai_gc_plot(positions, cai_weights, amino_acids, sequence, seq_name, color='#4ECDC4'):
    """
    Create an interactive plot showing CAI weights and GC content together.
    
    This helps visualize codon quality (CAI) alongside GC content patterns.
    The two metrics are plotted on separate Y-axes since they have different scales.
    
    Args:
        positions: List of amino acid positions
        cai_weights: CAI weight at each position (0-1 scale)
        amino_acids: The actual amino acid at each position
        sequence: Full DNA sequence
        seq_name: Name for the plot title
        color: Color for the CAI line
    
    Returns:
        Plotly figure object with interactive features
    """
    
    # Calculate GC content in 25bp windows around each position
    gc_content_10bp = [calculate_gc_window(sequence, pos, 25) for pos in positions]
    
    # Create a plot with two Y-axes (one for CAI, one for GC%)
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f'CAI Weights and 25bp GC Content - {seq_name}']
    )
    
    # Add the CAI trace (primary Y-axis)
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids
        ),
        secondary_y=False,
    )
    
    # Add the GC content trace (secondary Y-axis)
    theme_colors = get_consistent_color_palette(1, "optimization")
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gc_content_10bp,
            mode='lines',
            name='25bp GC Content',
            line=dict(color=theme_colors['original'], width=2, dash='dot'),
            hovertemplate='<b>Position:</b> %{x}<br><b>25bp GC Content:</b> %{y:.1f}%<extra></extra>',
            opacity=0.7
        ),
        secondary_y=True,
    )
    
    # Set up the axes
    fig.update_xaxes(title_text="Amino Acid Position")
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="GC Content (%)", secondary_y=True, range=[0, 100])
    
    # Configure the overall layout
    fig.update_layout(
        height=500,
        hovermode='x unified',  # Show all values at once when hovering
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_interactive_cai_stop_codon_plot(positions, cai_weights, amino_acids, stop_codon_positions, seq_name, frame_type, color='#4ECDC4'):
    """
    Create an interactive plot showing CAI weights with stop codon locations.
    
    Stop codons are shown as vertical bars to highlight where premature termination
    could occur in alternate reading frames. This is crucial for identifying
    potential frame-shifting issues.
    
    Args:
        positions: Amino acid positions along the sequence
        cai_weights: CAI weight at each position
        amino_acids: Amino acids at each position
        stop_codon_positions: Where stop codons appear in the alternate frame
        seq_name: Sequence name for the title
        frame_type: Which frame we're analyzing (+1, +2, etc.)
        color: Color for the CAI line
    
    Returns:
        Plotly figure with CAI line and stop codon bars
    """
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add the CAI weights as a line plot
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids
        ),
        secondary_y=False,
    )
    
    # Add stop codons as bars if we found any
    if stop_codon_positions:
        theme_colors = get_consistent_color_palette(1, "optimization")
        fig.add_trace(
            go.Bar(
                x=stop_codon_positions,
                y=[1] * len(stop_codon_positions),  # Bars extend to full height of secondary axis
                name=f'{frame_type} Stop Codons',
                marker_color=theme_colors['original'],
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br><b>Stop Codon</b><extra></extra>'
            ),
            secondary_y=True,
        )

    # Configure the axes
    fig.update_xaxes(title_text="Amino Acid Position")
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Stop Codon", secondary_y=True, showticklabels=False, range=[0, 1])
    
    # Set up the layout
    fig.update_layout(
        title=f'CAI Weights and {frame_type} Stop Codon Locations - {seq_name}',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_interactive_bar_chart(x_data, y_data, labels, title, color_scheme='viridis'):
    """
    Create a themed interactive bar chart.
    Colors automatically match the selected theme.
    """
    theme_analysis_colors = get_consistent_color_palette(len(x_data), "analysis")
    fig = go.Figure(data=go.Bar(
        x=x_data,
        y=y_data,
        text=[f'{val:.1f}' for val in y_data],  # Show values on bars
        textposition='auto',
        marker_color=theme_analysis_colors,
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig

def create_interactive_pie_chart(values, labels, title, show_percentages=True):
    """
    Create an interactive pie chart with theme colors.
    Useful for showing codon usage distribution or other proportional data.
    """
    theme_colors = THEMES[st.session_state.active_theme]["colors"]["analysis"]
    
    # Match colors to the number of slices we need
    chart_colors = []
    for i in range(len(labels)):
        chart_colors.append(theme_colors[i % len(theme_colors)])
    
    # Show either percentages or absolute counts
    textinfo = 'label+percent' if show_percentages else 'label+value'
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo=textinfo,
        marker=dict(
            colors=chart_colors,
            line=dict(color='#FFFFFF', width=2)  # White borders between slices
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center the title
            font=dict(size=14)
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05  # Position legend to the right
        ),
        margin=dict(l=20, r=120, t=50, b=20)
    )
    
    return fig

def create_interactive_comparison_chart(sequences, original_values, optimized_values, metric_name, y_title):
    """
    Create a before/after comparison chart.
    Shows original vs optimized values side-by-side for easy comparison.
    """
    fig = go.Figure()
    
    colors = get_consistent_color_palette(1, "optimization")
    
    # Original values in one color
    fig.add_trace(go.Bar(
        name='Original',
        x=sequences,
        y=original_values,
        marker_color=colors['original'],
        hovertemplate='<b>%{x}</b><br>Original ' + metric_name + ': %{y}<extra></extra>'
    ))
    
    # Optimized values in another color
    fig.add_trace(go.Bar(
        name='Optimized',
        x=sequences,
        y=optimized_values,
        marker_color=colors['optimized'],
        hovertemplate='<b>%{x}</b><br>Optimized ' + metric_name + ': %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{metric_name}: Before vs After Optimization',
        xaxis_title='Sequence',
        yaxis_title=y_title,
        barmode='group',  # Bars side-by-side, not stacked
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_stacked_bar_chart(x_data, y_data_dict, title, y_title):
    """
    Create stacked bar chart for showing composition data.
    Each bar is divided into segments representing different categories.
    """
    fig = go.Figure()
    
    colors = get_consistent_color_palette(len(y_data_dict), "analysis")
    
    for i, (label, values) in enumerate(y_data_dict.items()):
        fig.add_trace(go.Bar(
            name=label,
            x=x_data,
            y=values,
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sequence',
        yaxis_title=y_title,
        barmode='stack',  # Stack bars on top of each other
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_cai_gc_overlay_plot(
    positions, cai_weights, amino_acids, sequence, seq_name,
    plus1_stop_positions=None, minus1_stop_positions=None, slippery_positions=None,
    show_options=None,  
    color='#4ECDC4'
):
    """
    Create a comprehensive interactive plot with toggleable overlays.
    
    This is the "everything at once" view that lets users click legend items
    to show/hide different features (CAI, GC content, stop codons, slippery sites).
    Helps identify problematic regions in the sequence.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # CAI weights - always visible by default
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids,
            visible=True
        ),
        secondary_y=False,
    )

    # GC content - always visible by default
    gc_content_25bp = [calculate_gc_window(sequence, pos, 25) for pos in positions]
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=gc_content_25bp,
            mode='lines',
            name='25bp GC Content',
            line=dict(color='#888', width=2, dash='dot'),
            hovertemplate='<b>Position:</b> %{x}<br><b>25bp GC Content:</b> %{y:.1f}%<extra></extra>',
            opacity=0.7,
            visible=True
        ),
        secondary_y=True,
    )

    # +1 frame stop codons - hidden by default (click legend to show)
    if plus1_stop_positions:
        fig.add_trace(
            go.Bar(
                x=plus1_stop_positions,
                y=[100] * len(plus1_stop_positions),  # Scale to 100 to match GC axis
                name='+1 Stops',
                marker_color='#FF6B6B',
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>+1 Stop Codon<extra></extra>',
                visible='legendonly'  # Hidden until user clicks legend
            ),
            secondary_y=True,
        )

    # -1 frame stop codons - also hidden by default
    if minus1_stop_positions:
        fig.add_trace(
            go.Bar(
                x=minus1_stop_positions,
                y=[100] * len(minus1_stop_positions),
                name='-1 Stops',
                marker_color='#4ECDC4',
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>-1 Stop Codon<extra></extra>',
                visible='legendonly'
            ),
            secondary_y=True,
        )

    # Slippery sites (TTTT, TTTC motifs) - hidden by default
    if slippery_positions:
        slippery_aa_positions = [pos['amino_acid_position'] for pos in slippery_positions]
        slippery_motifs = [pos['motif'] for pos in slippery_positions]
        fig.add_trace(
            go.Bar(
                x=slippery_aa_positions,
                y=[100] * len(slippery_aa_positions),
                name='Slippery Sites',
                marker_color='#FFD700',  # Gold color for visibility
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>Motif: %{customdata}<extra></extra>',
                customdata=slippery_motifs,
                visible='legendonly'
            ),
            secondary_y=True,
        )

    # Set up axes
    fig.update_xaxes(
        title_text="Amino Acid Position",
        range=[1, len(amino_acids) + 1],  # Start at 1, not 0
        fixedrange=True  # Don't let user zoom out past data limits
    )

    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(
        title_text="GC Content (%) / Events", 
        secondary_y=True, 
        range=[0, 100],
        showticklabels=True
    )

    fig.update_layout(
        title=f'CAI/GC/Stop/Slippery Chart - {seq_name}',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder="normal"
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )

    return fig

    


def display_stateful_overlay_chart(positions, cai_weights, amino_acids, sequence, seq_name, plus1_stop_positions, minus1_stop_positions, slippery_positions, cai_color='#4ECDC4'):
    """Renders the chart as a self-contained HTML component to prevent Streamlit reruns."""

    st.info("💡 **Interactive Chart**: Click on legend items (e.g., '+1 Stops') to toggle their visibility on the chart.")

    # 1. Create the Plotly figure object as before.
    overlay_fig = create_interactive_cai_gc_overlay_plot(
    positions=positions,
    cai_weights=cai_weights,
    amino_acids=amino_acids,
    sequence=sequence,
    seq_name=seq_name,
    plus1_stop_positions=plus1_stop_positions,
    minus1_stop_positions=minus1_stop_positions,
    slippery_positions=slippery_positions,
    color=cai_color
)

    
    chart_html = overlay_fig.to_html(full_html=False, include_plotlyjs='cdn')


    components.html(chart_html, height=550, scrolling=True)

def create_interactive_cai_slippery_plot(positions, cai_weights, amino_acids, slippery_positions, seq_name, color='#4ECDC4'):
    """Create interactive plot combining CAI weights and slippery motif locations"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add CAI weights trace
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=cai_weights,
            mode='lines+markers',
            name='CAI Weight',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Position:</b> %{x}<br><b>CAI Weight:</b> %{y:.3f}<br><b>AA:</b> %{customdata}<extra></extra>',
            customdata=amino_acids
        ),
        secondary_y=False,
    )
    
    # Add slippery motif bars
    if slippery_positions:
        slippery_aa_positions = [pos['amino_acid_position'] for pos in slippery_positions]
        slippery_motifs = [pos['motif'] for pos in slippery_positions]
        
        theme_colors = get_consistent_color_palette(1, "optimization")
        fig.add_trace(
            go.Bar(
                x=slippery_aa_positions,
                y=[1] * len(slippery_aa_positions),
                name='Slippery Motifs',
                marker_color=theme_colors['original'],
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br><b>Motif:</b> %{customdata}<extra></extra>',
                customdata=slippery_motifs
            ),
            secondary_y=True,
        )

    # Set x-axis title
    fig.update_xaxes(title_text="Amino Acid Position")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Slippery Motif", secondary_y=True, showticklabels=False, range=[0, 1])
    
    # Update layout
    fig.update_layout(
        title=f'CAI Weights and Slippery Motif Locations - {seq_name}',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig

def create_enhanced_chart(data, chart_type, title, colors=None, xlabel="Sequence", ylabel="Value"):
    """Create enhanced charts with consistent styling"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set consistent styling
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    if colors is None:
        colors = get_consistent_color_palette(len(data), "analysis")
    
    if chart_type == "bar":
        bars = ax.bar(range(len(data)), data, color=colors, 
                      edgecolor='#2C3E50', linewidth=1.5, alpha=0.9)
        
        # Add value labels with consistent styling
        for bar, value in zip(bars, data):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + max(data) * 0.02,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_enhanced_summary_stats(result, original_seq=""):
    """Calculate enhanced summary statistics"""
    stats = {}
    
    # Basic metrics
    if 'Sequence_Length' in result:
        stats['Sequence_Length_bp'] = result['Sequence_Length']
    if 'Protein_Length' in result:
        stats['Protein_Length_aa'] = result['Protein_Length']
    
    # GC Content
    if 'GC_Content' in result:
        stats['GC_Content_percent'] = f"{result['GC_Content']:.1f}%"
    
    # Stop codon change (instead of reduction)
    if 'Plus1_Total_Stops' in result:
        stats['Plus1_Stop_Count'] = result['Plus1_Total_Stops']
        if original_seq:
            orig_stops = number_of_plus1_stops(original_seq)
            change = result['Plus1_Total_Stops'] - orig_stops['total']
            stats['Stop_Codon_Change'] = f"{change:+d}"
    
    # Slippery motifs
    if 'Slippery_Motifs' in result:
        stats['Slippery_Motifs'] = result['Slippery_Motifs']
    
    # CAI metrics
    if 'CAI_Weights' in result and result['CAI_Weights']:
        try:
            weights = [float(w) for w in result['CAI_Weights'].split(',')]
            stats['Average_CAI'] = f"{sum(weights)/len(weights):.3f}"
        except:
            pass
    
    # Advanced metrics
    if original_seq and 'Optimized_DNA' in result:
        orig_gc = calculate_gc_content(original_seq)
        opt_gc = calculate_gc_content(result['Optimized_DNA'])
        stats['GC_Content_Change'] = f"{opt_gc - orig_gc:+.1f}%"
    
    return stats


def count_specific_slippery_motifs(dna_seq):
    """Count in-frame slippery motifs (TTTT and TTTC at codon boundaries)"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    counts = {'TTTT': 0, 'TTTC': 0}
    for i in range(0, len(dna_seq_upper) - 3, 3):
        motif = dna_seq_upper[i:i+4]
        if motif == 'TTTT':
            counts['TTTT'] += 1
        elif motif == 'TTTC':
            counts['TTTC'] += 1
    counts['total'] = counts['TTTT'] + counts['TTTC']
    return counts

def calculate_slippery_motifs_per_100bp(dna_seq):
    """Calculate in-frame slippery motifs per 100bp"""
    sequence_length = len(dna_seq.replace(' ', '').replace('\n', ''))
    if sequence_length == 0:
        return {'TTTT': 0.0, 'TTTC': 0.0}
    slippery_counts = count_specific_slippery_motifs(dna_seq)
    return {
        'TTTT': (slippery_counts['TTTT'] / sequence_length) * 100,
        'TTTC': (slippery_counts['TTTC'] / sequence_length) * 100,
    }

def validate_dna_sequence(sequence):
    """Validate DNA sequence and return cleaned version"""
    if not sequence:
        return False, "", "No DNA sequence provided"
    cleaned = sequence.upper().replace('\n', '').replace(' ', '').replace('\t', '').replace('U', 'T')
    invalid_bases = set(cleaned) - set(VALID_DNA_BASES)
    if invalid_bases:
        return False, "", f"Invalid characters found: {', '.join(invalid_bases)}. Only A, T, G, C allowed."
    if len(cleaned) % 3 != 0:
        logger.warning(f"Sequence length ({len(cleaned)}) is not a multiple of 3")
    return True, cleaned, ""



def scan_for_immunogenic_peptides(protein_sequence, epitope_df, frame_name):
    """Scan protein sequence for immunogenic peptides"""
    findings = []
    
    if epitope_df.empty:
        return findings
    
    # Get the epitope column name from the dataframe attributes
    epitope_column = epitope_df.attrs.get('epitope_column', None)
    
    if epitope_column is None:
        # Fallback: try to find the Name column or use the 3rd column
        possible_columns = ['Name', 'Name_1', 'Name_2', 'Name_3']
        for col in possible_columns:
            if col in epitope_df.columns:
                epitope_column = col
                break
        
        if epitope_column is None and len(epitope_df.columns) >= 3:
            epitope_column = epitope_df.columns[2]  # 3rd column
    
    if epitope_column is None or epitope_column not in epitope_df.columns:
        st.error(f"Could not find epitope column. Available columns: {list(epitope_df.columns)}")
        return findings
    
    protein_upper = protein_sequence.upper()
    
    for idx, row in epitope_df.iterrows():
        try:
            epitope = str(row[epitope_column]).upper().strip()
            
            # Skip invalid entries
            if pd.isna(epitope) or epitope == 'NAN' or epitope == '' or len(epitope) < 3:
                continue
                
            # Find all occurrences of this epitope in the protein sequence
            start = 0
            while True:
                pos = protein_upper.find(epitope, start)
                if pos == -1:
                    break
                
                finding = {
                    'epitope': epitope,
                    'position': pos + 1,  # 1-based position
                    'length': len(epitope),
                    'frame': frame_name,
                    'end_position': pos + len(epitope)
                }
                
                # Add additional information from first few columns if available
                if len(epitope_df.columns) > 0 and pd.notna(row.iloc[0]):
                    finding['iedb_iri'] = row.iloc[0]
                if len(epitope_df.columns) > 1 and pd.notna(row.iloc[1]):
                    finding['object_type'] = row.iloc[1]
                
                findings.append(finding)
                start = pos + 1  # Look for overlapping occurrences
                
        except Exception as e:
            continue  # Skip problematic rows
    
    return findings

def calculate_gc_content(sequence):
    """Calculate GC content percentage of DNA sequence"""
    if not sequence:
        return 0.0
    
    clean_seq = sequence.upper().replace(' ', '').replace('\n', '')
    valid_bases = [base for base in clean_seq if base in 'ATGC']
    
    if not valid_bases:
        return 0.0
    
    gc_count = sum(1 for base in valid_bases if base in 'GC')
    return (gc_count / len(valid_bases)) * 100

def calculate_local_gc_content(sequence, window_size=10, step_size=1):
    """
    Calculate GC content for overlapping windows of a given sequence.
    Returns a list of GC percentages for each window.
    """
    gc_percentages = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window = sequence[i:i+window_size]
        gc_count = sum(1 for base in window.upper() if base in 'GC')
        gc_percentage = (gc_count / window_size) * 100
        gc_percentages.append(gc_percentage)
    return gc_percentages

def get_codon_gc_content(codon):
    """Calculate the GC content of a single 3-base codon."""
    if len(codon) != 3:
        return 0
    return (codon.upper().count('G') + codon.upper().count('C')) / 3.0 * 100

def calculate_stops_per_100bp(sequence, plus1_stops):
    """Calculate +1 frame stops per 100bp"""
    if not sequence:
        return 0.0
    
    sequence_length_bp = len(sequence)
    if sequence_length_bp == 0:
        return 0.0
    
    stops_per_100bp = (plus1_stops / sequence_length_bp) * 100
    return stops_per_100bp

def translate_dna(seq):
    """Translate DNA sequence to protein"""
    protein = ""
    genetic_code = st.session_state.genetic_code
    for i in range(0, len(seq) - 2, 3):
        codon_val = seq[i:i+3].upper()
        aa = genetic_code.get(codon_val, '?')
        protein += aa
    return protein


def codon_optimize(protein_seq):
    """Standard codon optimization using most frequent codons"""
    preferred_codons = st.session_state.preferred_codons
    optimized = ''.join(preferred_codons.get(aa, 'NNN') for aa in protein_seq if aa != 'X')
    return optimized

def get_codon_weights_row(dna_seq):
    """Calculate CAI weights for DNA sequence"""
    codon_weights = st.session_state.codon_weights
    codons_list = [dna_seq[i:i+3].upper() for i in range(0, len(dna_seq) - 2, 3)]
    weights = [codon_weights.get(c, 1e-6) for c in codons_list]
    return weights, codons_list


def number_of_slippery_motifs(dna_seq):
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos = 0
    search_end = len(dna_seq_upper) - 4
    slippery_count = 0
    for i in range(start_pos, search_end, 3):
        codon = dna_seq_upper[i:i+3]
        next_base = dna_seq_upper[i+3] if i+3 < len(dna_seq_upper) else ''
        if codon == 'TTT' and next_base in ('T', 'C'):
            slippery_count += 1
    return slippery_count

def get_slippery_motif_positions(dna_seq):
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    start_pos = 0
    search_end = len(dna_seq_upper) - 4
    slippery_positions = []
    for i in range(start_pos, search_end, 3):
        codon = dna_seq_upper[i:i+3]
        next_base = dna_seq_upper[i+3] if i+3 < len(dna_seq_upper) else ''
        if codon == 'TTT' and next_base in ('T', 'C'):
            motif = codon + next_base
            aa_position = ((i - start_pos) // 3) + 1
            slippery_positions.append({
                'motif': motif,
                'nucleotide_position': i + 1,
                'amino_acid_position': aa_position,
                'codon_position': f"{i+1}-{i+4}"
            })
    return slippery_positions




def number_of_plus1_stops(dna_seq):
    """Count stop codons in +1 frame across the entire sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    counts = Counter()

    for i in range(1, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def number_of_minus1_stops(dna_seq):
    """Count stop codons in -1 frame across the entire sequence"""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    counts = Counter()

    for i in range(2, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            counts[codon] += 1
    
    total_stops = sum(counts.values())
    return {'TAA': counts['TAA'], 'TAG': counts['TAG'], 'TGA': counts['TGA'], 'total': total_stops}

def get_plus1_stop_positions(dna_seq):
    """Get positions of stop codons in +1 frame"""
    positions = []
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    for i in range(1, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            aa_position = (i // 3) + 1
            positions.append(aa_position)
    return positions

def get_minus1_stop_positions(dna_seq):
    """Get positions of stop codons in -1 frame"""
    positions = []
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons_set = {"TAA", "TAG", "TGA"}
    
    for i in range(2, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if codon in stop_codons_set:
            aa_position = (i // 3) + 1
            positions.append(aa_position)
    return positions

def balanced_optimisation(dna_seq, bias_weight_input=None):
    """Uses EXACT original equation and bonus system but compares options at each position to avoid skipping"""
    bias_weight = bias_weight_input if bias_weight_input is not None else st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)
    
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    aa_to_codons = st.session_state.aa_to_codons
    
    # Protein translation
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))
    
    def get_highest_freq_codon(amino_acid):
        """Get the highest frequency codon for an amino acid"""
        if amino_acid in aa_to_codons and aa_to_codons[amino_acid]:
            return max(aa_to_codons[amino_acid], key=lambda x: x[1])[0]
        else:
            for codon, aa in genetic_code.items():
                if aa == amino_acid:
                    return codon
            return "NNN"
    
    def get_canonical_future_sequence(protein_remainder):
        """Convert remaining protein to canonical codons"""
        canonical = ""
        for aa in protein_remainder:
            canonical += get_highest_freq_codon(aa)
        return canonical
    
    def calculate_original_bonus(temp_seq, current_pos, num_codons):
        """
        EXACT original bonus calculation:
        - Single codon: check position current_pos + 1, bonus = 1 if stop, 0 otherwise
        - Two codon: check positions current_pos + 1 and current_pos + 4
                    bonus = 2 if both stops, 1 if one stop, 0 if no stops
        """
        if num_codons == 1:
            # Single codon: check only current_pos + 1
            plus1_pos = current_pos + 1
            if plus1_pos + 3 <= len(temp_seq):
                codon_plus1 = temp_seq[plus1_pos:plus1_pos+3]
                if codon_plus1 in PLUS1_STOP_CODONS:
                    return 1
            return 0
        
        elif num_codons == 2:
            # Two codon: check current_pos + 1 and current_pos + 4
            plus1_pos1 = current_pos + 1
            plus1_pos2 = current_pos + 4
            
            stop1 = False
            stop2 = False
            
            if plus1_pos1 + 3 <= len(temp_seq):
                codon1_plus1 = temp_seq[plus1_pos1:plus1_pos1+3]
                if codon1_plus1 in PLUS1_STOP_CODONS:
                    stop1 = True
            
            if plus1_pos2 + 3 <= len(temp_seq):
                codon2_plus1 = temp_seq[plus1_pos2:plus1_pos2+3]
                if codon2_plus1 in PLUS1_STOP_CODONS:
                    stop2 = True
            
            # Original bonus system: 2 for both, 1 for one, 0 for none
            if stop1 and stop2:
                return 2
            elif stop1 or stop2:
                return 1
            else:
                return 0
        
        return 0
    
    def evaluate_single_codon(idx, current_optimised, aa):
        """Evaluate single codon using EXACT original equation"""
        if aa not in aa_to_codons:
            current_codon = dna_seq_upper[idx:idx+3]
            return current_codon, 0
        
        remaining_protein = protein_str[len(current_optimised)//3 + 1:]
        canonical_future = get_canonical_future_sequence(remaining_protein)
        
        best_codon = dna_seq_upper[idx:idx+3]
        best_score = -1
        
        # Get current codon frequency for baseline
        current_codon = dna_seq_upper[idx:idx+3]
        current_freq = 0
        for syn_c, freq_val in aa_to_codons[aa]:
            if syn_c == current_codon:
                current_freq = freq_val
                break
        
        # Calculate baseline score for current codon
        temp_seq_orig = current_optimised + current_codon + canonical_future
        bonus_orig = calculate_original_bonus(temp_seq_orig, len(current_optimised), 1)
        baseline_score = current_freq + bias_weight * bonus_orig
        best_score = baseline_score
        
        for syn_codon, freq in aa_to_codons[aa]:
            temp_seq = current_optimised + syn_codon + canonical_future
            
            # EXACT ORIGINAL EQUATION: score = frequency + bias_weight * bonus
            bonus = calculate_original_bonus(temp_seq, len(current_optimised), 1)
            score = freq + bias_weight * bonus
            
            # Deterministic tie-breaking (same as original)
            if score > best_score or (score == best_score and syn_codon < best_codon):
                best_score = score
                best_codon = syn_codon
        
        return best_codon, best_score
    
    def evaluate_two_codon(idx, current_optimised, aa1, aa2):
        """Evaluate two-codon using EXACT original equation"""
        if aa1 not in aa_to_codons or aa2 not in aa_to_codons:
            return None, None, -1
        
        remaining_protein = protein_str[len(current_optimised)//3 + 2:]
        canonical_future = get_canonical_future_sequence(remaining_protein)
        
        best_c1, best_c2 = None, None
        best_score = -1
        
        for c1, f1 in aa_to_codons[aa1]:
            for c2, f2 in aa_to_codons[aa2]:
                temp_seq = current_optimised + c1 + c2 + canonical_future
                
                # EXACT ORIGINAL EQUATION: score = (f1 * f2) + bias_weight * bonus
                bonus = calculate_original_bonus(temp_seq, len(current_optimised), 2)
                score = (f1 * f2) + bias_weight * bonus
                
                # Deterministic tie-breaking (same as original)
                if score > best_score or (score == best_score and c1 + c2 < (best_c1 or "") + (best_c2 or "")):
                    best_score = score
                    best_c1, best_c2 = c1, c2
        
        return best_c1, best_c2, best_score
    
    optimised_seq = ""
    idx = 0
    
    while idx < len(dna_seq_upper) - 2:
        current_codon = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(current_codon, str(Seq(current_codon).translate()))
        
        # ALWAYS evaluate both single and two-codon options at each position
        single_codon, single_score = evaluate_single_codon(idx, optimised_seq, aa)
        best_option = ("single", single_codon, single_score, 3)
        
        # Two codon option (if possible)
        if idx < len(dna_seq_upper) - 5:
            next_codon = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(next_codon, str(Seq(next_codon).translate()))
            
            two_c1, two_c2, two_score = evaluate_two_codon(idx, optimised_seq, aa, aa2)
            if two_score > best_option[2]:
                best_option = ("two", (two_c1, two_c2), two_score, 6)
        
        # Apply the best option
        option_type, codons, score, advance = best_option
        
        if option_type == "single":
            optimised_seq += codons
        elif option_type == "two":
            optimised_seq += codons[0] + codons[1]
        
        idx += advance
    
    # Handle remaining nucleotides
    if idx < len(dna_seq_upper):
        optimised_seq += dna_seq_upper[idx:]
    
    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in original equation no-skipping optimization!")
        return dna_seq_upper
    
    return optimised_seq

def MaxStop(dna_seq):
    """MaxStop optimization with fixed bias_weight=1 (not tunable)"""
    bias_weight = 1
    
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    aa_to_codons = st.session_state.aa_to_codons
    
    # Protein translation
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))
    
    def get_highest_freq_codon(amino_acid):
        """Get the highest frequency codon for an amino acid"""
        if amino_acid in aa_to_codons and aa_to_codons[amino_acid]:
            return max(aa_to_codons[amino_acid], key=lambda x: x[1])[0]
        else:
            for codon, aa in genetic_code.items():
                if aa == amino_acid:
                    return codon
            return "NNN"
    
    def get_canonical_future_sequence(protein_remainder):
        """Convert remaining protein to canonical codons"""
        canonical = ""
        for aa in protein_remainder:
            canonical += get_highest_freq_codon(aa)
        return canonical
    
    def calculate_original_bonus(temp_seq, current_pos, num_codons):
        """
        EXACT original bonus calculation:
        - Single codon: check position current_pos + 1, bonus = 1 if stop, 0 otherwise
        - Two codon: check positions current_pos + 1 and current_pos + 4
                    bonus = 2 if both stops, 1 if one stop, 0 if no stops
        """
        if num_codons == 1:
            # Single codon: check only current_pos + 1
            plus1_pos = current_pos + 1
            if plus1_pos + 3 <= len(temp_seq):
                codon_plus1 = temp_seq[plus1_pos:plus1_pos+3]
                if codon_plus1 in PLUS1_STOP_CODONS:
                    return 1
            return 0
        
        elif num_codons == 2:
            # Two codon: check current_pos + 1 and current_pos + 4
            plus1_pos1 = current_pos + 1
            plus1_pos2 = current_pos + 4
            
            stop1 = False
            stop2 = False
            
            if plus1_pos1 + 3 <= len(temp_seq):
                codon1_plus1 = temp_seq[plus1_pos1:plus1_pos1+3]
                if codon1_plus1 in PLUS1_STOP_CODONS:
                    stop1 = True
            
            if plus1_pos2 + 3 <= len(temp_seq):
                codon2_plus1 = temp_seq[plus1_pos2:plus1_pos2+3]
                if codon2_plus1 in PLUS1_STOP_CODONS:
                    stop2 = True
            
            # Original bonus system: 2 for both, 1 for one, 0 for none
            if stop1 and stop2:
                return 2
            elif stop1 or stop2:
                return 1
            else:
                return 0
        
        return 0
    
    def evaluate_single_codon(idx, current_optimised, aa):
        """Evaluate single codon using EXACT original equation"""
        if aa not in aa_to_codons:
            current_codon = dna_seq_upper[idx:idx+3]
            return current_codon, 0
        
        remaining_protein = protein_str[len(current_optimised)//3 + 1:]
        canonical_future = get_canonical_future_sequence(remaining_protein)
        
        best_codon = dna_seq_upper[idx:idx+3]
        best_score = -1
        
        # Get current codon frequency for baseline
        current_codon = dna_seq_upper[idx:idx+3]
        current_freq = 0
        for syn_c, freq_val in aa_to_codons[aa]:
            if syn_c == current_codon:
                current_freq = freq_val
                break
        
        # Calculate baseline score for current codon
        temp_seq_orig = current_optimised + current_codon + canonical_future
        bonus_orig = calculate_original_bonus(temp_seq_orig, len(current_optimised), 1)
        baseline_score = current_freq + bias_weight * bonus_orig
        best_score = baseline_score
        
        for syn_codon, freq in aa_to_codons[aa]:
            temp_seq = current_optimised + syn_codon + canonical_future
            
            # EXACT ORIGINAL EQUATION: score = frequency + bias_weight * bonus
            bonus = calculate_original_bonus(temp_seq, len(current_optimised), 1)
            score = freq + bias_weight * bonus
            
            # Deterministic tie-breaking (same as original)
            if score > best_score or (score == best_score and syn_codon < best_codon):
                best_score = score
                best_codon = syn_codon
        
        return best_codon, best_score
    
    def evaluate_two_codon(idx, current_optimised, aa1, aa2):
        """Evaluate two-codon using EXACT original equation"""
        if aa1 not in aa_to_codons or aa2 not in aa_to_codons:
            return None, None, -1
        
        remaining_protein = protein_str[len(current_optimised)//3 + 2:]
        canonical_future = get_canonical_future_sequence(remaining_protein)
        
        best_c1, best_c2 = None, None
        best_score = -1
        
        for c1, f1 in aa_to_codons[aa1]:
            for c2, f2 in aa_to_codons[aa2]:
                temp_seq = current_optimised + c1 + c2 + canonical_future
                
                # EXACT ORIGINAL EQUATION: score = (f1 * f2) + bias_weight * bonus
                bonus = calculate_original_bonus(temp_seq, len(current_optimised), 2)
                score = (f1 * f2) + bias_weight * bonus
                
                # Deterministic tie-breaking (same as original)
                if score > best_score or (score == best_score and c1 + c2 < (best_c1 or "") + (best_c2 or "")):
                    best_score = score
                    best_c1, best_c2 = c1, c2
        
        return best_c1, best_c2, best_score
    
    optimised_seq = ""
    idx = 0
    
    while idx < len(dna_seq_upper) - 2:
        current_codon = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(current_codon, str(Seq(current_codon).translate()))
        
        # ALWAYS evaluate both single and two-codon options at each position
        single_codon, single_score = evaluate_single_codon(idx, optimised_seq, aa)
        best_option = ("single", single_codon, single_score, 3)
        
        # Two codon option (if possible)
        if idx < len(dna_seq_upper) - 5:
            next_codon = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(next_codon, str(Seq(next_codon).translate()))
            
            two_c1, two_c2, two_score = evaluate_two_codon(idx, optimised_seq, aa, aa2)
            if two_score > best_option[2]:
                best_option = ("two", (two_c1, two_c2), two_score, 6)
        
        # Apply the best option
        option_type, codons, score, advance = best_option
        
        if option_type == "single":
            optimised_seq += codons
        elif option_type == "two":
            optimised_seq += codons[0] + codons[1]
        
        idx += advance
    
    # Handle remaining nucleotides
    if idx < len(dna_seq_upper):
        optimised_seq += dna_seq_upper[idx:]
    
    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in MaxStop optimization!")
        return dna_seq_upper
    
    return optimised_seq



def third_aa_has_A_G_synonymous(aa):
    """Check if amino acid has synonymous codons starting with A or G"""
    for codon_val in synonymous_codons.get(aa, []):
        if codon_val.startswith(('A', 'G')):
            return True
    return False


def load_codon_data_from_file(file_content):
    """Load codon usage data from uploaded file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        required_columns = ['triplet', 'amino_acid', 'fraction']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['triplet'] = df['triplet'].str.upper().str.strip()
        df['amino_acid'] = df['amino_acid'].str.upper().str.strip().replace({'*': 'X'})
        df = df.dropna(subset=['triplet', 'amino_acid', 'fraction'])
        
        genetic_code = df.set_index('triplet')['amino_acid'].to_dict()
        max_fraction = df.groupby('amino_acid')['fraction'].transform('max')
        df['weight'] = df['fraction'] / max_fraction
        codon_weights = df.set_index('triplet')['weight'].to_dict()
        preferred_codons = df.sort_values('fraction', ascending=False).drop_duplicates('amino_acid').set_index('amino_acid')['triplet'].to_dict()
        human_codon_usage = df.set_index('triplet')['fraction'].to_dict()
        
        aa_to_codons = defaultdict(list)
        for codon_val, freq in human_codon_usage.items():
            aa = genetic_code.get(codon_val, None)
            if aa and aa != 'X':
                aa_to_codons[aa].append((codon_val, freq))
        
        return genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, df
    except Exception as e:
        raise Exception(f"Error loading codon file: {e}")


def create_download_link(df, filename):
    """Create download link for DataFrame as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    processed_data = output.getvalue()
    return processed_data



def run_single_optimization(sequence, method, bias_weight=None):
    """Run single sequence optimization"""
    is_valid, clean_seq, error_msg = validate_dna_sequence(sequence)
    if not is_valid:
        return None, error_msg
    
    try:
        protein_seq = translate_dna(clean_seq)
        
        if method == "Standard Codon Optimization":
            optimized = codon_optimize(protein_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "In-Frame Analysis":  # Updated from "CAI Weight Analysis"
            weights, codons_list = get_codon_weights_row(clean_seq)
            slippery_motifs = number_of_slippery_motifs(clean_seq)
            result = {
                'Position': list(range(1, len(codons_list) + 1)),
                'DNA_Codon': codons_list,
                'CAI_Weight': weights,
                'Amino_Acid': [st.session_state.genetic_code.get(c, '?') for c in codons_list],
                'Slippery_Motifs': slippery_motifs,
                'Method': method
            }
        elif method == "Balanced Optimization":
            optimized = balanced_optimisation(clean_seq, bias_weight)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        elif method == "MaxStop":
            if 'maxstop_result_seq' in st.session_state:
                del st.session_state['maxstop_result_seq']
            optimized = MaxStop(clean_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ', '.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
       
        elif method == "+1 Frame Analysis":  # Updated from "Sequence Analysis"
            plus1_stop_counts = number_of_plus1_stops(clean_seq)
            start_pos, end_pos = find_coding_sequence_bounds(clean_seq)
            slippery_count = number_of_slippery_motifs(clean_seq)
            gc_content = calculate_gc_content(clean_seq)
            minus1_stop_counts = number_of_minus1_stops(clean_seq)
            
            if start_pos is not None and end_pos is not None:
                coding_length = end_pos - start_pos
                plus1_len = coding_length // 3
                coding_info = f"{start_pos}-{end_pos} ({coding_length} bp)"
            elif start_pos is not None:
                coding_length = len(clean_seq) - start_pos
                plus1_len = coding_length // 3
                coding_info = f"{start_pos}-end ({coding_length} bp, no stop found)"
            else:
                plus1_len = 0
                coding_info = "No valid coding sequence found"
                coding_length = 0
            
            result = {
                'Sequence_Length': len(clean_seq),
                'Protein_Length': len(protein_seq),
                'GC_Content': gc_content,
                'Coding_Info': coding_info,
                'Plus1_TAA_Count': plus1_stop_counts['TAA'],
                'Plus1_TAG_Count': plus1_stop_counts['TAG'],
                'Plus1_TGA_Count': plus1_stop_counts['TGA'],
                'Plus1_Total_Stops': plus1_stop_counts['total'],
                'minus1_TAA_Count': minus1_stop_counts['TAA'],
                'minus1_TAG_Count': minus1_stop_counts['TAG'],
                'minus1_TGA_Count': minus1_stop_counts['TGA'],
                'minus1_Total_Stops': minus1_stop_counts['total'],
                'Slippery_Motifs': slippery_count,
                'Stop_Density': plus1_stop_counts['total']/max(1, plus1_len) if plus1_len > 0 else 0,
                'Method': method
            }
        
        return result, None
    except Exception as e:
        return None, str(e)
    




def translate_frame(dna_sequence, frame_offset):
    """Translate DNA sequence in a specific frame (0, 1, or 2 for +1, +2, +3 frames; or -1, -2 for other frames)"""
    try:
        if frame_offset >= 0:
            # Positive frame (0 = normal, 1 = +1 frame, 2 = +2 frame)
            start_pos = frame_offset
        else:
            # Negative frame (-1 = -1 frame, -2 = -2 frame)
            start_pos = len(dna_sequence) + frame_offset
            if start_pos < 0:
                start_pos = 0
        
        protein = ""
        genetic_code = st.session_state.genetic_code
        
        for i in range(start_pos, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i+3].upper()
            if len(codon) == 3:
                aa = genetic_code.get(codon, 'X')
                protein += aa
            else:
                break
        
        return protein
    except Exception as e:
        return ""



def create_immunogenic_peptide_summary(findings_plus1, findings_minus1):
    """Create a summary of immunogenic peptide findings"""
    if not findings_plus1 and not findings_minus1:
        return None
    
    all_findings = []
    
    # Add +1 frame findings
    for finding in findings_plus1:
        finding_copy = finding.copy()
        finding_copy['frame'] = '+1 Frame'
        all_findings.append(finding_copy)
    
    # Add -1 frame findings
    for finding in findings_minus1:
        finding_copy = finding.copy()
        finding_copy['frame'] = '-1 Frame'
        all_findings.append(finding_copy)
    
    if not all_findings:
        return None
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_findings)
    
    # Reorder columns for better display
    priority_cols = ['frame', 'epitope', 'position', 'end_position', 'length']
    other_cols = [col for col in summary_df.columns if col not in priority_cols]
    summary_df = summary_df[priority_cols + other_cols]
    
    return summary_df

def main():
    """Main Streamlit application"""

  
    # Apply the selected theme CSS
    inject_app_theme()
    # Initialize research engines
    

    st.title("🐎 Harmonized Optimization of Oligos and Frames")
    st.markdown("Welcome to HOOF: your optimization and analysis companion!")

    with st.expander("Read Me"):
        st.markdown('''
        ### Optimization Algorithms
        - **Standard Codon Optimization**: This method replaces each codon in your sequence with the most frequently used synonymous codon from the provided codon usage table. This is a straightforward way to potentially increase protein expression levels.
        - **Balanced Optimization**: This algorithm considers both codon usage frequency and the introduction of +1 frameshift-inducing stop codons. It tries to find a balance between using high-frequency codons and strategically placing codons that can terminate out-of-frame translation, which can be beneficial for mRNA vaccine design. The "Bias Weight" slider in the sidebar allows you to control how strongly the algorithm favors introducing these +1 stop codons.
        - **MaxStop**: This method specifically aims to introduce TAA or TAG stop codons in the +1 reading frame. It can perform double substitutions to create stop-stop motifs like TAATAA or TAGTAG.
        

        ### Analysis
        - **+1 Frame Analysis**: This analysis scans the sequence for stop codons in the +1 reading frame. It also includes a feature to scan for known immunogenic peptides in all three reading frames (+0, +1, -1). This is useful for identifying potential off-target immune responses from your translated sequence and its out-of-frame products. 
        - **In-frame Analysis**: This option analyzes the input sequence in its primary reading frame (0). It calculates the Codon Adaptation Index (CAI) for each codon, which is a measure of how well the codon is adapted to the codon usage of a reference organism (in this case, humans by default). It also calculates the GC content of the sequence. This analysis is useful for assessing the baseline quality of your sequence before optimization.

        
        ''')
    
    # Sidebar for settings and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Define available codon files
        CODON_FILES = {
            "Human (Homo sapiens)": "HumanCodons.xlsx",
            "Mouse (Mus musculus)": "MouseCodons.xlsx", 
            "E. coli": "E.coliCodons.xlsx"
        }
        
        # Check which files actually exist
        available_files = {}
        for species, filename in CODON_FILES.items():
            if os.path.exists(filename):
                available_files[species] = filename
        
        # Codon usage file selection
        st.subheader("Codon Usage Selection")
        
        if available_files:
            if 'selected_codon_file' not in st.session_state:
                st.session_state.selected_codon_file = list(available_files.keys())[0]  # First available
            
            if st.session_state.selected_codon_file not in available_files:
                st.session_state.selected_codon_file = list(available_files.keys())[0]
            
            selected_codon_species = st.selectbox(
                "Select organism codon usage:",
                list(available_files.keys()),
                index=list(available_files.keys()).index(st.session_state.selected_codon_file),
                key="codon_species_selector"
            )
            
            if selected_codon_species != st.session_state.selected_codon_file:
                st.session_state.selected_codon_file = selected_codon_species
                st.session_state.genetic_code = {}
                st.session_state.codon_weights = {}
                st.session_state.preferred_codons = {}
                st.session_state.human_codon_usage = {}
                st.session_state.aa_to_codons = defaultdict(list)
                if 'codon_data_loaded' in st.session_state:
                    del st.session_state.codon_data_loaded
                if 'codon_file_source' in st.session_state:
                    del st.session_state.codon_file_source
            
            # Auto-load selected codon file if not already loaded
            selected_file_path = available_files[selected_codon_species]
            
            if not st.session_state.genetic_code and 'codon_data_loaded' not in st.session_state:
                try:
                    with open(selected_file_path, 'rb') as f:
                        file_content = f.read()
                    genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                    st.session_state.genetic_code = genetic_code
                    st.session_state.codon_weights = codon_weights
                    st.session_state.preferred_codons = preferred_codons
                    st.session_state.human_codon_usage = human_codon_usage
                    st.session_state.aa_to_codons = aa_to_codons
                    st.session_state.codon_data_loaded = True
                    st.session_state.codon_file_source = f"{selected_codon_species} ({selected_file_path})"
                    st.success(f"✅ Loaded {len(codon_df)} codon entries from {selected_codon_species}")
                except Exception as e:
                    st.error(f"❌ Could not load {selected_file_path}: {e}")
        
        else:
            # No organism files available - show file status and allow upload
            st.warning("⚠️ No organism codon files found")
            st.markdown("**Missing files:**")
            for species, filename in CODON_FILES.items():
                st.write(f"❌ {filename} ({species})")
            st.info("💡 Use the upload option below or add codon files to the application directory")
        
        # Display current codon file status
        if st.session_state.genetic_code:
            codon_source = st.session_state.get('codon_file_source', 'Unknown')
            st.success(f"**Active:** {codon_source}")
            
            # Show some basic stats about the loaded codon usage
            with st.expander("📊 Codon Usage Stats", expanded=False):
                if st.session_state.human_codon_usage:
                    num_codons = len(st.session_state.human_codon_usage)
                    num_amino_acids = len(st.session_state.aa_to_codons)
                    avg_frequency = sum(st.session_state.human_codon_usage.values()) / num_codons if num_codons > 0 else 0
                    
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("Total Codons", num_codons)
                        st.metric("Amino Acids", num_amino_acids)
                    with stat_col2:
                        st.metric("Avg Frequency", f"{avg_frequency:.3f}")
                        
                    # Show top 5 most frequent codons
                    top_codons = sorted(st.session_state.human_codon_usage.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.markdown("**Top 5 Codons:**")
                    for codon, freq in top_codons:
                        aa = st.session_state.genetic_code.get(codon, '?')
                        st.write(f"• {codon} ({aa}): {freq:.3f}")
            
            if st.button("🔄 Switch Codon Usage", help="Change to a different organism's codon usage"):
                # Clear current data to force reload
                st.session_state.genetic_code = {}
                st.session_state.codon_weights = {}
                st.session_state.preferred_codons = {}
                st.session_state.human_codon_usage = {}
                st.session_state.aa_to_codons = defaultdict(list)
                if 'codon_data_loaded' in st.session_state:
                    del st.session_state.codon_data_loaded
                if 'codon_file_source' in st.session_state:
                    del st.session_state.codon_file_source
                st.rerun()
        
        # Manual file upload (always available)
        st.markdown("---")
        st.markdown("**Upload Codon Usage File**")
        uploaded_file = st.file_uploader(
            "Upload Codon Usage File (.xlsx)", 
            type=['xlsx'],
            help="Upload a codon usage frequency file (Excel format)",
            key="codon_uploader"
        )
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read()
                genetic_code, codon_weights, preferred_codons, human_codon_usage, aa_to_codons, codon_df = load_codon_data_from_file(file_content)
                st.session_state.genetic_code = genetic_code
                st.session_state.codon_weights = codon_weights
                st.session_state.preferred_codons = preferred_codons
                st.session_state.human_codon_usage = human_codon_usage
                st.session_state.aa_to_codons = aa_to_codons
                st.session_state.codon_data_loaded = True
                st.session_state.codon_file_source = f"Custom Upload ({uploaded_file.name})"
                st.session_state.selected_codon_file = f"Custom ({uploaded_file.name})"
                st.success(f"✅ Loaded {len(codon_df)} codon entries from {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading codon file: {e}")
        
        # Only show warning if no codon data is loaded at all
        if not st.session_state.genetic_code:
            st.warning("⚠️ **No codon usage data loaded**")
            st.info("Please upload a codon usage file to continue, or add organism files to the application directory.")
        
        st.divider()
        
        st.subheader("Algorithm Settings")

        bias_weight = st.slider(
            "Bias Weight (Balanced Optimization)", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)),
            step=0.01,  # smaller step for finer control
            help="Weight for +1 frame stop codon bias in balanced optimization"
        )

        st.session_state.config["bias_weight"] = bias_weight
        st.divider()

        
        # Theme selection
        st.subheader("Appearance")
        theme_name = st.selectbox(
            "Select Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.active_theme),
            help="Change the color scheme of the application."
        )
        if theme_name != st.session_state.active_theme:
            st.session_state.active_theme = theme_name
            st.rerun()
        
        st.info(THEMES[st.session_state.active_theme]["info"])
        
        # Accumulation settings
        st.subheader("Result Management")
        accumulate_results = st.checkbox(
            "Accumulate Results", 
            help="Collect multiple single-sequence results before download"
        )
        
        if st.session_state.accumulated_results:
            st.info(f"Accumulated: {len(st.session_state.accumulated_results)} results")
            if st.button("Clear Accumulated Results"):
                st.session_state.accumulated_results = []
                st.session_state.run_counter = 0
                st.rerun()

    # Main interface tabs
    tab1, tab2, tab6 = st.tabs(["Single Sequence", "Batch Optimization", "About"])


    with tab1:
        st.header("Single Sequence Optimization")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Check for and handle transferred sequence
            if 'transfer_sequence' in st.session_state and st.session_state.transfer_sequence:
                transfer_info = st.session_state.get('transfer_sequence_info', {})
                source_info = transfer_info.get('source', 'another tab')
                st.success(f"🎯 Sequence from **{source_info}** has been loaded below.")
                
                st.session_state.sequence_input_area = st.session_state.transfer_sequence
                
                # Clean up session state after using the transferred sequence
                del st.session_state.transfer_sequence
                if 'transfer_sequence_info' in st.session_state:
                    del st.session_state.transfer_sequence_info

            sequence_input = st.text_area(
                "DNA Sequence",
                height=150,
                placeholder="Enter DNA sequence (A, T, G, C only) - CODING SEQUENCE ONLY",
                help="Paste your DNA sequence here. Spaces and newlines will be removed automatically. You can also transfer sequences from the CDS Database Search tab.",
                key="sequence_input_area"
            )
        
        with col2:
            operation_type = st.radio(
                "Choose Operation Type",
                ("Optimization", "Analysis"),
                key="single_op_type"
            )

            if operation_type == "Optimization":
                method_to_run = st.selectbox(
                    "Choose Optimization Method",
                    [
                        "Standard Codon Optimization",
                        "MaxStop",
                        "Balanced Optimization",
                    ],
                    help="Choose the optimization algorithm to apply"
                )
            else:  # Analysis
                method_to_run = st.selectbox(
                    "Choose Analysis Method",
                    [
                        "In-Frame Analysis",
                        "+1 Frame Analysis",
                    ],
                    help="Choose the analysis algorithm to apply"
                )

            # Accumulation settings moved here
            st.markdown("**Result Management:**")
            accumulate_results = st.checkbox(
                "Accumulate Results",
                help="Collect multiple single-sequence results before download",
                key="accumulate_results_tab1"
            )

            if st.session_state.accumulated_results:
                st.info(f"Accumulated: {len(st.session_state.accumulated_results)} results")
                if st.button("Clear Accumulated Results", key="clear_accumulated_tab1"):
                    st.session_state.accumulated_results = []
                    st.session_state.run_counter = 0
                    st.rerun()

            run_button = st.button(f"Run {operation_type}", type="primary")

        # Results section - using full width outside of columns
        if run_button:
            if not sequence_input.strip():
                st.error("Please enter a DNA sequence")
            else:
                with st.spinner("Processing sequence..."):
                    result, error = run_single_optimization(sequence_input, method_to_run, bias_weight)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Optimization completed successfully - Scroll down to see some magical results!")
                    
                    # Full-width results section
                    st.divider()
                    
                    # Display results using full page width
                    if method_to_run == "In-Frame Analysis":
                        df = pd.DataFrame(result)
                        st.subheader("In-Frame Analysis Results")
                        
                        # Create interactive In-Frame graph with GC content
                        if not df.empty and 'CAI_Weight' in df.columns:
                            st.subheader("📊 Interactive CAI/GC/Stop/Slippery Chart")
                            
                            positions = df['Position'].tolist()
                            cai_weights = df['CAI_Weight'].tolist()
                            amino_acids = df['Amino_Acid'].tolist()
                            plus1_stop_positions = get_plus1_stop_positions(sequence_input)
                            minus1_stop_positions = get_minus1_stop_positions(sequence_input)
                            slippery_positions = get_slippery_motif_positions(sequence_input)
                            colors = get_consistent_color_palette(1, "optimization")
                            fig = create_interactive_cai_gc_overlay_plot(
                                positions,
                                cai_weights,
                                amino_acids,
                                sequence_input,
                                f"Sequence ({len(sequence_input)} bp)",
                                plus1_stop_positions=plus1_stop_positions,
                                minus1_stop_positions=minus1_stop_positions,
                                slippery_positions=slippery_positions,
                                
                                color=colors['optimized']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            
                            st.subheader("📊 Summary Statistics")
                            # Calculate enhanced summary stats
                            sequence_length = len(sequence_input.replace('\n', '').replace(' ', ''))
                            protein_length = len(df['Amino_Acid']) if 'Amino_Acid' in df else 0
                            gc_content = calculate_gc_content(sequence_input)
                            average_cai = np.mean(df['CAI_Weight']) if 'CAI_Weight' in df else 0
                            slippery_motifs = number_of_slippery_motifs(sequence_input)

                            col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
                            with col_sum1:
                                st.metric("Sequence Length", f"{sequence_length} bp")
                            with col_sum2:
                                st.metric("Protein Length", f"{protein_length} aa")
                            with col_sum3:
                                st.metric("GC Content", f"{gc_content:.1f}%")
                            with col_sum4:
                                st.metric("Average CAI", f"{average_cai:.3f}")
                            with col_sum5:
                                st.metric("Slippery Motifs", slippery_motifs)
                        # Slippery motif locations
                        st.subheader("📍 Slippery Motif Locations")
                        slippery_positions = get_slippery_motif_positions(sequence_input)
                        if slippery_positions:
                            slippery_df = pd.DataFrame(slippery_positions)
                            slippery_df.columns = ['Motif', 'Nucleotide Position', 'Amino Acid Position', 'Codon Range']
                            st.dataframe(slippery_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("✅ No slippery motifs (TTTT or TTTC) found in the coding sequence.")

                        with st.expander("View Detailed In-Frame Data"):
                            st.dataframe(df, use_container_width=True)
                        
                    elif method_to_run == "+1 Frame Analysis":
                        st.subheader("+1 Frame Analysis Results")
                        
                        # Load immunogenic peptides
                        epitope_df = load_immunogenic_peptides()
                        
                        # Create metrics display using full width
                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
                        with metric_col1:
                            st.metric("Sequence Length", f"{result['Sequence_Length']} bp")
                        with metric_col2:
                            st.metric("Protein Length", f"{result['Protein_Length']} aa")
                        with metric_col3:
                            st.metric("GC Content", f"{result['GC_Content']:.1f}%")
                        with metric_col4:
                            st.metric("Total +1 Stops", result['Plus1_Total_Stops'])
                        with metric_col5:
                            st.metric("Slippery Motifs", result['Slippery_Motifs'])
                        with metric_col6:
                            st.metric("Total -1 Stops", result['minus1_Total_Stops'])

                       

                        col1, col2 = st.columns(2)

                        with col1:
                            # +1 Stop codon distribution pie chart
                            if result['Plus1_Total_Stops'] > 0:
                                st.markdown("#### 🥧 +1 Frame Stop Codons")
                                pie_data_plus1 = []
                                pie_labels_plus1 = []
                                if 'Plus1_TAA_Count' in result and result['Plus1_TAA_Count'] > 0:
                                    pie_data_plus1.append(result['Plus1_TAA_Count'])
                                    pie_labels_plus1.append('TAA')
                                if 'Plus1_TAG_Count' in result and result['Plus1_TAG_Count'] > 0:
                                    pie_data_plus1.append(result['Plus1_TAG_Count'])
                                    pie_labels_plus1.append('TAG')
                                if 'Plus1_TGA_Count' in result and result['Plus1_TGA_Count'] > 0:
                                    pie_data_plus1.append(result['Plus1_TGA_Count'])
                                    pie_labels_plus1.append('TGA')

                                fig_pie_plus1 = create_interactive_pie_chart(pie_data_plus1, pie_labels_plus1, "+1 Frame Stop Codon Distribution", show_percentages=False)
                                st.plotly_chart(fig_pie_plus1, use_container_width=True, key="single_plus1_pie_chart")
                            else:
                                st.info("No +1 frame stop codons found.")

                        with col2:
                            # -1 Stop codon distribution pie chart
                            if result['minus1_Total_Stops'] > 0:
                                st.markdown("#### 🥧 -1 Frame Stop Codons")
                                pie_data_minus1 = []
                                pie_labels_minus1 = []
                                if result['minus1_TAA_Count'] > 0:
                                    pie_data_minus1.append(result['minus1_TAA_Count'])
                                    pie_labels_minus1.append('TAA')
                                if result['minus1_TAG_Count'] > 0:
                                    pie_data_minus1.append(result['minus1_TAG_Count'])
                                    pie_labels_minus1.append('TAG')
                                if result['minus1_TGA_Count'] > 0:
                                    pie_data_minus1.append(result['minus1_TGA_Count'])
                                    pie_labels_minus1.append('TGA')

                                fig_pie_minus1 = create_interactive_pie_chart(pie_data_minus1, pie_labels_minus1, "-1 Frame Stop Codon Distribution", show_percentages=False)
                                st.plotly_chart(fig_pie_minus1, use_container_width=True, key="single_minus1_pie_chart")
                            else:
                                st.info("No -1 frame stop codons found.")
                        
                        with st.expander("View Summary Details"):
                            # +1 Stops
                            st.markdown("##### +1 Frame Stops")
                            plus1_data = {
                                'Codon': ['TAA', 'TAG', 'TGA', '**Total**'],
                                'Count': [result.get('Plus1_TAA_Count', 0), result.get('Plus1_TAG_Count', 0), result.get('Plus1_TGA_Count', 0), result.get('Plus1_Total_Stops', 0)]
                            }
                            plus1_df = pd.DataFrame(plus1_data)
                            st.dataframe(plus1_df, use_container_width=True)

                            # -1 Stops
                            st.markdown("##### -1 Frame Stops")
                            minus1_data = {
                                'Codon': ['TAA', 'TAG', 'TGA', '**Total**'],
                                'Count': [result.get('minus1_TAA_Count', 0), result.get('minus1_TAG_Count', 0), result.get('minus1_TGA_Count', 0), result.get('minus1_Total_Stops', 0)]
                            }
                            minus1_df = pd.DataFrame(minus1_data)
                            st.dataframe(minus1_df, use_container_width=True)

                            # Slippery Motifs
                            st.markdown("##### Slippery Motifs")
                            st.metric(label="Total Count", value=result.get('Slippery_Motifs', 0))

                        st.divider()
                        # Add the new graphs for single sequence analysis (matching batch analysis)
                        
                        st.subheader("📊 Interactive CAI and Stop Codon Analysis")

                        with st.expander("🧬 Interactive CAI and Stop Codon Charts", expanded=False):
                            
                            cai_result, cai_error = run_single_optimization(sequence_input, "In-Frame Analysis")
                            if not cai_error and isinstance(cai_result, dict) and 'Position' in cai_result:
                                cai_df = pd.DataFrame(cai_result)
                                positions = cai_df['Position'].tolist()
                                cai_weights = cai_df['CAI_Weight'].tolist()
                                amino_acids = cai_df['Amino_Acid'].tolist()

                                # Get stop codon positions
                                plus1_stop_positions = get_plus1_stop_positions(sequence_input)
                                minus1_stop_positions = get_minus1_stop_positions(sequence_input)

                                # Create +1 stop codon plot
                                if plus1_stop_positions:
                                    fig_plus1 = create_interactive_cai_stop_codon_plot(
                                        positions,
                                        cai_weights,
                                        amino_acids,
                                        plus1_stop_positions,
                                        f"Sequence ({len(sequence_input)} bp)",
                                        "+1 Frame"
                                    )
                                    st.plotly_chart(fig_plus1, use_container_width=True, key="single_plus1_cai_stop_plot")
                                else:
                                    st.info("No +1 stop codons found to plot against CAI.")

                                # Create -1 stop codon plot
                                if minus1_stop_positions:
                                    fig_minus1 = create_interactive_cai_stop_codon_plot(
                                        positions,
                                        cai_weights,
                                        amino_acids,
                                        minus1_stop_positions,
                                        f"Sequence ({len(sequence_input)} bp)",
                                        "-1 Frame"
                                    )
                                    st.plotly_chart(fig_minus1, use_container_width=True, key="single_minus1_cai_stop_plot")
                                else:
                                    st.info("No -1 stop codons found to plot against CAI.")
                            else:
                                st.warning("Could not generate CAI data for stop codon plots.")


                        st.divider()
                        
                        # IMMUNOGENIC PEPTIDE SCANNING - NEW SECTION
                        if not epitope_df.empty:
                            st.subheader("🔬 Immunogenic Peptide Scanning")
                            
                            # Translate +1 and -1 frames
                            plus1_protein = translate_frame(sequence_input, 1)  # +1 frame
                            minus1_protein = translate_frame(sequence_input, 2)  # -1 frame (offset by 2 to get -1)
                            
                            # Scan for immunogenic peptides
                            plus1_findings = scan_for_immunogenic_peptides(plus1_protein, epitope_df, "+1 Frame")
                            minus1_findings = scan_for_immunogenic_peptides(minus1_protein, epitope_df, "-1 Frame")
                            
                            total_findings = len(plus1_findings) + len(minus1_findings)
                            
                            # Display summary metrics
                            scan_col1, scan_col2, scan_col3, scan_col4 = st.columns(4)
                            with scan_col1:
                                st.metric("Epitopes in +1 Frame", len(plus1_findings))
                            with scan_col2:
                                st.metric("Epitopes in -1 Frame", len(minus1_findings))
                            with scan_col3:
                                st.metric("Total Epitopes Found", total_findings)
                            with scan_col4:
                                st.metric("Epitopes in Database", len(epitope_df))
                            
                            if total_findings > 0:
                                st.warning(f"⚠️ **WHOOPSIE**: Found {total_findings} immunogenic peptides in alternative reading frames!")
                                
                                # Create detailed summary
                                summary_df = create_immunogenic_peptide_summary(plus1_findings, minus1_findings)
                                if summary_df is not None:
                                    st.subheader("📋 Detailed Epitope Findings")
                                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                    
                                    # Show frame-specific details
                                    if plus1_findings:
                                        with st.expander(f"🔍 +1 Frame Epitopes ({len(plus1_findings)} found)", expanded=True):
                                            for i, finding in enumerate(plus1_findings, 1):
                                                st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}'")

                                    if minus1_findings:
                                        with st.expander(f"🔍 -1 Frame Epitopes ({len(minus1_findings)} found)", expanded=True):
                                            for i, finding in enumerate(minus1_findings, 1):
                                                st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}'")
                                    
                                    # Download button for epitope findings
                                    if summary_df is not None:
                                        excel_data = create_download_link(summary_df, f"Immunogenic_Peptides_Found_{len(summary_df)}.xlsx")
                                        st.download_button(
                                            label="📥 Download Epitope Findings (Excel)",
                                            data=excel_data,
                                            file_name=f"Immunogenic_Peptides_Found_{len(summary_df)}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            help="Download complete list of found immunogenic peptides"
                                        )
                            else:
                                st.success("✅ **Good news**: No known immunogenic peptides found in +1 or -1 reading frames!")
                        
                        else:
                            st.info("ℹ️ Immunogenic peptide scanning disabled - epitope_table_export.xlsx not found")
                        
                    else:
                        # Standard optimization results
                        # Define the definitive optimized sequence FIRST to ensure consistency
                        if method_to_run == "MaxStop" and 'maxstop_result_seq' in st.session_state:
                            optimized_seq = st.session_state['maxstop_result_seq']
                        else:
                            optimized_seq = result.get('Optimized_DNA', sequence_input)

                        st.subheader("Optimization Results")
                        
                        # Show sequence comparison for optimization methods using full width
                        if 'Optimized_DNA' in result:
                            st.subheader("Sequence Comparison")
                            
                            seq_col1, seq_col2 = st.columns(2)
                            
                            with seq_col1:
                                display_copyable_sequence(result['Original_DNA'], "Original Sequence", "orig")
                            with seq_col2:
                                display_copyable_sequence(optimized_seq, "Optimized Sequence", "opt") # Use the consistent variable
                            
                        else:
                            result_data = []
                            for key, value in result.items():
                                if key != 'Method':
                                    result_data.append({'Field': key.replace('_', ' ').title(), 'Value': str(value)})
                            
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df, use_container_width=True)

                        # +1 Frame Stop Codon Distribution
                        with st.expander("🥧 +1 Frame Stop Codon Distribution", expanded=False):
                            pie_data_plus1 = []
                            pie_labels_plus1 = []
                            plus1_stops = number_of_plus1_stops(optimized_seq)
                            for codon in ['TAA', 'TAG', 'TGA']:
                                count = plus1_stops[codon]
                                if count > 0:
                                    pie_data_plus1.append(count)
                                    pie_labels_plus1.append(codon)
                            if pie_data_plus1:
                                fig_pie_plus1 = create_interactive_pie_chart(pie_data_plus1, pie_labels_plus1, "+1 Frame Stop Codon Distribution", show_percentages=False)
                                st.plotly_chart(fig_pie_plus1, use_container_width=True, key="single_plus1_pie_chart")
                            else:
                                st.info("No +1 frame stop codons found or data not available for this method.")

                        # -1 Frame Stop Codon Distribution
                        with st.expander("🥧 -1 Frame Stop Codon Distribution", expanded=False):
                            pie_data_minus1 = []
                            pie_labels_minus1 = []
                            minus1_stops = number_of_minus1_stops(optimized_seq)
                            for codon in ['TAA', 'TAG', 'TGA']:
                                count = minus1_stops[codon]
                                if count > 0:
                                    pie_data_minus1.append(count)
                                    pie_labels_minus1.append(codon)
                            if pie_data_minus1:
                                fig_pie_minus1 = create_interactive_pie_chart(pie_data_minus1, pie_labels_minus1, "-1 Frame Stop Codon Distribution", show_percentages=False)
                                st.plotly_chart(fig_pie_minus1, use_container_width=True, key="single_minus1_pie_chart")
                            else:
                                st.info("No -1 frame stop codons found or data not available for this method.")

                        # Stops and Slippery Motifs per 100bp
                        with st.expander("📊 Stops and Slippery Motifs per 100bp", expanded=False):
                            sequence_length = len(optimized_seq.replace('\n', '').replace(' ', ''))                            
                            plus1_stops = number_of_plus1_stops(optimized_seq)
                            stops_per_100bp = {
                                'TAA': [(plus1_stops['TAA'] / sequence_length) * 100 if sequence_length > 0 else 0],
                                'TAG': [(plus1_stops['TAG'] / sequence_length) * 100 if sequence_length > 0 else 0],
                                'TGA': [(plus1_stops['TGA'] / sequence_length) * 100 if sequence_length > 0 else 0],
                            }
                            stops_fig = create_interactive_stacked_bar_chart(
                                ['Optimized Sequence'],
                                stops_per_100bp,
                                '+1 Frame Stops per 100bp by Type',
                                '+1 Frame Stops per 100bp'
                            )
                            st.plotly_chart(stops_fig, use_container_width=True, key="single_stops_per_100bp_opt")

                            sequence_length = len(optimized_seq.replace('\n', '').replace(' ', ''))
                            slippery_counts = count_specific_slippery_motifs(optimized_seq)
                            slippery_per_100bp = {
                                'TTTT': [ (slippery_counts['TTTT'] / sequence_length) * 100 if sequence_length > 0 else 0 ],
                                'TTTC': [ (slippery_counts['TTTC'] / sequence_length) * 100 if sequence_length > 0 else 0 ],
                            }

                            slippery_fig = create_interactive_stacked_bar_chart(
                                ['Optimized Sequence'],
                                slippery_per_100bp,
                                'Slippery Sites per 100bp by Type',
                                'Slippery Sites per 100bp'
                            )
                            st.plotly_chart(slippery_fig, use_container_width=True, key="single_slippery_per_100bp_opt")

                            

                            # CAI and stop codon analysis charts
                            st.subheader("📊 Interactive CAI and Stop Codon Analysis")
                        with st.expander("🧬 Interactive CAI and Stop Codon Charts", expanded=False):
                            cai_result, cai_error = run_single_optimization(optimized_seq, "In-Frame Analysis")
                            if not cai_error and isinstance(cai_result, dict) and 'Position' in cai_result:
                                cai_df = pd.DataFrame(cai_result)
                                positions = cai_df['Position'].tolist()
                                cai_weights = cai_df['CAI_Weight'].tolist()
                                amino_acids = cai_df['Amino_Acid'].tolist()

                                plus1_stop_positions = get_plus1_stop_positions(optimized_seq)
                                minus1_stop_positions = get_minus1_stop_positions(optimized_seq)
                                slippery_positions = get_slippery_motif_positions(optimized_seq)

                                if plus1_stop_positions:
                                    fig_plus1 = create_interactive_cai_stop_codon_plot(
                                        positions, cai_weights, amino_acids, plus1_stop_positions,
                                        f"Optimized Sequence ({len(optimized_seq)} bp)", "+1 Frame"
                                    )
                                    st.plotly_chart(fig_plus1, use_container_width=True, key="single_plus1_cai_stop_plot_opt")
                                else:
                                    st.info("No +1 stop codons found to plot against CAI.")

                                if minus1_stop_positions:
                                    fig_minus1 = create_interactive_cai_stop_codon_plot(
                                        positions, cai_weights, amino_acids, minus1_stop_positions,
                                        f"Optimized Sequence ({len(optimized_seq)} bp)", "-1 Frame"
                                    )
                                    st.plotly_chart(fig_minus1, use_container_width=True, key="single_minus1_cai_stop_plot_opt")
                                else:
                                    st.info("No -1 stop codons found to plot against CAI.")

                                if slippery_positions:
                                    fig_slippery = create_interactive_cai_slippery_plot(
                                        positions, cai_weights, amino_acids, slippery_positions,
                                        f"Optimized Sequence ({len(optimized_seq)} bp)"
                                    )
                                    st.plotly_chart(fig_slippery, use_container_width=True, key="single_slippery_cai_plot_opt")
                                else:
                                    st.info("No slippery motifs found to plot against CAI.")
                            else:
                                st.warning("Could not generate CAI data for stop codon/slippery motif plots.")

                    
                    # Accumulation option
                    if accumulate_results:
                        st.session_state.run_counter += 1
                        result_with_id = result.copy()
                        result_with_id['Run_ID'] = st.session_state.run_counter
                        st.session_state.accumulated_results.append(result_with_id)
                        st.info(f"Result added to accumulation buffer (Total: {len(st.session_state.accumulated_results)})")
        
        # Display accumulated results if any exist
        if st.session_state.accumulated_results:
            st.divider()
            st.subheader("📚 Accumulated Results")
            
            with st.expander(f"View Accumulated Results ({len(st.session_state.accumulated_results)} total)", expanded=False):
                # Convert accumulated results to DataFrame
                acc_df = pd.DataFrame(st.session_state.accumulated_results)
                
                # Reorder columns
                if 'Run_ID' in acc_df.columns:
                    cols = ['Run_ID'] + [col for col in acc_df.columns if col != 'Run_ID']
                    acc_df = acc_df[cols]
                
                st.dataframe(acc_df, use_container_width=True)
                
                # Download accumulated results
                excel_data = create_download_link(acc_df, f"Accumulated_Results_{len(st.session_state.accumulated_results)}_runs.xlsx")
                st.download_button(
                    label="Download Accumulated Results (Excel)",
                    data=excel_data,
                    file_name=f"Accumulated_Results_{len(st.session_state.accumulated_results)}_runs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                
    with tab2:
        st.header("Batch Optimization")
        st.markdown("Upload multiple sequences for batch optimization")
        
        batch_file = st.file_uploader(
            "Upload Sequence File",
            type=['txt', 'fasta', 'fa'],
            help="Upload a text file with sequences or FASTA format file"
        )
        
        if batch_file is not None:
            try:
                # Process uploaded file
                content = batch_file.read()
                
                # Handle different content types
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                elif content is None:
                    st.error("Failed to read file content")
                    st.stop()
                
                sequences = []
                
                if content.strip().startswith('>'):
                    # FASTA format
                    lines = content.strip().splitlines()
                    current_seq, current_name = "", ""
                    for line in lines:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append((current_name, current_seq))
                            current_name, current_seq = line[1:].strip(), ""
                        else:
                            current_seq += line.upper()
                    if current_seq:
                        sequences.append((current_name, current_seq))
                else:
                    # Text format - one sequence per line
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    for i, line in enumerate(lines):
                        sequences.append((f"Sequence_{i+1}", line.upper()))
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                sequences = []
            
            if sequences:
                st.success(f"Loaded {len(sequences)} sequences")
                
                batch_operation_type = st.radio(
                    "Choose Operation Type",
                    ("Analysis", "Optimization"),
                    key="batch_op_type"
                )

                if batch_operation_type == "Analysis":
                    batch_method = st.selectbox(
                        "Batch Analysis Method",
                        [
                            "In-Frame Analysis",
                            "+1 Frame Analysis",
                        ],
                        help="Choose the analysis algorithm to apply to the batch"
                    )
                else:  # Optimization
                    batch_method = st.selectbox(
                        "Batch Optimization Method",
                        [
                            "Standard Codon Optimization",
                            "MaxStop",
                            "Balanced Optimization",
                        ],
                        help="Choose the optimization algorithm to apply to the batch"
                    )

                if st.button(f"Process Batch with {batch_operation_type}", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    for i, (name, seq) in enumerate(sequences):
                        status_text.text(f"Processing {name}...")
                        progress_bar.progress((i + 1) / len(sequences))
                        
                        result, error = run_single_optimization(seq, batch_method, bias_weight)
                        if error:
                            results.append({'Sequence_Name': name, 'Error': error})
                        else:
                            result_with_name = result.copy()
                            result_with_name['Sequence_Name'] = name
                            
                            # For MaxStop, overwrite the Optimized_DNA with the guaranteed correct one from session_state
                            if batch_method == "MaxStop" and 'maxstop_result_seq' in st.session_state:
                                result_with_name['Optimized_DNA'] = st.session_state['maxstop_result_seq']

                            results.append(result_with_name)
                    
                    status_text.text("I'M DONE! Processing complete.")
                    
                    if results:
                        # Convert to DataFrame
                        batch_df = pd.DataFrame(results)
                        
                        # Reorder columns to put Sequence_Name first
                        cols = ['Sequence_Name'] + [col for col in batch_df.columns if col != 'Sequence_Name']
                        batch_df = batch_df[cols]
                        
                        # In-Frame Analysis - Individual Interactive Charts for Each Sequence
                        if batch_method == "In-Frame Analysis" and not batch_df.empty:
                            st.subheader("📊 Interactive Individual In-Frame Analysis")
                            
                            # Create a unique key for this batch session
                            # Create a unique key for this batch session that includes the codon usage
                            current_organism = st.session_state.get('codon_file_source', 'Unknown')
                            batch_key = f"batch_{len(sequences)}_{hash(str([name for name, _ in sequences]))}_{hash(current_organism)}"
                            cai_data_key = f'batch_cai_data_{batch_key}'

                                # Initialize cai_sequences
                            cai_sequences = []

                                # Process sequences if not already cached OR if organism changed
                            if cai_data_key not in st.session_state:
                                with st.spinner("Processing In-Frame data for all sequences..."):
                                    st.session_state[cai_data_key] = []
                                    
                                    progress_cai = st.progress(0)
                                    status_cai = st.empty()
                                    
                                    for i, (name, seq) in enumerate(sequences):
                                        status_cai.text(f"Processing {name}... ({i+1}/{len(sequences)})")
                                        try:
                                            result, error = run_single_optimization(seq, batch_method, bias_weight)
                                            if not error and isinstance(result, dict) and 'Position' in result:
                                                st.session_state[cai_data_key].append({
                                                    'name': name,
                                                    'sequence': seq,
                                                    'cai_data': pd.DataFrame(result)
                                                })
                                            progress_cai.progress((i + 1) / len(sequences))
                                        except Exception as e:
                                            continue
                                    
                                    # Clear progress indicators after processing is complete
                                    progress_cai.empty()
                                    status_cai.empty()
                            
                            # Get the processed sequences from session state
                            cai_sequences = st.session_state.get(cai_data_key, [])
                            
                            # Display results
                            if cai_sequences:
                                # Display all In-Frame interactive graphs
                                colors = get_consistent_color_palette(len(cai_sequences), "analysis")
                                for i, selected_data in enumerate(cai_sequences):
                                    df = selected_data['cai_data']
                                    seq_name = selected_data['name']
                                    seq_sequence = selected_data['sequence']
                                    
                                    st.markdown(f"### 📊 Interactive In-Frame Analysis for: {seq_name}")
                                    
                                    if not df.empty and 'CAI_Weight' in df.columns:
                                        positions = df['Position'].tolist()
                                        cai_weights = df['CAI_Weight'].tolist()
                                        amino_acids = df['Amino_Acid'].tolist()
                                        
                                        plus1_stop_positions = get_plus1_stop_positions(seq_sequence)
                                        minus1_stop_positions = get_minus1_stop_positions(seq_sequence)
                                        slippery_positions = get_slippery_motif_positions(seq_sequence)
                                        
                                        # Use different color for each sequence
                                        cai_color = colors[i % len(colors)]
                                        
                                        display_stateful_overlay_chart(
                                            positions=positions,
                                            cai_weights=cai_weights,
                                            amino_acids=amino_acids,
                                            sequence=seq_sequence,
                                            seq_name=seq_name,
                                            plus1_stop_positions=plus1_stop_positions,
                                            minus1_stop_positions=minus1_stop_positions,
                                            slippery_positions=slippery_positions,
                                            cai_color=cai_color  # Add this parameter
                                        )
                                        
                                        # Statistics including GC content
                                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                        with col_stat1:
                                            st.metric("Average CAI", f"{np.mean(cai_weights):.3f}")
                                        with col_stat2:
                                            st.metric("Sequence Length", f"{len(seq_sequence)} bp")
                                        with col_stat3:
                                            # Calculate GC content from the original sequence
                                            gc_content = calculate_gc_content(seq_sequence)
                                            st.metric("GC Content", f"{gc_content:.1f}%")
                                        with col_stat4:
                                            slippery_motifs = number_of_slippery_motifs(seq_sequence)
                                            st.metric("Slippery Motifs", slippery_motifs)

                                

                                        # Add slippery motif location analysis
                                        st.subheader("📍 Slippery Motif Locations")

                                        slippery_positions = get_slippery_motif_positions(seq_sequence)

                                        if slippery_positions:
                                            

                                            # Show detailed table of slippery motif positions
                                            st.markdown("#### 📋 Detailed Slippery Motif Positions")
                                            slippery_df = pd.DataFrame(slippery_positions)
                                            slippery_df.columns = ['Motif', 'Nucleotide Position', 'Amino Acid Position', 'Codon Range']

                                            st.dataframe(slippery_df, use_container_width=True, hide_index=True)

                                            # Download slippery motif data
                                            excel_data = create_download_link(slippery_df, f"Slippery_Motifs_{len(slippery_positions)}_found.xlsx")

                                            st.download_button(
                                                label="📥 Download Slippery Motif Positions (Excel)",
                                                data=excel_data,
                                                file_name=f"Slippery_Motifs_{len(slippery_positions)}_found.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                help="Download detailed positions of slippery motifs",
                                                key=f"download_slippery_{i}"
                                            )
                                        else:
                                            st.info("✅ No slippery motifs (TTTT or TTTC) found in the coding sequence.")

                                        
                                        # Data table in expandable section
                                        with st.expander(f"📋 View detailed In-Frame data for {seq_name}"):
                                            st.dataframe(df, use_container_width=True)
                                        
                                        st.divider()  # Add separator between sequences
                                    else:
                                        st.warning(f"No In-Frame data available for {seq_name}")
                            else:
                                st.warning("No valid In-Frame data found for any sequences")

                        # +1 Frame Analysis visualization with interactive charts
                        elif batch_method == "+1 Frame Analysis" and not batch_df.empty:
                            st.subheader("📊 Interactive Batch +1 Frame Analysis")
                            
                            # Check if we have the required columns and valid data
                            required_cols = ['Plus1_TAA_Count', 'Plus1_TAG_Count', 'Plus1_TGA_Count']
                            gc_available = 'GC_Content' in batch_df.columns
                            
                            if all(col in batch_df.columns for col in required_cols):
                                
                                # Overall statistics first
                                total_taa = batch_df['Plus1_TAA_Count'].sum()
                                total_tag = batch_df['Plus1_TAG_Count'].sum()
                                total_tga = batch_df['Plus1_TGA_Count'].sum()
                                total_stops = total_taa + total_tag + total_tga
                                
                                # Summary statistics
                                st.markdown("#### 📈 Overall Statistics")
                                avg_gc = batch_df['GC_Content'].mean()
                                avg_len = batch_df['Sequence_Length'].mean()
                                avg_prot_len = batch_df['Protein_Length'].mean()
                                total_plus1_stops = batch_df['Plus1_Total_Stops'].sum()
                                total_minus1_stops = batch_df['minus1_Total_Stops'].sum()
                                total_slippery = batch_df['Slippery_Motifs'].sum()

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Sequences", len(sequences))
                                    
                                with col2:
                                    
                                    st.metric("Total +1 Stops", total_plus1_stops)
                                    
                                with col3:
                                    
                                    st.metric("Total -1 Stops", total_minus1_stops)
                                    
                                with col4: 
                                    st.metric("Total Slippery Motifs", total_slippery)
                                
                                
                                # Individual sequence pie charts
                                if total_stops > 0:
                                    st.markdown("#### 🥧 Individual Sequence Stop Codon Distribution")
                                    
                                    # Create pie charts for each sequence that has stops
                                    sequences_with_stops_data = batch_df[batch_df['Plus1_Total_Stops'] > 0]
                                    
                                    if not sequences_with_stops_data.empty:
                                        # Create columns for pie charts (2 per row)
                                        cols_per_row = 2
                                        num_sequences = len(sequences_with_stops_data)
                                        
                                        for i in range(0, num_sequences, cols_per_row):
                                            cols = st.columns(cols_per_row)
                                            
                                            for j in range(cols_per_row):
                                                if i + j < num_sequences:
                                                    seq_data = sequences_with_stops_data.iloc[i + j]
                                                    seq_name = seq_data['Sequence_Name']
                                                    
                                                    taa_count = seq_data['Plus1_TAA_Count']
                                                    tag_count = seq_data['Plus1_TAG_Count']
                                                    tga_count = seq_data['Plus1_TGA_Count']
                                                    total_seq_stops = seq_data['Plus1_Total_Stops']
                                                    
                                                    if total_seq_stops > 0:
                                                        with cols[j]:
                                                            # Filter out zero values
                                                            pie_data = []
                                                            pie_labels = []
                                                            pie_colors = []
                                                            color_map = {'TAA': '#FF6B6B', 'TAG': '#4ECDC4', 'TGA': '#45B7D1'}
                                                            
                                                            for codon, count in [('TAA', taa_count), ('TAG', tag_count), ('TGA', tga_count)]:
                                                                if count > 0:
                                                                    pie_data.append(count)
                                                                    pie_labels.append(codon)
                                                                    pie_colors.append(color_map[codon])
                                                            
                                                            if pie_data:
                                                                fig_individual = go.Figure(data=[go.Pie(
                                                                    labels=pie_labels,
                                                                    values=pie_data,
                                                                    hole=.4,
                                                                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                                                                    textinfo='label+value',
                                                                    textfont_size=10,
                                                                    marker=dict(
                                                                        colors=pie_colors,
                                                                        line=dict(color='#FFFFFF', width=2)
                                                                    )
                                                                )])
                                                                
                                                                fig_individual.update_layout(
                                                                    title={
                                                                        'text': f'{seq_name[:20]}{"..." if len(seq_name) > 20 else ""}<br><sub>+1 Frame Stops</sub>',
                                                                        'x': 0.5,
                                                                        'font': {'size': 12}
                                                                    },
                                                                    annotations=[dict(
                                                                        text=f'{total_seq_stops}<br>Stops', 
                                                                        x=0.5, y=0.5, 
                                                                        font_size=11, 
                                                                        showarrow=False,
                                                                        font=dict(color="#2C3E50", weight="bold")
                                                                    )],
                                                                    height=300,
                                                                    showlegend=False,
                                                                    margin=dict(t=50, b=10, l=10, r=10)
                                                                )
                                                                
                                                                st.plotly_chart(fig_individual, use_container_width=True)
                                    else:
                                        st.info("No sequences with +1 frame stops found for individual visualization.")
                                
                                # BATCH IMMUNOGENIC PEPTIDE SCANNING - NEW SECTION
                                    epitope_df = load_immunogenic_peptides()
                                    
                                    if not epitope_df.empty:
                                        st.subheader("🔬 Batch Immunogenic Peptide Scanning")
                                        
                                        batch_epitope_findings = []
                                        sequences_with_epitopes = 0
                                        total_epitopes_found = 0
                                        
                                        with st.spinner("Scanning all sequences for immunogenic peptides..."):
                                            progress_epitope = st.progress(0)
                                            status_epitope = st.empty()
                                            
                                            for i, (name, seq) in enumerate(sequences):
                                                status_epitope.text(f"Scanning {name} for epitopes... ({i+1}/{len(sequences)})")
                                                
                                                # Translate +1 and -1 frames
                                                plus1_protein = translate_frame(seq, 1)
                                                minus1_protein = translate_frame(seq, 2)
                                                
                                                # Scan for immunogenic peptides
                                                plus1_findings = scan_for_immunogenic_peptides(plus1_protein, epitope_df, "+1 Frame")
                                                minus1_findings = scan_for_immunogenic_peptides(minus1_protein, epitope_df, "-1 Frame")
                                                
                                                # Record findings for this sequence
                                                if plus1_findings or minus1_findings:
                                                    sequences_with_epitopes += 1
                                                    
                                                    for finding in plus1_findings + minus1_findings:
                                                        finding['sequence_name'] = name
                                                        finding['sequence_length'] = len(seq)
                                                        batch_epitope_findings.append(finding)
                                                        total_epitopes_found += 1
                                                
                                                progress_epitope.progress((i + 1) / len(sequences))
                                            
                                            # Clear progress indicators
                                            progress_epitope.empty()
                                            status_epitope.empty()
                                        
                                        # Display batch epitope scanning results
                                        scan_col1, scan_col2, scan_col3, scan_col4 = st.columns(4)
                                        with scan_col1:
                                            st.metric("Sequences Scanned", len(sequences))
                                        with scan_col2:
                                            st.metric("Sequences with Epitopes", sequences_with_epitopes)
                                        with scan_col3:
                                            st.metric("Total Epitopes Found", total_epitopes_found)
                                        with scan_col4:
                                            epitope_rate = (sequences_with_epitopes / len(sequences) * 100) if len(sequences) > 0 else 0
                                            st.metric("Epitope Rate", f"{epitope_rate:.1f}%")
                                        
                                        if total_epitopes_found > 0:
                                            st.warning(f"⚠️ **What the Heck**: Found {total_epitopes_found} immunogenic peptides across {sequences_with_epitopes} sequences!")

                                            # Create batch epitope summary
                                            if batch_epitope_findings:
                                                batch_epitope_df = pd.DataFrame(batch_epitope_findings)
                                                
                                                # Reorder columns for better display
                                                priority_cols = ['sequence_name', 'frame', 'epitope', 'position', 'end_position', 'length']
                                                other_cols = [col for col in batch_epitope_df.columns if col not in priority_cols]
                                                batch_epitope_df = batch_epitope_df[priority_cols + other_cols]
                                                
                                                st.subheader("📋 Batch Epitope Findings Summary")
                                                st.dataframe(batch_epitope_df, use_container_width=True, hide_index=True)
                                                
                                                # Summary by sequence
                                                st.subheader("📊 Epitope Summary by Sequence")
                                                
                                                epitope_summary = batch_epitope_df.groupby(['sequence_name', 'frame']).size().reset_index(name='epitope_count')
                                                epitope_pivot = epitope_summary.pivot(index='sequence_name', columns='frame', values='epitope_count').fillna(0).astype(int)
                                                
                                                if '+1 Frame' not in epitope_pivot.columns:
                                                    epitope_pivot['+1 Frame'] = 0
                                                if '-1 Frame' not in epitope_pivot.columns:
                                                    epitope_pivot['-1 Frame'] = 0
                                                
                                                epitope_pivot['Total'] = epitope_pivot['+1 Frame'] + epitope_pivot['-1 Frame']
                                                epitope_pivot = epitope_pivot.sort_values('Total', ascending=False)
                                                
                                                st.dataframe(epitope_pivot, use_container_width=True)
                                                
                                                
                                                
                                                # Download batch epitope findings
                                                excel_data = create_download_link(batch_epitope_df, f"Batch_Immunogenic_Peptides_{total_epitopes_found}_epitopes.xlsx")
                                                st.download_button(
                                                    label="📥 Download Batch Epitope Findings (Excel)",
                                                    data=excel_data,
                                                    file_name=f"Batch_Immunogenic_Peptides_{total_epitopes_found}_epitopes.xlsx",
                                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                    help="Download complete list of immunogenic peptides found across all sequences"
                                                )
                                        else:
                                            st.success("✅ **Excellent**: No known immunogenic peptides found in any sequence!")
                                    
                                    else:
                                        st.info("ℹ️ Batch immunogenic peptide scanning disabled - epitope_table_export.xlsx not found")
                                
                                if total_stops > 0:
                                    # Interactive summary charts with breakdown by stop codon type
                                    st.markdown("#### 📊 Interactive Summary Charts")

                                    # Chart 1: +1 Stops per 100bp broken down by TAA, TAG, TGA
                                    sequence_names = batch_df['Sequence_Name'].tolist()
                                    sequence_lengths = batch_df['Sequence_Length'].tolist() if 'Sequence_Length' in batch_df.columns else [1] * len(sequence_names)
                                    
                                    # Calculate stops per 100bp for each type
                                    taa_per_100bp = [(batch_df.iloc[i]['Plus1_TAA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                    tag_per_100bp = [(batch_df.iloc[i]['Plus1_TAG_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                    tga_per_100bp = [(batch_df.iloc[i]['Plus1_TGA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]

                                    # Create interactive stacked bar chart
                                    stops_data = {
                                        'TAA': taa_per_100bp,
                                        'TAG': tag_per_100bp,
                                        'TGA': tga_per_100bp
                                    }
                                    
                                    stops_fig = create_interactive_stacked_bar_chart(
                                        sequence_names,
                                        stops_data,
                                        '+1 Frame Stops per 100bp by Type',
                                        '+1 Frame Stops per 100bp'
                                    )
                                    st.plotly_chart(stops_fig, use_container_width=True)

                                    # Chart 2: Slippery Sites per 100bp broken down by TTTT and TTTC
                                    tttt_counts = []
                                    tttc_counts = []

                                    # Calculate specific slippery motifs for each sequence
                                    for i, (name, seq) in enumerate(sequences):
                                        slippery_breakdown = count_specific_slippery_motifs(seq)
                                        seq_length = len(seq) if len(seq) > 0 else 1
                                        tttt_per_100bp = (slippery_breakdown['TTTT'] / seq_length) * 100
                                        tttc_per_100bp = (slippery_breakdown['TTTC'] / seq_length) * 100
                                        tttt_counts.append(tttt_per_100bp)
                                        tttc_counts.append(tttc_per_100bp)

                                    # Create interactive stacked bar chart for slippery motifs
                                    slippery_data = {
                                        'TTTT': tttt_counts,
                                        'TTTC': tttc_counts
                                    }
                                    
                                    slippery_fig = create_interactive_stacked_bar_chart(
                                        sequence_names,
                                        slippery_data,
                                        'Slippery Sites per 100bp by Type',
                                        'Slippery Sites per 100bp'
                                    )
                                    st.plotly_chart(slippery_fig, use_container_width=True, key="batch_slippery_fig")

                                else:
                                    st.info("No +1 frame stop codons found in any sequence.")

                                # -1 Frame Analysis visualization
                                st.subheader("📊 Interactive Batch -1 Frame Analysis")

                                # Check if we have the required columns and valid data
                                required_cols = ['minus1_TAA_Count', 'minus1_TAG_Count', 'minus1_TGA_Count', 'minus1_Total_Stops']
                                if all(col in batch_df.columns for col in required_cols):

                                    # Overall statistics first
                                    total_taa = batch_df['minus1_TAA_Count'].sum()
                                    total_tag = batch_df['minus1_TAG_Count'].sum()
                                    total_tga = batch_df['minus1_TGA_Count'].sum()
                                    total_stops = total_taa + total_tag + total_tga

                                    # Summary statistics
                                    st.markdown("#### 📈 Overall Statistics")
                                    if gc_available:
                                        avg_gc = batch_df['GC_Content'].mean()
                                        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                                        with col_stat2:
                                            st.metric("Avg GC Content", f"{avg_gc:.1f}%")
                                    else:
                                        col_stat1, col_stat3, col_stat4, col_stat5 = st.columns(4)

                                    with col_stat1:
                                        st.metric("Total Sequences", len(sequences))
                                    with col_stat3:
                                        st.metric("Total -1 Stops", total_stops)
                                    with col_stat4:
                                        avg_stops = total_stops / len(sequences) if len(sequences) > 0 else 0
                                        st.metric("Avg Stops/Seq", f"{avg_stops:.1f}")
                                    with col_stat5:
                                        sequences_with_stops = len(batch_df[batch_df['minus1_Total_Stops'] > 0])
                                        st.metric("Seqs with Stops", f"{sequences_with_stops}/{len(sequences)}")


                                    # Individual sequence pie charts
                                    if total_stops > 0:
                                        st.markdown("#### 🥧 Individual Sequence Stop Codon Distribution")

                                        # Create pie charts for each sequence that has stops
                                        sequences_with_stops_data = batch_df[batch_df['minus1_Total_Stops'] > 0]

                                        if not sequences_with_stops_data.empty:
                                            # Create columns for pie charts (2 per row)
                                            cols_per_row = 2
                                            num_sequences = len(sequences_with_stops_data)

                                            for i in range(0, num_sequences, cols_per_row):
                                                cols = st.columns(cols_per_row)

                                                for j in range(cols_per_row):
                                                    if i + j < num_sequences:
                                                        seq_data = sequences_with_stops_data.iloc[i + j]
                                                        seq_name = seq_data['Sequence_Name']

                                                        taa_count = seq_data['minus1_TAA_Count']
                                                        tag_count = seq_data['minus1_TAG_Count']
                                                        tga_count = seq_data['minus1_TGA_Count']
                                                        total_seq_stops = seq_data['minus1_Total_Stops']

                                                        if total_seq_stops > 0:
                                                            with cols[j]:
                                                                # Filter out zero values
                                                                pie_data = []
                                                                pie_labels = []
                                                                pie_colors = []
                                                                color_map = {'TAA': '#FF6B6B', 'TAG': '#4ECDC4', 'TGA': '#45B7D1'}

                                                                for codon, count in [('TAA', taa_count), ('TAG', tag_count), ('TGA', tga_count)]:
                                                                    if count > 0:
                                                                        pie_data.append(count)
                                                                        pie_labels.append(codon)
                                                                        pie_colors.append(color_map[codon])

                                                                if pie_data:
                                                                    fig_individual = go.Figure(data=[go.Pie(
                                                                        labels=pie_labels,
                                                                        values=pie_data,
                                                                        hole=.4,
                                                                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                                                                        textinfo='label+value',
                                                                        textfont_size=10,
                                                                        marker=dict(
                                                                            colors=pie_colors,
                                                                            line=dict(color='#FFFFFF', width=2)
                                                                        )
                                                                    )])

                                                                    fig_individual.update_layout(
                                                                        title={
                                                                            'text': f'{seq_name[:20]}{"..." if len(seq_name) > 20 else ""}<br><sub>-1 Frame Stops</sub>',
                                                                            'x': 0.5,
                                                                            'font': {'size': 12}
                                                                        },
                                                                        annotations=[dict(
                                                                            text=f'{total_seq_stops}<br>Stops', 
                                                                            x=0.5, y=0.5, 
                                                                            font_size=11, 
                                                                            showarrow=False,
                                                                            font=dict(color="#2C3E50", weight="bold")
                                                                        )],
                                                                        height=300,
                                                                        showlegend=False,
                                                                        margin=dict(t=50, b=10, l=10, r=10)
                                                                    )

                                                                    st.plotly_chart(fig_individual, use_container_width=True)
                                        else:
                                            st.info("No sequences with -1 frame stops found for individual visualization.")

                                    if total_stops > 0:
                                        # Interactive summary charts with breakdown by stop codon type
                                        st.markdown("#### 📊 Interactive Summary Charts")

                                        # Chart 1: -1 Stops per 100bp broken down by TAA, TAG, TGA
                                        sequence_names = batch_df['Sequence_Name'].tolist()
                                        sequence_lengths = batch_df['Sequence_Length'].tolist() if 'Sequence_Length' in batch_df.columns else [1] * len(sequence_names)

                                        # Calculate stops per 100bp for each type
                                        taa_per_100bp = [(batch_df.iloc[i]['minus1_TAA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                        tag_per_100bp = [(batch_df.iloc[i]['minus1_TAG_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]
                                        tga_per_100bp = [(batch_df.iloc[i]['minus1_TGA_Count'] / sequence_lengths[i]) * 100 if sequence_lengths[i] > 0 else 0 for i in range(len(sequence_names))]

                                        # Create interactive stacked bar chart
                                        stops_data = {
                                            'TAA': taa_per_100bp,
                                            'TAG': tag_per_100bp,
                                            'TGA': tga_per_100bp
                                        }

                                        stops_fig = create_interactive_stacked_bar_chart(
                                            sequence_names,
                                            stops_data,
                                            '-1 Frame Stops per 100bp by Type',
                                            '-1 Frame Stops per 100bp'
                                        )
                                        st.plotly_chart(stops_fig, use_container_width=True, key="batch_minus1_stops_fig")

                                    else:
                                        st.info("No -1 frame stop codons found in any sequence.")


                                else:
                                    st.warning("Analysis data not available for visualization.")

                                # Add the new graphs for batch analysis
                                st.subheader("📊 Interactive CAI and Stop Codon Analysis (Batch)")

                            for i, (name, seq) in enumerate(sequences):
                                with st.expander(f"🧬 Analysis for: {name}", expanded=False):
                                    
                                    # Get CAI data
                                    cai_result, cai_error = run_single_optimization(seq, "In-Frame Analysis")
                                    if not cai_error and isinstance(cai_result, dict) and 'Position' in cai_result:
                                        cai_df = pd.DataFrame(cai_result)
                                        positions = cai_df['Position'].tolist()
                                        cai_weights = cai_df['CAI_Weight'].tolist()
                                        amino_acids = cai_df['Amino_Acid'].tolist()

                                        # Get stop codon positions
                                        plus1_stop_positions = get_plus1_stop_positions(seq)
                                        minus1_stop_positions = get_minus1_stop_positions(seq)

                                        # Create +1 stop codon plot
                                        if plus1_stop_positions:
                                            fig_plus1 = create_interactive_cai_stop_codon_plot(
                                                positions,
                                                cai_weights,
                                                amino_acids,
                                                plus1_stop_positions,
                                                name,
                                                "+1 Frame"
                                            )
                                            st.plotly_chart(fig_plus1, use_container_width=True, key=f"batch_plus1_cai_stop_plot_{i}")
                                        else:
                                            st.info(f"No +1 stop codons found in {name} to plot against CAI.")

                                        # Create -1 stop codon plot
                                        if minus1_stop_positions:
                                            fig_minus1 = create_interactive_cai_stop_codon_plot(
                                                positions,
                                                cai_weights,
                                                amino_acids,
                                                minus1_stop_positions,
                                                name,
                                                "-1 Frame"
                                            )
                                            st.plotly_chart(fig_minus1, use_container_width=True, key=f"batch_minus1_cai_stop_plot_{i}")
                                        else:
                                            st.info(f"No -1 stop codons found in {name} to plot against CAI.")
                                        
                                        st.divider()

                                    else:
                                        st.warning(f"Could not generate CAI data for {name}.")
                            
                                
                                    
                    
                                
                        # Display results for other optimization methods with interactive charts
                        elif batch_method in ["Standard Codon Optimization", "Balanced Optimization", 
                                              "MaxStop"]:
                            st.subheader(f"📊 Interactive Batch {batch_method} Results")
                            
                            # Check if we have optimization results
                            if 'Optimized_DNA' in batch_df.columns:
                                # Summary statistics
                                st.markdown("#### 📈 Optimization Summary")
                                
                                total_sequences = len(batch_df)
                                successful_optimizations = len(batch_df[batch_df['Optimized_DNA'].notna()])
                                
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Total Sequences", total_sequences)
                                with col_stat2:
                                    st.metric("Successful Optimizations", successful_optimizations)
                                with col_stat3:
                                    success_rate = (successful_optimizations / total_sequences * 100) if total_sequences > 0 else 0
                                    st.metric("Success Rate", f"{success_rate:.1f}%")
                                
                                # Display individual sequence results
                                st.markdown("#### 🧬 Individual Sequence Results")
                                
                                for idx, row in batch_df.iterrows():
                                    seq_name = row.get('Sequence_Name', f'Sequence_{idx+1}')
                                    
                                    with st.expander(f"📄 {seq_name}", expanded=False):
                                        if pd.notna(row.get('Optimized_DNA')):
                                            # Show sequence comparison
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                display_copyable_sequence(
                                                    row.get('Original_DNA', ''), 
                                                    "Original Sequence", 
                                                    f"batch_orig_{idx}"
                                                )
                                            
                                            with col2:
                                                display_copyable_sequence(
                                                    row.get('Optimized_DNA', ''), 
                                                    "Optimized Sequence", 
                                                    f"batch_opt_{idx}"
                                                )
                                            
                                            # Show metrics
                                            st.markdown("**📊 Optimization Metrics:**")

                                            # Create three columns for metrics
                                            metric_col1, metric_col2, metric_col3 = st.columns(3)

                                            with metric_col1:
                                                st.markdown("**🧬 Sequence Properties**")
                                                orig_len = len(row.get('Original_DNA', ''))
                                                opt_len = len(row.get('Optimized_DNA', ''))
                                                st.metric("Sequence Length", f"{orig_len} bp", delta=f"{opt_len - orig_len} bp" if opt_len != orig_len else None)
                                                
                                                if 'Protein' in row:
                                                    st.metric("Protein Length", f"{len(row['Protein'])} aa")

                                            with metric_col2:
                                                st.markdown("**🧪 GC & CAI Analysis**")
                                                
                                                # GC Content
                                                orig_gc = calculate_gc_content(row.get('Original_DNA', ''))
                                                opt_gc = calculate_gc_content(row.get('Optimized_DNA', ''))
                                                gc_change = opt_gc - orig_gc
                                                st.metric(
                                                    "GC Content", 
                                                    f"{opt_gc:.1f}%", 
                                                    delta=f"{gc_change:+.1f}%",
                                                    delta_color="inverse"  # Red if too high, green if moderate
                                                )
                                                
                                                # CAI Analysis
                                                orig_seq = row.get('Original_DNA', '')
                                                opt_seq = row.get('Optimized_DNA', '')
                                                if orig_seq and opt_seq:
                                                    orig_weights, _ = get_codon_weights_row(orig_seq)
                                                    opt_weights, _ = get_codon_weights_row(opt_seq)
                                                    orig_cai = sum(orig_weights) / len(orig_weights) if orig_weights else 0
                                                    opt_cai = sum(opt_weights) / len(opt_weights) if opt_weights else 0
                                                    cai_change = opt_cai - orig_cai
                                                    
                                                    st.metric(
                                                        "CAI Score", 
                                                        f"{opt_cai:.3f}", 
                                                        delta=f"{cai_change:+.3f}",
                                                        delta_color="normal"  # Green is good for CAI
                                                    )

                                            with metric_col3:
                                                st.markdown("**🛑 Stop Codon Analysis**")
                                                
                                                # +1 Frame stops
                                                orig_stops = number_of_plus1_stops(row.get('Original_DNA', ''))
                                                opt_stops = number_of_plus1_stops(row.get('Optimized_DNA', ''))
                                                stops_change = opt_stops['total'] - orig_stops['total']
                                                
                                                st.metric(
                                                    "+1 Frame Stops", 
                                                    f"{opt_stops['total']}", 
                                                    delta=f"{stops_change:+d}",
                                                    delta_color="inverse"  # Red if increased, green if decreased
                                                )
                                                
                                                # Show stop codon breakdown if there are stops
                                                if opt_stops['total'] > 0:
                                                    st.caption(f"TAA: {opt_stops['TAA']}, TAG: {opt_stops['TAG']}, TGA: {opt_stops['TGA']}")
                                        
                                        else:
                                            if 'Error' in row and pd.notna(row['Error']):
                                                st.error(f"Error: {row['Error']}")
                                            else:
                                                st.warning("No optimization results available")
                                
                                # Interactive summary comparison charts
                                if successful_optimizations > 0:
                                    st.markdown("#### 📊 Interactive Optimization Impact Analysis")
                                    
                                    # Calculate metrics for all sequences
                                    metrics_data = []
                                    for idx, row in batch_df.iterrows():
                                        if pd.notna(row.get('Optimized_DNA')):
                                            orig_seq = row.get('Original_DNA', '')
                                            opt_seq = row.get('Optimized_DNA', '')
                                            
                                            if orig_seq and opt_seq:
                                                # Calculate all metrics
                                                orig_stops = number_of_plus1_stops(orig_seq)
                                                opt_stops = number_of_plus1_stops(opt_seq)
                                                
                                                # Calculate CAI
                                                orig_weights, _ = get_codon_weights_row(orig_seq)
                                                opt_weights, _ = get_codon_weights_row(opt_seq)
                                                orig_avg_cai = sum(orig_weights) / len(orig_weights) if orig_weights else 0
                                                opt_avg_cai = sum(opt_weights) / len(opt_weights) if opt_weights else 0
                                                
                                                metrics_data.append({
                                                    'Sequence': row.get('Sequence_Name', f'Seq_{idx+1}'),
                                                    'Original_Stops': orig_stops['total'],
                                                    'Optimized_Stops': opt_stops['total'],
                                                    'Stop_Change': opt_stops['total'] - orig_stops['total'],
                                                    'Original_GC': calculate_gc_content(orig_seq),
                                                    'Optimized_GC': calculate_gc_content(opt_seq),
                                                    'Original_CAI': orig_avg_cai,
                                                    'Optimized_CAI': opt_avg_cai,
                                                    'CAI_Change': opt_avg_cai - orig_avg_cai
                                                })
                                    
                                    if metrics_data:
                                        metrics_df = pd.DataFrame(metrics_data)
                                        
                                        # Create interactive comparison charts
                                        col_chart1, col_chart2 = st.columns(2)
                                        
                                        with col_chart1:
                                            # +1 Frame Stops Comparison
                                            stops_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_Stops'].tolist(),
                                                metrics_df['Optimized_Stops'].tolist(),
                                                '+1 Frame Stops',
                                                'Number of Stops'
                                            )
                                            st.plotly_chart(stops_comparison_fig, use_container_width=True, key="batch_stops_comparison_fig")
                                        
                                        with col_chart2:
                                            # GC Content Comparison
                                            gc_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_GC'].tolist(),
                                                metrics_df['Optimized_GC'].tolist(),
                                                'GC Content',
                                                'GC Content (%)'
                                            )
                                            st.plotly_chart(gc_comparison_fig, use_container_width=True, key="batch_gc_comparison_fig")
                                        
                                        # CAI Comparison
                                        if 'Original_CAI' in metrics_df.columns and 'Optimized_CAI' in metrics_df.columns:
                                            st.markdown("#### 📊 Interactive CAI Comparison")
                                            
                                            cai_comparison_fig = create_interactive_comparison_chart(
                                                metrics_df['Sequence'].tolist(),
                                                metrics_df['Original_CAI'].tolist(),
                                                metrics_df['Optimized_CAI'].tolist(),
                                                'CAI Score',
                                                'CAI (Codon Adaptation Index)'
                                            )
                                            st.plotly_chart(cai_comparison_fig, use_container_width=True, key="batch_cai_comparison_fig")
                                        
                                        # Summary statistics table - UPDATED
                                        st.markdown("#### 📋 Optimization Summary Report")

                                        # Create summary metrics in a more visual way - UPDATED
                                        summary_col1, summary_col2, summary_col3 = st.columns(3)

                                        # Calculate all averages
                                        avg_orig_cai = metrics_df['Original_CAI'].mean()
                                        avg_opt_cai = metrics_df['Optimized_CAI'].mean()
                                        cai_improvement = ((avg_opt_cai - avg_orig_cai) / avg_orig_cai) * 100 if avg_orig_cai > 0 else 0
                                        avg_stops_change = metrics_df['Stop_Change'].mean()  # Changed from reduction to change
                                        total_stops_changed = metrics_df['Stop_Change'].sum()  # Changed from removed to changed
                                        avg_gc_change = (metrics_df['Optimized_GC'] - metrics_df['Original_GC']).mean()

                                        with summary_col1:
                                            st.markdown("**🎯 CAI Performance**")
                                            st.metric("Original Avg CAI", f"{avg_orig_cai:.3f}")
                                            st.metric("Optimized Avg CAI", f"{avg_opt_cai:.3f}")
                                            st.metric(
                                                "CAI Improvement", 
                                                f"{cai_improvement:.1f}%",
                                                delta=f"{cai_improvement:.1f}%",
                                                delta_color="normal"
                                            )

                                        with summary_col2:
                                            st.markdown("**🛑 Stop Codon Changes**")  # Updated label
                                            st.metric("Avg Stops Changed", f"{avg_stops_change:.1f}")  # Updated metric
                                            st.metric("Total Stops Changed", f"{total_stops_changed}")  # Updated metric

                                        with summary_col3:
                                            st.markdown("**🧬 GC Content Changes**")
                                            st.metric(
                                                "Avg GC Change", 
                                                f"{avg_gc_change:+.1f}%",
                                                delta=f"{avg_gc_change:+.1f}%",
                                                delta_color="inverse"
                                            )
                                            best_cai_seq = metrics_df.loc[metrics_df['CAI_Change'].idxmax(), 'Sequence']
                                            st.metric("Best CAI Improvement", f"{best_cai_seq[:15]}...")

                                        # Add a detailed breakdown table
                                        st.markdown("#### 📊 Detailed Sequence Metrics")
                                        display_df = metrics_df[['Sequence', 'Original_CAI', 'Optimized_CAI', 'CAI_Change', 
                                                                'Original_Stops', 'Optimized_Stops', 'Stop_Change',  # Updated column name
                                                                'Original_GC', 'Optimized_GC']].copy()

                                        # Format the dataframe for display
                                        display_df['CAI_Change'] = display_df['CAI_Change'].apply(lambda x: f"{x:+.3f}")
                                        display_df['Stop_Change'] = display_df['Stop_Change'].apply(lambda x: f"{x:+d}")  # Updated column formatting
                                        display_df['Original_GC'] = display_df['Original_GC'].apply(lambda x: f"{x:.1f}%")
                                        display_df['Optimized_GC'] = display_df['Optimized_GC'].apply(lambda x: f"{x:.1f}%")
                                        display_df['Original_CAI'] = display_df['Original_CAI'].apply(lambda x: f"{x:.3f}")
                                        display_df['Optimized_CAI'] = display_df['Optimized_CAI'].apply(lambda x: f"{x:.3f}")

                                        # Rename columns for better display
                                        display_df.columns = ['Sequence', 'Orig CAI', 'Opt CAI', 'CAI Δ', 
                                                            'Orig Stops', 'Opt Stops', 'Stops Δ',  # Updated column name
                                                            'Orig GC', 'Opt GC']

                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            column_config={
                                                "Sequence": st.column_config.TextColumn("Sequence", width="medium"),
                                                "CAI Δ": st.column_config.TextColumn("CAI Δ", help="Change in CAI score"),
                                                "Stops Δ": st.column_config.TextColumn("Stops Δ", help="Change in stop codons")  # Updated help text
                                            }
                                        )
                            
                            else:
                                st.warning("No optimization results found in the batch data.")
                            
                            # Display the data table at the end
                            st.markdown("#### 📋 Complete Results Table")
                            st.dataframe(batch_df, use_container_width=True)
                        
                        # Add accumulation option for batch results
                        st.divider()
                        accumulate_batch = st.checkbox("Accumulate Batch Results", help="Add these batch results to accumulated collection")

                        if accumulate_batch and results:
                            # Add batch ID and timestamp
                            batch_id = f"Batch_{len(st.session_state.batch_accumulated_results) + 1}"
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            for result in results:
                                result['Batch_ID'] = batch_id
                                result['Timestamp'] = timestamp
                                st.session_state.batch_accumulated_results.append(result)
                            
                            st.success(f"Batch results added to accumulation (Total batches: {len(set([r['Batch_ID'] for r in st.session_state.batch_accumulated_results]))})")

                        # Download button
                        excel_data = create_download_link(batch_df, f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx")
                        st.download_button(
                            label="Download Batch Results (Excel)",
                            data=excel_data,
                            file_name=f"Batch_{batch_method}_{len(sequences)}_sequences.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No valid results generated from batch processing")
        
        # Display accumulated batch results
        if st.session_state.batch_accumulated_results:
            st.divider()
            st.subheader("📚 Accumulated Batch Results")
            
            with st.expander(f"View All Accumulated Results ({len(st.session_state.batch_accumulated_results)} sequences from {len(set([r['Batch_ID'] for r in st.session_state.batch_accumulated_results]))} batches)", expanded=False):
                acc_batch_df = pd.DataFrame(st.session_state.batch_accumulated_results)
                
                # Reorder columns
                priority_cols = ['Batch_ID', 'Timestamp', 'Sequence_Name', 'Method']
                other_cols = [col for col in acc_batch_df.columns if col not in priority_cols]
                acc_batch_df = acc_batch_df[priority_cols + other_cols]
                
                st.dataframe(acc_batch_df, use_container_width=True)
                
                # Download accumulated results
                excel_data = create_download_link(acc_batch_df, f"Accumulated_Batch_Results_{len(st.session_state.batch_accumulated_results)}_sequences.xlsx")
                st.download_button(
                    label="Download All Accumulated Results (Excel)",
                    data=excel_data,
                    file_name=f"Accumulated_Batch_Results_{len(st.session_state.batch_accumulated_results)}_sequences.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                if st.button("Clear Accumulated Batch Results"):
                    st.session_state.batch_accumulated_results = []
                    st.rerun()
                    
            if sequences and batch_file is None:  # File was uploaded but no sequences found
                st.warning("No valid sequences found in uploaded file. Please check the file format.")
                    
    
    with tab6:
        st.header("About")
        st.markdown("""
        ### DNA Codon Optimization Tool v2.5
        
        This bioinformatics application provides comprehensive DNA sequence optimization and analysis capabilities, allowing for sequences not prone to +1 ribosomal frameshfting.
        
        **Available Methods:**
        
        - **In-Frame Analysis**: Calculates Codon Adaptation Index for sequence assessment with interactive 10bp GC content window
        - **+1 Frame Analysis**: Comprehensive analysis including slippery motifs and frame analysis with interactive visualizations
        - **Standard Codon Optimization**: Uses most frequent codons for each amino acid            
        - **Balanced Optimization**: Advanced algorithm considering codon usage and +1 frame effects
        - **MaxStop**: Specialized for alternative reading frame stop codon creation
        
         **Interactive Features:**
        - **Hover Information**: Detailed tooltips on all charts showing exact values
        - **Zoom and Pan**: Interactive exploration of large datasets
        - **Click to Select**: Interactive data point selection where applicable
        - **Responsive Design**: Charts adapt to different screen sizes
        - **Real-time Updates**: Interactive controls update visualizations instantly
        
        **Core Features:**
        - Single sequence and batch processing
        - Result accumulation and export
        - Real-time validation and feedback
        - Configurable algorithm parameters
        
        
        """)


                    
             

                            
                 

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
