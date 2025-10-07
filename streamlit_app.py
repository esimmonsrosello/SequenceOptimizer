import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from collections import defaultdict, Counter
from Bio.Seq import Seq
import io
import requests
import time
import re
from dotenv import load_dotenv
from typing import List, Dict
from anthropic import Anthropic
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="HOOF",
    page_icon=":horse:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration and Constants
BIAS_WEIGHT_DEFAULT = 0.5
FRAME_OFFSET = 1
VALID_DNA_BASES = 'ATGC'
CONFIG_FILE = "codon_optimizer_config.json"
DEFAULT_CONFIG = {
    "codon_file_path": "HumanCodons.xlsx",
    "bias_weight": BIAS_WEIGHT_DEFAULT,
    "auto_open_files": True,
    "default_output_dir": "."
}

# Add to the THEMES dictionary
THEMES = {
    "Default": {
        "info": "Default color scheme with vibrant, high-contrast colors.",
        "colors": {
            "utr5": "#1900FF",
            "cds": "#4ECDC4", 
            "utr3": "#FF6B6B",
            "signal_peptide": "#8A2BE2",
            "optimization": {'original': '#FF8A80', 'optimized': '#4ECDC4'},
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
    # ADD THESE NEW COLOR-BLIND FRIENDLY THEMES
    "Colorblind Safe": {
        "info": "High contrast colors optimized for colorblind users (deuteranopia/protanopia safe).",
        "colors": {
            "utr5": "#000000",      # Black
            "cds": "#E69F00",       # Orange
            "utr3": "#56B4E9",      # Sky Blue
            "signal_peptide": "#009E73",  # Bluish Green
            "optimization": {'original': '#CC79A7', 'optimized': '#E69F00'},  # Pink -> Orange
            "analysis": ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'],
            "gradient": ['#FFF2CC', '#FFE699', '#FFD966', '#FFCC33', '#E69F00', '#CC8F00', '#B37F00', '#996F00']
        }
    },
    "High Contrast": {
        "info": "Maximum contrast theme for accessibility.",
        "colors": {
            "utr5": "#000000",      # Black
            "cds": "#FFFFFF",       # White
            "utr3": "#FF0000",      # Red
            "signal_peptide": "#00FF00",  # Green
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
# --- App Theme CSS --- (for styling the Streamlit UI itself)
APP_THEMES_CSS = {
    "Default": "",  # No custom CSS for the default theme
    "Oceanic": """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #F0F8FF;
            }
            [data-testid="stSidebar"] {
                background-color: #E0F7FA;
            }
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #004D40;
            }
        </style>
    """,
    "Sunset": """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #FFF3E0;
            }
            [data-testid="stSidebar"] {
                background-color: #FFE0B2;
            }
            h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1jicfl2 {
                color: #5D4037;
            }
        </style>
    """
}

def inject_app_theme():
    """Injects the CSS for the currently selected theme."""
    theme_css = APP_THEMES_CSS.get(st.session_state.active_theme, "")
    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)


# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()
if 'active_theme' not in st.session_state:
    st.session_state.active_theme = "Default"
if 'accumulated_results' not in st.session_state:
    st.session_state.accumulated_results = []
if 'batch_accumulated_results' not in st.session_state:
    st.session_state.batch_accumulated_results = []
if 'mrna_design_cds_paste' not in st.session_state:
    st.session_state.mrna_design_cds_paste = ""
if 'run_counter' not in st.session_state:
    st.session_state.run_counter = 0
if 'genetic_code' not in st.session_state:
    st.session_state.genetic_code = {}
if 'codon_weights' not in st.session_state:
    st.session_state.codon_weights = {}
if 'preferred_codons' not in st.session_state:
    st.session_state.preferred_codons = {}
if 'human_codon_usage' not in st.session_state:
    st.session_state.human_codon_usage = {}
if 'aa_to_codons' not in st.session_state:
    st.session_state.aa_to_codons = defaultdict(list)

# Database search results caching
if 'cached_search_results' not in st.session_state:
    st.session_state.cached_search_results = None
if 'cached_search_query' not in st.session_state:
    st.session_state.cached_search_query = ""
if 'cached_download_df' not in st.session_state:
    st.session_state.cached_download_df = None

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codon_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
Slippery_Motifs = {"TTTT", "TTTC"}
PLUS1_STOP_CODONS = {"TAA", "TAG"}
PLUS1_STOP_MOTIFS = {"TAATAA", "TAGTAG", "TAGTAA", "TAATAG"}
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

synonymous_codons = defaultdict(list)
for codon_val, aa_val in STANDARD_GENETIC_CODE.items(): 
    synonymous_codons[aa_val].append(codon_val)
    
FIRST_AA_CANDIDATES = ['L', 'I', 'V']
SECOND_AA_CANDIDATES = ['V', 'I']

# Utility Functions
def calculate_gc_window(sequence, position, window_size=25):
    """Calculate GC content for a sliding window around a given position"""
    # Convert position from 1-based to 0-based indexing
    center_pos = (position - 1) * 3  # Convert amino acid position to nucleotide position
    
    # Calculate window boundaries
    start = max(0, center_pos - window_size // 2)
    end = min(len(sequence), center_pos + window_size // 2 + 1)
    
    # Extract window sequence
    window_seq = sequence[start:end]
    
    if len(window_seq) == 0:
        return 0.0
    
    # Calculate GC content
    gc_count = sum(1 for base in window_seq.upper() if base in 'GC')
    return (gc_count / len(window_seq)) * 100

@st.cache_data
def load_immunogenic_peptides(file_path="epitope_table_export.xlsx"):
    """Load immunogenic peptides from Excel file"""
    try:
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            
            
            
            # Clean column names - remove extra spaces and handle duplicates
            df.columns = df.columns.str.strip()
            
            # Handle duplicate column names by keeping only the first occurrence
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
            
            
            
            # Look for the Name column (should be the 3rd column based on your structure)
            name_column = None
            possible_name_columns = ['Name', 'Name_1', 'Name_2', 'Name_3']
            
            for col in possible_name_columns:
                if col in df.columns:
                    name_column = col
                    break
            
            # If still not found, try to find it by position (3rd column)
            if name_column is None and len(df.columns) >= 3:
                name_column = df.columns[2]  # 3rd column (0-indexed)
                
            
            if name_column is None:
                st.error(f"Could not find Name column. Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            
            
            
            
            # Clean and prepare the data
            df_clean = df.dropna(subset=[name_column])
            df_clean = df_clean[df_clean[name_column].notna()]
            df_clean[name_column] = df_clean[name_column].astype(str).str.upper().str.strip()
            
            # Filter out very short sequences and invalid entries
            df_clean = df_clean[df_clean[name_column].str.len() >= 3]
            df_clean = df_clean[df_clean[name_column] != 'NAN']
            df_clean = df_clean[df_clean[name_column] != '']
            
            # Store the column name for later use
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
    """Generate consistent color palettes for charts based on the active theme"""
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    
    if palette_type == "optimization":
        return theme_colors["optimization"]
    elif palette_type == "analysis":
        base_colors = theme_colors["analysis"]
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    elif palette_type == "gradient":
        return theme_colors["gradient"]

def display_copyable_sequence(sequence, label, key_suffix=""):
    """Display sequence in a copyable format"""
    st.text_area(
        label,
        sequence,
        height=120,
        key=f"copy_{key_suffix}",
        help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
    )

def display_colored_mrna_sequence(utr5_seq, cds_seq, utr3_seq, signal_peptide_seq="", tag_sequence_seq="", key_suffix=""):
    """Display mRNA sequence with 5'UTR, CDS, 3'UTR, signal peptide, and optional tag highlighted in different colors."""
    st.subheader("Full mRNA Sequence (Colored)")

    # Define colors for each section from the active theme
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    color_utr5 = theme_colors["utr5"]
    color_cds = theme_colors["cds"]
    color_utr3 = theme_colors["utr3"]
    color_signal_peptide = theme_colors["signal_peptide"]
    color_tag = "#FFA500"  # Orange color for tags

    # Create HTML string with colored spans
    colored_parts = []
    full_sequence_parts = []
    
    if utr5_seq:
        colored_parts.append(f'<span style="color: {color_utr5}; font-weight: bold;">{utr5_seq}</span>')
        full_sequence_parts.append(utr5_seq)
    
    if signal_peptide_seq:
        colored_parts.append(f'<span style="color: {color_signal_peptide}; font-weight: bold;">{signal_peptide_seq}</span>')
        full_sequence_parts.append(signal_peptide_seq)
    
    if cds_seq:
        colored_parts.append(f'<span style="color: {color_cds}; font-weight: bold;">{cds_seq}</span>')
        full_sequence_parts.append(cds_seq)
    
    if tag_sequence_seq:
        colored_parts.append(f'<span style="color: {color_tag}; font-weight: bold;">{tag_sequence_seq}</span>')
        full_sequence_parts.append(tag_sequence_seq)
    
    if utr3_seq:
        colored_parts.append(f'<span style="color: {color_utr3}; font-weight: bold;">{utr3_seq}</span>')
        full_sequence_parts.append(utr3_seq)

    colored_html = f"""
    <div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.8em;">
        {''.join(colored_parts)}
    </div>
    """
    st.markdown(colored_html, unsafe_allow_html=True)

    # Also provide a copyable text area for the full sequence
    full_sequence = ''.join(full_sequence_parts)
    st.text_area(
        "Copy Full mRNA Sequence:",
        full_sequence,
        height=120,
        key=f"copy_full_mrna_{key_suffix}",
        help="Click in the text area and use Ctrl+A to select all, then Ctrl+C to copy"
    )
    
    # Update legend
    legend_items = []
    if utr5_seq:
        legend_items.append(f'<span style="color: {color_utr5};">■</span> 5\' UTR ({len(utr5_seq)} bp)')
    if signal_peptide_seq:
        legend_items.append(f'<span style="color: {color_signal_peptide};">■</span> Signal Peptide ({len(signal_peptide_seq)} bp)')
    if cds_seq:
        legend_items.append(f'<span style="color: {color_cds};">■</span> CDS ({len(cds_seq)} bp)')
    if tag_sequence_seq:
        legend_items.append(f'<span style="color: {color_tag};">■</span> 3\' Tag ({len(tag_sequence_seq)} bp)')
    if utr3_seq:
        legend_items.append(f'<span style="color: {color_utr3};">■</span> 3\' UTR ({len(utr3_seq)} bp)')
    
    legend_html = f"""
    <div style="font-size: 0.8em; color: gray;">
        {' &nbsp;&nbsp; '.join(legend_items)}
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    
def create_geneious_like_visualization(utr5_seq, cds_seq, utr3_seq, signal_peptide_seq="", tag_sequence_seq="", double_stop_codon="TAATAA", key_suffix=""):
    """
    Create a Geneious-like visualization of the mRNA sequence with nucleotides and amino acids.
    Amino acids are only shown for the coding sequence (signal peptide + CDS + tag).
    """
    # Process the sequences to handle the stop codon correctly
    processed_cds = cds_seq.strip()
    processed_tag = tag_sequence_seq.strip()
    stop_codons = {"TAA", "TAG", "TGA"}

    # Remove existing stop codons from the tag or CDS
    if processed_tag:
        while len(processed_tag) >= 3 and processed_tag[-3:].upper() in stop_codons:
            processed_tag = processed_tag[:-3]
    else:
        while len(processed_cds) >= 3 and processed_cds[-3:].upper() in stop_codons:
            processed_cds = processed_cds[:-3]

    # Generate a unique suffix based on key_suffix and a random value
    unique_id = f"{key_suffix}_{id(utr5_seq)}"
    
    # Get theme colors
    theme_colors = THEMES[st.session_state.active_theme]["colors"]
    color_utr5 = theme_colors["utr5"]
    color_cds = theme_colors["cds"]
    color_utr3 = theme_colors["utr3"]
    color_signal_peptide = theme_colors["signal_peptide"]
    color_tag = theme_colors.get("tag", "#FFA500")  # Use theme color or fallback
    
    # Create the visualization sections
    sections = []
    
    # 5' UTR Section
    if utr5_seq:
        sections.append({
            'name': "5' UTR",
            'sequence': utr5_seq,
            'color': color_utr5,
            'show_aa': False
        })
    
    # Signal Peptide Section
    if signal_peptide_seq:
        sections.append({
            'name': "Signal Peptide",
            'sequence': signal_peptide_seq,
            'color': color_signal_peptide,
            'show_aa': True
        })
    
    # CDS Section
    if processed_cds:
        sections.append({
            'name': "CDS",
            'sequence': processed_cds,
            'color': color_cds,
            'show_aa': True
        })
    
    # 3' Tag Section (NEW)
    if processed_tag:
        sections.append({
            'name': "3' Tag",
            'sequence': processed_tag,
            'color': color_tag,
            'show_aa': True
        })

    # Stop Codon Section
    sections.append({
        'name': "Stop Codon",
        'sequence': double_stop_codon,
        'color': color_cds,  # Same color as CDS for now
        'show_aa': True
    })
    
    # 3' UTR Section
    if utr3_seq:
        sections.append({
            'name': "3' UTR",
            'sequence': utr3_seq,
            'color': color_utr3,
            'show_aa': False
        })
    
    # Display each section
    for section_idx, section in enumerate(sections):
        st.markdown(f"#### {section['name']} ({len(section['sequence'])} bp)")
        
        seq = section['sequence']
        color = section['color']
        show_aa = section['show_aa']
        
        # For coding sequences, we want to align codons properly
        # Use chunk size that's divisible by 3 for coding regions
        if show_aa:
            # Use 60 nucleotides (20 codons) for coding regions
            chunk_size = 60
        else:
            # Use 60 nucleotides for non-coding regions
            chunk_size = 60
        
        for chunk_idx, i in enumerate(range(0, len(seq), chunk_size)):
            chunk = seq[i:i+chunk_size]
            start_pos = i + 1
            end_pos = min(i + chunk_size, len(seq))
            
            # Display position info
            st.markdown(f"**Position {start_pos}-{end_pos}**")
            
            if show_aa and len(chunk) >= 3:
                # For coding sequences, create aligned nucleotide and amino acid display
                
                # Split nucleotides into codons for better visualization
                codons = []
                amino_acids = []
                
                for j in range(0, len(chunk) - 2, 3):
                    codon = chunk[j:j+3]
                    if len(codon) == 3:
                        codons.append(codon)
                        aa = st.session_state.genetic_code.get(codon.upper(), 'X')
                        amino_acids.append(aa)
                
                # Handle remaining nucleotides (less than 3)
                remaining = chunk[len(codons)*3:]
                if remaining:
                    codons.append(remaining + " " * (3 - len(remaining)))  # Pad with spaces
                    amino_acids.append(" ")  # Space for incomplete codon
                
                # Create spaced codon display
                spaced_codons = "   ".join(codons)  # 3 spaces between codons

                # Center each AA under its codon
                spaced_aas = "   ".join([f" {aa} " for aa in amino_acids])  # pad each AA with 1 space

                # Display nucleotides (codons)
                nucleotide_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    border-left: 4px solid {color};
                    margin: 5px 0;
                ">
                    <div style="
                        color: {color}; 
                        font-weight: bold; 
                        font-size: 1.1em; 
                        letter-spacing: 1px;
                        word-break: break-all;
                    ">
                        {spaced_codons}
                    </div>
                </div>
                """
                st.markdown(nucleotide_html, unsafe_allow_html=True)
                
                # Display amino acids aligned with codons
                aa_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #fff; 
                    padding: 5px 10px; 
                    border-radius: 3px; 
                    border-left: 4px solid {color};
                    margin: 0 0 10px 0;
                ">
                    <div style="
                        color: #333; 
                        font-size: 1.0em; 
                        letter-spacing: 13.5px;
                        font-weight: bold;
                    ">
                        {spaced_aas}
                    </div>
                </div>
                """
                st.markdown(aa_html, unsafe_allow_html=True)
                
            else:
                # For non-coding sequences, just display nucleotides
                nucleotide_html = f"""
                <div style="
                    font-family: 'Courier New', monospace; 
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px; 
                    border-left: 4px solid {color};
                    margin: 5px 0;
                ">
                    <div style="
                        color: {color}; 
                        font-weight: bold; 
                        font-size: 1.1em; 
                        letter-spacing: 1px;
                        word-break: break-all;
                    ">
                        {chunk}
                    </div>
                </div>
                """
                st.markdown(nucleotide_html, unsafe_allow_html=True)
            
            # Add some spacing between chunks
            if i + chunk_size < len(seq):
                st.markdown("---")
    
    # Add summary information
    st.markdown("### Sequence Summary")
    
    total_length = len(utr5_seq) + len(signal_peptide_seq) + len(processed_cds) + len(processed_tag) + len(double_stop_codon) + len(utr3_seq)
    coding_length = len(signal_peptide_seq) + len(processed_cds) + len(processed_tag) + len(double_stop_codon)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Length", f"{total_length} bp")
    with col2:
        st.metric("Coding Length", f"{coding_length} bp")
    with col3:
        if coding_length > 0:
            protein_length = coding_length // 3
            st.metric("Protein Length", f"{protein_length} aa")
        else:
            st.metric("Protein Length", "0 aa")
    with col4:
        full_seq = utr5_seq + signal_peptide_seq + processed_cds + processed_tag + double_stop_codon + utr3_seq
        gc_content = calculate_gc_content(full_seq) if full_seq else 0
        st.metric("GC Content", f"{gc_content:.1f}%")
    
    # Legend
    st.markdown("### Legend")
    legend_items = []
    
    if utr5_seq:
        legend_items.append(f'<span style="color: {color_utr5}; font-weight: bold;">■</span> 5\' UTR')
    if signal_peptide_seq:
        legend_items.append(f'<span style="color: {color_signal_peptide}; font-weight: bold;">■</span> Signal Peptide')
    if processed_cds:
        legend_items.append(f'<span style="color: {color_cds}; font-weight: bold;">■</span> CDS')
    if processed_tag:
        legend_items.append(f'<span style="color: {color_tag}; font-weight: bold;">■</span> 3\' Tag')
    legend_items.append(f'<span style="color: {color_cds}; font-weight: bold;">■</span> Stop Codon')
    if utr3_seq:
        legend_items.append(f'<span style="color: {color_utr3}; font-weight: bold;">■</span> 3\' UTR')
    
    legend_html = f"""
    <div style="font-size: 0.9em; margin: 10px 0;">
        {' &nbsp;&nbsp; '.join(legend_items)}
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Add explanation
    st.info("💡 **Reading Guide**: In coding regions, nucleotides are grouped by codons (3 letters) with the corresponding amino acid shown below each codon.")



def find_coding_sequence_bounds(dna_seq):
    """Find start and stop positions of coding sequence, prioritizing ACCATG."""
    dna_seq_upper = dna_seq.upper().replace('U', 'T')
    stop_codons = {"TAA", "TAG", "TGA"}
    
    start_pos = None
    
    # Always prioritize finding the ACCATG Kozak sequence.
    accatg_pos = dna_seq_upper.find('ACCATG')
    if accatg_pos != -1:
        # The actual start codon (ATG) begins 3 bases into "ACCATG".
        start_pos = accatg_pos + 3
    else:
        # Fallback: if no ACCATG, find the first occurrence of ATG.
        atg_pos = dna_seq_upper.find('ATG')
        if atg_pos != -1:
            # The sequence starts at the beginning of the first ATG found.
            start_pos = atg_pos
            
    if start_pos is None:
        # If no start codon is found at all, we can't proceed.
        return None, None
    
    # Find end position - first in-frame stop codon, starting from our found start_pos.
    end_pos = None
    for i in range(start_pos, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        if len(codon) == 3 and codon in stop_codons:
            end_pos = i  # Position of the stop codon itself.
            break
            
    return start_pos, end_pos


def create_interactive_cai_gc_plot(positions, cai_weights, amino_acids, sequence, seq_name, color='#4ECDC4'):
    """Create interactive plot combining CAI weights and GC content"""
    
    # Calculate 10bp window GC content for each position
    gc_content_10bp = [calculate_gc_window(sequence, pos, 25) for pos in positions]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f'CAI Weights and 25bp GC Content - {seq_name}']
    )
    
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
    
    # Add GC content trace
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
    
    # Set x-axis title
    fig.update_xaxes(title_text="Amino Acid Position")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="GC Content (%)", secondary_y=True, range=[0, 100])
    
    # Update layout
    fig.update_layout(
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

def create_interactive_cai_stop_codon_plot(positions, cai_weights, amino_acids, stop_codon_positions, seq_name, frame_type, color='#4ECDC4'):
    """Create interactive plot combining CAI weights and stop codon locations"""
    
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
    
    # Add stop codon bars
    if stop_codon_positions:
        theme_colors = get_consistent_color_palette(1, "optimization")
        fig.add_trace(
            go.Bar(
                x=stop_codon_positions,
                y=[1] * len(stop_codon_positions), # Bars will go up to y=1 on secondary axis
                name=f'{frame_type} Stop Codons',
                marker_color=theme_colors['original'],  # Use theme color for stops
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br><b>Stop Codon</b><extra></extra>'
            ),
            secondary_y=True,
        )

    # Set x-axis title
    fig.update_xaxes(title_text="Amino Acid Position")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Stop Codon", secondary_y=True, showticklabels=False, range=[0, 1])
    
    # Update layout
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
    """Create interactive bar chart using the active theme"""
    theme_analysis_colors = get_consistent_color_palette(len(x_data), "analysis")
    fig = go.Figure(data=go.Bar(
        x=x_data,
        y=y_data,
        text=[f'{val:.1f}' for val in y_data],
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
    """Create interactive pie chart using the active theme"""
    # Get theme colors - use first few colors from analysis palette
    theme_colors = THEMES[st.session_state.active_theme]["colors"]["analysis"]
    
    # Create color list matching the number of labels
    chart_colors = []
    for i in range(len(labels)):
        chart_colors.append(theme_colors[i % len(theme_colors)])
    
    # For single sequence analysis, show absolute numbers
    textinfo = 'label+percent' if show_percentages else 'label+value'
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo=textinfo,
        marker=dict(
            colors=chart_colors,
            line=dict(color='#FFFFFF', width=2)  # White borders for better definition
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
            x=1.05
        ),
        margin=dict(l=20, r=120, t=50, b=20)  # Adjust margins for legend
    )
    
    return fig

def create_interactive_comparison_chart(sequences, original_values, optimized_values, metric_name, y_title):
    """Create interactive before/after comparison chart"""
    fig = go.Figure()
    
    colors = get_consistent_color_palette(1, "optimization")
    
    # Add original values
    fig.add_trace(go.Bar(
        name='Original',
        x=sequences,
        y=original_values,
        marker_color=colors['original'],
        hovertemplate='<b>%{x}</b><br>Original ' + metric_name + ': %{y}<extra></extra>'
    ))
    
    # Add optimized values
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
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_stacked_bar_chart(x_data, y_data_dict, title, y_title):
    """Create interactive stacked bar chart"""
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
        barmode='stack',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_interactive_cai_gc_overlay_plot(
    positions, cai_weights, amino_acids, sequence, seq_name,
    plus1_stop_positions=None, minus1_stop_positions=None, slippery_positions=None,
    show_options=None,  # Parameter ignored for compatibility
    color='#4ECDC4'
):
    """Create interactive plot with a clickable legend to toggle overlays."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # CAI weights (Visible by default)
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

    # GC content (Visible by default)
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

    # +1 stops (Hidden by default, toggled via legend)
    if plus1_stop_positions:
        fig.add_trace(
            go.Bar(
                x=plus1_stop_positions,
                y=[100] * len(plus1_stop_positions), # Use 100 to match the GC axis
                name='+1 Stops',
                marker_color='#FF6B6B',
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>+1 Stop Codon<extra></extra>',
                visible='legendonly'
            ),
            secondary_y=True,
        )

    # -1 stops (Hidden by default, toggled via legend)
    if minus1_stop_positions:
        fig.add_trace(
            go.Bar(
                x=minus1_stop_positions,
                y=[100] * len(minus1_stop_positions), # Use 100 to match the GC axis
                name='-1 Stops',
                marker_color='#4ECDC4',
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>-1 Stop Codon<extra></extra>',
                visible='legendonly'
            ),
            secondary_y=True,
        )

    # Slippery sites (Hidden by default, toggled via legend)
    if slippery_positions:
        slippery_aa_positions = [pos['amino_acid_position'] for pos in slippery_positions]
        slippery_motifs = [pos['motif'] for pos in slippery_positions]
        fig.add_trace(
            go.Bar(
                x=slippery_aa_positions,
                y=[100] * len(slippery_aa_positions), # Use 100 to match the GC axis
                name='Slippery Sites',
                marker_color='#FFD700',
                opacity=0.6,
                width=0.8,
                hovertemplate='<b>Position:</b> %{x}<br>Motif: %{customdata}<extra></extra>',
                customdata=slippery_motifs,
                visible='legendonly'
            ),
            secondary_y=True,
        )

    fig.update_xaxes(
        title_text="Amino Acid Position",
        range=[1, len(amino_acids) + 1],  # Ensure x-axis starts at 1 and covers all positions
        fixedrange=True  # Prevent zooming out beyond data limits
    )

    # Primary Y-axis for CAI
    fig.update_yaxes(title_text="CAI Weight", secondary_y=False, range=[0, 1])

    # Secondary Y-axis for GC Content and event markers
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

    # 2. Convert the figure to a self-contained HTML block.
    # This packages all the necessary JavaScript and data into one string.
    chart_html = overlay_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. Render the HTML using st.components.v1.html.
    # This creates a sandboxed iframe, isolating the chart's state from Streamlit.
    components.html(chart_html, height=550, scrolling=True)

# Validation of final CDS (before adding UTRs)
def validate_final_coding_sequence(full_coding_dna):
    STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}
    problems = []

    # 1. Check start codon
    if not full_coding_dna.startswith("ATG"):
        problems.append("⚠️ CDS does not start with an ATG start codon.")

    # 2. Check for in-frame stop codons before the final codons
    internal_sequence = full_coding_dna[:-6]  # Exclude last 2 codons (6 bases)
    for i in range(0, len(internal_sequence) - 2, 3):
        codon = internal_sequence[i:i+3]
        if codon in STANDARD_STOP_CODONS:
            problems.append(f"❌ Premature stop codon ({codon}) found at position {i+1}-{i+3}.")
            break

    # 3. Check if it ends with exactly two stop codons (no more, no less)
    final_codons = [full_coding_dna[-6:-3], full_coding_dna[-3:]]
    if not all(c in STANDARD_STOP_CODONS for c in final_codons):
        problems.append(f"❌ Final two codons are not valid stop codons: {final_codons}")
    elif len(full_coding_dna) >= 9:
        third_last_codon = full_coding_dna[-9:-6]
        if third_last_codon in STANDARD_STOP_CODONS:
            problems.append(f"❌ More than two stop codons at the end. Found extra stop codon before the final two: {third_last_codon}")

    return problems


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

def adjust_gc_content(sequence, max_gc=70.0, min_gc=55.0):
    """
    Adjusts the GC content of a sequence to be within a target range by using synonymous codons.
    Prioritizes swapping high-GC codons for low-GC codons.
    """
    # Check if codon data is loaded
    if not st.session_state.genetic_code or not st.session_state.aa_to_codons:
        st.error("Codon usage data not loaded. Cannot adjust GC content.")
        return sequence

    current_gc = calculate_gc_content(sequence)
    if current_gc <= max_gc:
        st.info(f"Initial GC content ({current_gc:.1f}%) is already within the target range (<= {max_gc}%). No adjustment needed.")
        return sequence

    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    new_codons = list(codons)
    
    # Create a list of potential swaps, prioritized by GC content reduction
    potential_swaps = []
    for i, codon in enumerate(codons):
        aa = st.session_state.genetic_code.get(codon)
        if not aa or aa == '*':
            continue

        current_codon_gc = get_codon_gc_content(codon)
        
        # Find synonymous codons with lower GC content
        for syn_codon, freq in st.session_state.aa_to_codons.get(aa, []):
            syn_codon_gc = get_codon_gc_content(syn_codon)
            if syn_codon_gc < current_codon_gc:
                gc_reduction = current_codon_gc - syn_codon_gc
                # Store index, new codon, and the reduction amount for prioritization
                potential_swaps.append({'index': i, 'new_codon': syn_codon, 'reduction': gc_reduction, 'original_codon': codon})

    # Sort swaps by the amount of GC reduction (descending)
    potential_swaps.sort(key=lambda x: x['reduction'], reverse=True)

    # Apply swaps until GC content is acceptable
    swapped_indices = set()
    for swap in potential_swaps:
        if current_gc <= max_gc:
            break # Stop if we've reached the target
        
        idx = swap['index']
        if idx not in swapped_indices:
            new_codons[idx] = swap['new_codon']
            swapped_indices.add(idx)
            # Recalculate GC content after the swap
            current_gc = calculate_gc_content("".join(new_codons))

    final_sequence = "".join(new_codons)
    final_gc = calculate_gc_content(final_sequence)
    st.success(f"GC content adjusted from {calculate_gc_content(sequence):.1f}% to {final_gc:.1f}%")
    
    return final_sequence

def enforce_local_gc_content(sequence, target_max_gc=70.0, window_size=10, step_size=1):
    """
    Enforces local GC content by adjusting codons in windows exceeding target_max_gc.
    Attempts to maintain protein sequence.
    """
    if not st.session_state.aa_to_codons or not st.session_state.genetic_code:
        st.error("Codon usage data not loaded. Cannot enforce local GC adjustment.")
        return sequence

    current_sequence = list(sequence) # Convert to list for mutability
    original_protein = translate_dna("".join(current_sequence))
    changes_made = 0

    # Iterate and adjust
    for i in range(0, len(current_sequence) - window_size + 1, step_size):
        window_start = i
        window_end = i + window_size
        window_seq = "".join(current_sequence[window_start:window_end])
        
        local_gc = calculate_gc_content(window_seq) # Use the existing calculate_gc_content for the window

        if local_gc > target_max_gc:
            # Identify codons within this window that can be swapped
            # This is the most complex part:
            # 1. Find codons in the window.
            # 2. For each codon, find synonymous codons with lower GC content.
            # 3. Prioritize swaps that reduce GC and are within the window.
            # 4. Apply swap and re-check local GC.
            
            # For simplicity in this first pass, let's try a greedy approach:
            # Iterate through codons in the window and try to swap them if they are high GC
            
            # Map nucleotide position to codon index
            codon_indices_in_window = set()
            for bp_idx in range(window_start, window_end):
                codon_idx = bp_idx // 3
                codon_indices_in_window.add(codon_idx)

            # Sort to ensure consistent processing
            sorted_codon_indices = sorted(list(codon_indices_in_window))

            for codon_idx in sorted_codon_indices:
                codon_start_bp = codon_idx * 3
                codon_end_bp = codon_start_bp + 3

                # Ensure the codon is fully within the original sequence bounds
                if codon_end_bp <= len(sequence):
                    original_codon = "".join(current_sequence[codon_start_bp:codon_end_bp])
                    aa = st.session_state.genetic_code.get(original_codon)

                    if aa and aa != '*': # Don't optimize stop codons
                        original_codon_gc = get_codon_gc_content(original_codon)
                        
                        best_syn_codon = original_codon
                        max_gc_reduction = 0

                        # Find a synonymous codon with lower GC
                        for syn_c, _ in st.session_state.aa_to_codons.get(aa, []):
                            syn_c_gc = get_codon_gc_content(syn_c)
                            if syn_c_gc < original_codon_gc:
                                if (original_codon_gc - syn_c_gc) > max_gc_reduction:
                                    max_gc_reduction = original_codon_gc - syn_c_gc
                                    best_syn_codon = syn_c
                        
                        if best_syn_codon != original_codon:
                            # Temporarily apply the swap and check if it helps the local GC
                            temp_sequence_list = list(current_sequence)
                            temp_sequence_list[codon_start_bp:codon_end_bp] = list(best_syn_codon)
                            
                            temp_window_seq = "".join(temp_sequence_list[window_start:window_end])
                            temp_local_gc = calculate_gc_content(temp_window_seq)

                            if temp_local_gc <= target_max_gc: # If this swap fixes the window
                                current_sequence[codon_start_bp:codon_end_bp] = list(best_syn_codon)
                                changes_made += 1
                                # Re-check the current window's GC after a change
                                local_gc = calculate_gc_content("".join(current_sequence[window_start:window_end]))
                                if local_gc <= target_max_gc:
                                    break # Move to next window if this one is fixed
                            # else: # If the swap doesn't fix it, try another codon in the window or move on
    
    final_sequence = "".join(current_sequence)
    final_protein = translate_dna(final_sequence)

    if original_protein != final_protein:
        st.warning("Local GC adjustment changed protein sequence. Reverting to original CDS.")
        return sequence # Revert if protein sequence changed

    if changes_made > 0:
        st.success(f"Local GC content adjusted. {changes_made} codon swaps performed.")
    else:
        st.info("No local GC content adjustments needed or possible.")
    
    return final_sequence

def generate_detailed_mrna_summary(processed_cds, final_mrna_sequence, utr_5, utr_3):
    """Generate a detailed summary DataFrame for the designed mRNA."""
    
    # Basic lengths
    summary_data = {
        "Metric": ["Final mRNA Length", "5' UTR Length", "CDS Length", "3' UTR Length"],
        "Value": [f"{len(final_mrna_sequence)} bp", f"{len(utr_5)} bp", f"{len(processed_cds)} bp", f"{len(utr_3)} bp"]
    }
    
    # GC Content
    summary_data["Metric"].append("CDS GC Content")
    summary_data["Value"].append(f"{calculate_gc_content(processed_cds):.1f}%")
    
    # CAI
    cai_weights, _ = get_codon_weights_row(processed_cds)
    if cai_weights:
        summary_data["Metric"].extend(["Average CAI", "Sequence Length"])
        summary_data["Value"].extend([
            f"{sum(cai_weights)/len(cai_weights):.3f}",
            f"{len(processed_cds)} bp"
        ])

    # +1 Stops
    plus1_stops = number_of_plus1_stops(processed_cds)
    summary_data["Metric"].extend(["+1 Total Stops", "+1 TAA", "+1 TAG", "+1 TGA"])
    summary_data["Value"].extend([
        plus1_stops['total'],
        plus1_stops['TAA'],
        plus1_stops['TAG'],
        plus1_stops['TGA']
    ])

    # -1 Stops
    minus1_stops = number_of_minus1_stops(processed_cds)
    summary_data["Metric"].extend(["-1 Total Stops", "-1 TAA", "-1 TAG", "-1 TGA"])
    summary_data["Value"].extend([
        minus1_stops['total'],
        minus1_stops['TAA'],
        minus1_stops['TAG'],
        minus1_stops['TGA']
    ])

    # Slippery Motifs
    slippery_count = number_of_slippery_motifs(utr_5 + processed_cds)
    summary_data["Metric"].append("Slippery Motifs")
    summary_data["Value"].append(slippery_count)
    
    return pd.DataFrame(summary_data)

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

def reverse_translate_highest_cai(protein_seq):
    """Reverse translates a protein sequence into DNA using the highest CAI codons."""
    if not st.session_state.preferred_codons:
        st.error("Codon usage data not loaded. Cannot reverse translate.")
        return ""
    
    dna_seq = ""
    for aa in protein_seq:
        # Handle stop codons if they appear in the protein sequence (e.g., from a partial sequence)
        if aa == '*':
            # Use TAA as a default stop codon for reverse translation
            dna_seq += "TAA"
        else:
            codon = st.session_state.preferred_codons.get(aa)
            if codon:
                dna_seq += codon
            else:
                # Fallback if no preferred codon found (should not happen for standard AAs)
                st.warning(f"No preferred codon found for amino acid: {aa}. Using NNN.")
                dna_seq += "NNN" # NNN for unknown codon
    return dna_seq

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
    # Iterate through the sequence starting from the 1st base (0-indexed)
    # and check codons in the +1 frame (offset by 1 base)
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
    # Iterate through the sequence starting from the 2nd base (0-indexed)
    # and check codons in the -1 frame (offset by 2 bases)
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
    """Balanced optimization considering codon usage and +1 frame stops"""
    bias_weight = bias_weight_input if bias_weight_input is not None else st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)
    
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    aa_to_codons = st.session_state.aa_to_codons
    
    # Protein translation
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))
    
    optimised_seq = ""
    idx = 0
    while idx < len(dna_seq_upper) - 2:
        current_codon = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(current_codon, str(Seq(current_codon).translate()))

        if idx < len(dna_seq_upper) - 5:  # Check for two-codon substitutions
            next_codon_val = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(next_codon_val, str(Seq(next_codon_val).translate()))
            candidates = []
            
            if aa in aa_to_codons and aa2 in aa_to_codons:
                for c1, f1 in aa_to_codons[aa]:
                    for c2, f2 in aa_to_codons[aa2]:
                        combined = c1 + c2
                        codon1_plus1 = combined[1:4]
                        bonus = 0
                        if codon1_plus1 in PLUS1_STOP_CODONS and combined[2:5] in PLUS1_STOP_CODONS:
                            bonus += 2
                        elif codon1_plus1 in PLUS1_STOP_CODONS:
                            bonus += 1
                        
                        score = (f1 * f2) + bias_weight * bonus
                        candidates.append((score, c1, c2))
            
            if candidates:
                _, best1, best2 = max(candidates)
                optimised_seq += best1 + best2
                idx += 6
                continue
        
        # Single codon substitution
        best_codon_val = current_codon
        current_codon_freq = 0
        for syn_c, freq_val in aa_to_codons.get(aa, []):
            if syn_c == current_codon:
                current_codon_freq = freq_val
                break
        
        temp_seq_orig = optimised_seq + current_codon + dna_seq_upper[idx+3:]
        plus1_window_orig_start = len(optimised_seq) + 1
        bonus_orig = 0
        if plus1_window_orig_start < len(temp_seq_orig) - 2:
            codon_plus1_orig = temp_seq_orig[plus1_window_orig_start:plus1_window_orig_start+3]
            if codon_plus1_orig in PLUS1_STOP_CODONS:
                bonus_orig = bias_weight
        best_score = current_codon_freq + bonus_orig
        
        for syn_codon, freq in aa_to_codons.get(aa, []):
            temp_seq = optimised_seq + syn_codon + dna_seq_upper[idx+3:]
            plus1_codon_start_in_temp = len(optimised_seq) + 1
            
            bonus_val = 0
            if plus1_codon_start_in_temp < len(temp_seq) - 2:
                codon_plus1 = temp_seq[plus1_codon_start_in_temp:plus1_codon_start_in_temp+3]
                if codon_plus1 in PLUS1_STOP_CODONS:
                    bonus_val = bias_weight
            
            score = freq + bonus_val
            if score > best_score:
                best_score = score
                best_codon_val = syn_codon
        
        optimised_seq += best_codon_val
        idx += 3

    if idx < len(dna_seq_upper):
        optimised_seq += dna_seq_upper[idx:]
    
    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in balanced optimization!")
        return dna_seq_upper
    
    return optimised_seq

def nc_stop_codon_optimisation(dna_seq):
    """MaxStop"""
    dna_seq_upper = dna_seq.upper()
    genetic_code = st.session_state.genetic_code
    
    protein_str = ""
    for i in range(0, len(dna_seq_upper) - 2, 3):
        codon = dna_seq_upper[i:i+3]
        protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    synonymous_codons_local = defaultdict(list)
    for c, aa_val in genetic_code.items():
        synonymous_codons_local[aa_val].append(c)
    
    optimised_seq = ""
    idx = 0
    while idx < len(dna_seq_upper) - 2:
        codon_val = dna_seq_upper[idx:idx+3]
        aa = genetic_code.get(codon_val, str(Seq(codon_val).translate()))

        if idx < len(dna_seq_upper) - 5:  # Try double substitution
            codon2 = dna_seq_upper[idx+3:idx+6]
            aa2 = genetic_code.get(codon2, str(Seq(codon2).translate()))
            if aa in synonymous_codons_local and aa2 in synonymous_codons_local:
                double_subs_orig_check = [(c1, c2) for c1 in synonymous_codons_local[aa] 
                                        for c2 in synonymous_codons_local[aa2] 
                                        if (c1 + c2)[1:7] in {"TAATAA", "TAGTAG"}]
                if double_subs_orig_check:
                    best_c1, best_c2 = double_subs_orig_check[0]
                    optimised_seq += best_c1 + best_c2
                    idx += 6
                    continue
        
        best_codon_val = codon_val
        # For single codon, check if any synonym creates TAA or TAG in +1 frame
        if idx + 3 < len(dna_seq_upper):
            next_actual_codon = dna_seq_upper[idx+3:idx+6]
            for syn_c in synonymous_codons_local.get(aa, []):
                plus1_codon = syn_c[1:3] + next_actual_codon[0:1]
                if plus1_codon in {"TAA", "TAG"}:
                    best_codon_val = syn_c
                    break
        optimised_seq += best_codon_val
        idx += 3

    # Verify protein sequence unchanged
    final_protein_str = ""
    for i in range(0, len(optimised_seq) - 2, 3):
        codon = optimised_seq[i:i+3]
        final_protein_str += genetic_code.get(codon, str(Seq(codon).translate()))

    if final_protein_str != protein_str:
        logger.error("Protein sequence changed in nc_stop_codon_optimisation!")
        return dna_seq_upper
    
    return optimised_seq

def third_aa_has_A_G_synonymous(aa):
    """Check if amino acid has synonymous codons starting with A or G"""
    for codon_val in synonymous_codons.get(aa, []):
        if codon_val.startswith(('A', 'G')):
            return True
    return False

def JT_Plus1_Stop_Optimized(seq_input):
    """JT Plus1 stop optimization"""
    seq = seq_input.upper()
    out_seq = ''
    idx = 0
    while idx <= len(seq) - 9:
        c1, c2, c3 = seq[idx:idx+3], seq[idx+3:idx+6], seq[idx+6:idx+9]
        aa1 = STANDARD_GENETIC_CODE.get(c1, '?')
        aa2 = STANDARD_GENETIC_CODE.get(c2, '?')
        aa3 = STANDARD_GENETIC_CODE.get(c3, '?')

        if (aa1 in FIRST_AA_CANDIDATES and aa2 in SECOND_AA_CANDIDATES and
            aa3 in synonymous_codons and third_aa_has_A_G_synonymous(aa3)):
            found_motif = False
            for syn1 in synonymous_codons.get(aa1, []):
                if not syn1.endswith('TA'):
                    continue
                for syn2 in synonymous_codons.get(aa2, []):
                    if not syn2.startswith(('A', 'G')):
                        continue
                    for syn3 in synonymous_codons.get(aa3, []):
                        if not syn3.startswith(('A', 'G')):
                            continue
                        motif_check = syn1[1:] + syn2 + syn3[:1]
                        if motif_check in PLUS1_STOP_MOTIFS:
                            out_seq += syn1 + syn2 + syn3
                            idx += 9
                            found_motif = True
                            break
                    if found_motif:
                        break
                if found_motif:
                    break
            if not found_motif:
                out_seq += c1
                idx += 3
        else:
            out_seq += c1
            idx += 3
    out_seq += seq[idx:]
    return out_seq




class NCBISearchEngine:
    def __init__(self):
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.base_url = "https://www.ncbi.nlm.nih.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    def search_nucleotide_sequences(self, query: str, max_results: int = 10, quoted_terms: List[str] = None) -> List[Dict]:
        """Search NCBI nucleotide database using Google search with two-pass filtering"""
        if not self.serper_api_key:
            st.error("SERPER API key is required for NCBI search. Please check your .env file.")
            return []
        
        # Two-pass approach
        # Pass 1: Search using the entire query to find relevant candidates
        st.write(f"🔍 **Pass 1:** Searching for candidates matching entire query")
            
        try:
            url = "https://google.serper.dev/search"
            ncbi_query = f"site:ncbi.nlm.nih.gov/nuccore {query}"
            
            payload = {"q": ncbi_query, "num": max_results * 3}  # Get more candidates for filtering
            headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 403:
                st.error("SERPER API Key is invalid or doesn't have permission")
                return []
            elif response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return []
            
            search_results = response.json().get('organic', [])
            candidates = []
            
            for result in search_results:
                try:
                    title = result.get('title', '')
                    link = result.get('link', '')
                    snippet = result.get('snippet', '')
                    
                    accession = self.extract_accession_number(link, title)
                    clean_title = title.replace(' - Nucleotide - NCBI', '').replace(' - NCBI', '').strip()
                    
                    candidates.append({
                        'title': clean_title,
                        'accession': accession,
                        'description': snippet,
                        'link': link
                    })
                except Exception as e:
                    continue
            
            st.write(f"✅ **Pass 1 complete:** Found {len(candidates)} candidate sequences")
            
            # Pass 2: If quoted terms exist, filter candidates by relevance to the full query
            if quoted_terms:
                st.write(f"🎯 **Pass 2:** Filtering candidates for relevance (before CDS analysis)")
                filtered_candidates = []
                
                for candidate in candidates:
                    # Check if the candidate is relevant to the overall query context
                    searchable_text = f"{candidate['title']} {candidate['description']}".lower()
                    
                    # Simple relevance scoring - must contain some key terms from the query
                    query_words = [word.strip().lower() for word in query.replace('"', '').split() if len(word.strip()) > 2]
                    
                    # Count how many query words appear in the candidate
                    matches = sum(1 for word in query_words if word in searchable_text)
                    relevance_score = matches / len(query_words) if query_words else 0
                    
                    # Keep candidates with reasonable relevance (at least 30% of query words)
                    if relevance_score >= 0.3:
                        candidate['relevance_score'] = relevance_score
                        filtered_candidates.append(candidate)
                
                # Sort by relevance score
                filtered_candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                
                # Take top candidates up to max_results
                results = filtered_candidates[:max_results]
                
                st.write(f"✅ **Pass 2 complete:** {len(results)} relevant candidates selected for CDS analysis")
                
                if len(results) < len(candidates):
                    st.write(f"📊 **Filtered out:** {len(candidates) - len(results)} less relevant candidates")
            else:
                # No quoted terms, just take top results
                results = candidates[:max_results]
                st.write(f"ℹ️ **No quoted terms:** Taking top {len(results)} candidates")
            
            return results
            
        except Exception as e:
            st.error(f"Error searching NCBI: {str(e)}")
            return []
    
    def extract_accession_number(self, link: str, title: str) -> str:
        """Extract accession number from NCBI link or title"""
        try:
            if '/nuccore/' in link:
                parts = link.split('/nuccore/')
                if len(parts) > 1:
                    accession = parts[1].split('/')[0].split('?')[0]
                    return accession
            
            patterns = [
                r'\b([A-Z]{1,2}\d{5,8})\b',
                r'\b([A-Z]{2}_\d{6,9})\b',
                r'\b([A-Z]{3}\d{5})\b',
                r'\b([A-Z]{1}\d{5})\b',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, title)
                if match:
                    return match.group(1)
            
            return ""
        except:
            return ""

    def scrape_ncbi_page(self, accession: str, original_query: str = "") -> Dict:
        """Get NCBI data using structured formats with quote-based filtering"""
        try:
            result = {
                'accession': accession,
                'url': f"{self.base_url}/nuccore/{accession}",
                'success': True,
                'cds_sequences': [],
                'organism': '',
                'definition': '',
                'length': 0,
                'filtered_terms': []
            }
            
            st.write(f"🔍 **Getting structured data for {accession}**")
            
            # Extract quoted terms for filtering
            quoted_terms = self.extract_quoted_terms(original_query)
            if quoted_terms:
                st.write(f"🎯 **Filtering for quoted terms:** {', '.join(quoted_terms)}")
                result['filtered_terms'] = quoted_terms
            
            # Step 1: Get data (GenBank or FASTA format)
            raw_data = self.get_genbank_format(accession)
            
            if raw_data:
                st.write(f"✅ **Data retrieved:** {len(raw_data)} characters")
                
                # Check if we got FASTA format instead of GenBank
                if raw_data.startswith('FASTA_FORMAT\n'):
                    st.write(f"📄 **Processing FASTA format data**")
                    fasta_content = raw_data[13:]  # Remove "FASTA_FORMAT\n" prefix
                    result = self.process_fasta_data(fasta_content, accession)
                    
                    # Apply filtering to FASTA data if needed
                    if quoted_terms and result.get('cds_sequences'):
                        original_count = len(result['cds_sequences'])
                        filtered_cds = []
                        
                        for cds in result['cds_sequences']:
                            # For FASTA, check if the header contains the quoted terms
                            header = cds.get('header', '').lower()
                            definition = result.get('definition', '').lower()
                            
                            matches = any(term in header or term in definition for term in quoted_terms)
                            if matches:
                                filtered_cds.append(cds)
                        
                        result['cds_sequences'] = filtered_cds
                        st.write(f"🎯 **Filtered FASTA results:** {original_count} → {len(filtered_cds)} sequences")
                    
                else:
                    st.write(f"📄 **Processing GenBank format data**")
                    
                    # Extract metadata
                    metadata = self.parse_genbank_metadata(raw_data)
                    result.update(metadata)
                    
                    # Extract ORIGIN sequence
                    origin_sequence = self.extract_origin_from_genbank(raw_data)
                    
                    if origin_sequence:
                        st.write(f"✅ **ORIGIN sequence:** {len(origin_sequence)} bases")
                        st.write(f"**Sample:** {origin_sequence[:50]}...")

                    
                    if origin_sequence:
                        st.write(f"✅ **ORIGIN sequence:** {len(origin_sequence)} bases")
                        st.write(f"**Sample:** {origin_sequence[:50]}...")

                    
                    if origin_sequence:
                        st.write(f"✅ **ORIGIN sequence:** {len(origin_sequence)} bases")
                        st.write(f"**Sample:** {origin_sequence[:50]}...")
                        
                        # Extract CDS features with filtering
                        cds_features = self.parse_cds_features_from_genbank(raw_data, origin_sequence, accession, quoted_terms)
                        
                        # Strict filtering: if quoted terms provided and NO CDS match, reject the entire entry
                        if quoted_terms and not cds_features:
                            st.write(f"❌ **REJECTED:** No CDS matching '{', '.join(quoted_terms)}' - skipping entry**")
                            return {
                                'accession': accession,
                                'success': False,
                                'error': f"No CDS found matching quoted terms: {', '.join(quoted_terms)}",
                                'cds_sequences': [],
                                'filtered_terms': quoted_terms
                            }
                        
                        result['cds_sequences'] = cds_features
                        
                        if cds_features:
                            st.write(f"✅ **Found {len(cds_features)} matching CDS features**")
                            for i, cds in enumerate(cds_features):
                                st.write(f"  - {cds['protein_name']} ({cds['start_position']}-{cds['end_position']}, {cds['length']} bp)")
                        else:
                            st.write("❌ **No CDS features found in GenBank data**")
                    else:
                        st.write("❌ **No ORIGIN sequence found - trying FASTA fallback**")
                        result = self.fallback_fasta_approach(accession)
            else:
                st.write("❌ **Failed to retrieve any data**")
                result = self.fallback_fasta_approach(accession)
            
            return result
            
        except Exception as e:
            st.error(f"Error processing {accession}: {str(e)}")
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
        
        
    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract terms in quotes from the search query"""
        try:
            # Find all terms in double quotes
            quoted_terms = re.findall(r'"([^"]+)"', query)
            
            # Also look for single quotes as backup
            if not quoted_terms:
                quoted_terms = re.findall(r"'([^']+)'", query)
            
            # Clean and normalize terms
            cleaned_terms = []
            for term in quoted_terms:
                cleaned_term = term.strip().lower()
                if cleaned_term:
                    cleaned_terms.append(cleaned_term)
            
            return cleaned_terms
            
        except Exception as e:
            st.write(f"❌ **Error extracting quoted terms:** {str(e)}")
            return []

    def matches_quoted_terms(self, cds_info: Dict, quoted_terms: List[str]) -> bool:
        """Check if a CDS matches any of the quoted terms"""
        if not quoted_terms:
            return True  # No filtering if no quoted terms
        
        try:
            # Fields to check for matches (expanded to include more annotation fields)
            searchable_fields = [
                cds_info.get('gene_name', '').lower(),
                cds_info.get('product', '').lower(),
                cds_info.get('protein_name', '').lower(),
                cds_info.get('locus_tag', '').lower(),
                cds_info.get('note', '').lower(),  # Added note field
                cds_info.get('gene_synonym', '').lower(),  # Added gene synonym
                cds_info.get('function', '').lower()  # Added function field
            ]
            
            # Check if any quoted term matches any field
            for term in quoted_terms:
                for field in searchable_fields:
                    if field and term in field:
                        return True
            
            return False
            
        except Exception as e:
            st.write(f"❌ **Error checking CDS match:** {str(e)}")
            return False  # Changed to False - if error, don't match

    def process_fasta_data(self, fasta_content: str, accession: str) -> Dict:
        """Process FASTA format data"""
        try:
            lines = fasta_content.strip().split('\n')
            if not lines or not lines[0].startswith('>'):
                raise ValueError("Invalid FASTA format")
            
            header = lines[0][1:]  # Remove '>'
            sequence = ''.join(lines[1:]).upper()
            
            # Clean sequence
            sequence = re.sub(r'[^ATGCN]', '', sequence)
            
            if sequence and len(sequence) > 100:  # Reasonable sequence length
                return {
                    'accession': accession,
                    'url': f"{self.base_url}/nuccore/{accession}",
                    'success': True,
                    'cds_sequences': [{
                        'accession': accession,
                        'protein_name': f"Complete_sequence_{accession}",
                        'gene_name': '',
                        'product': 'Complete nucleotide sequence',
                        'locus_tag': '',
                        'start_position': 1,
                        'end_position': len(sequence),
                        'header': f">{header}",
                        'sequence': sequence,
                        'length': len(sequence),
                        'url': f"{self.base_url}/nuccore/{accession}",
                        'valid_dna': True
                    }],
                    'organism': '',
                    'definition': header,
                    'length': len(sequence)
                }
            else:
                raise ValueError("Invalid or too short sequence")
                
        except Exception as e:
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
    
    def get_genbank_format(self, accession: str) -> str:
        """Get GenBank format data directly using correct URLs"""
        try:
            # Use the correct NCBI E-utilities API endpoints that return raw text
            genbank_urls = [
                # E-utilities API - most reliable
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=gb&retmode=text",
                
                # Alternative E-utilities format
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={accession}&rettype=genbank&retmode=text",
                
                # NCBI sviewer with explicit text format
                f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?tool=portal&sendto=on&log$=seqview&db=nuccore&dopt=genbank&sort=&val={accession}&retmode=text",
                
                # Direct nuccore with specific parameters
                f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}?report=genbank&format=text&retmode=text",
            ]
            
            for url in genbank_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        content = response.text
                        
                        # Check if this is actually GenBank format (not HTML)
                        if content.startswith('<?xml') or content.startswith('<!DOCTYPE'):
                            continue
                        
                        # Check if this looks like GenBank format
                        if content.startswith('LOCUS') and ('ORIGIN' in content or 'FEATURES' in content):
                            return content
                        elif 'LOCUS' in content and 'DEFINITION' in content:
                            return content
                        
                except Exception as e:
                    continue
            
            # If all GenBank URLs fail, try getting FASTA directly here
            fasta_urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}?report=fasta&format=text",
            ]
            
            for url in fasta_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        content = response.text
                        if content.startswith('>') and not content.startswith('<?xml'):
                            # Return a special marker so we know this is FASTA
                            return f"FASTA_FORMAT\n{content}"
                        
                except Exception as e:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
        
    def parse_genbank_metadata(self, genbank_data: str) -> Dict:
        """Parse metadata from GenBank format"""
        metadata = {}
        
        try:
            # Extract definition
            def_match = re.search(r'DEFINITION\s+(.*?)(?=\nACCESSION|\nVERSION|\nKEYWORDS)', genbank_data, re.DOTALL)
            if def_match:
                metadata['definition'] = re.sub(r'\s+', ' ', def_match.group(1).strip())
            
            # Extract organism
            organism_match = re.search(r'ORGANISM\s+(.*?)(?=\n\s*REFERENCE|\n\s*COMMENT|\n\s*FEATURES)', genbank_data, re.DOTALL)
            if organism_match:
                organism_text = organism_match.group(1).strip()
                # Get just the first line (species name)
                metadata['organism'] = organism_text.split('\n')[0].strip()
            
            # Extract length from LOCUS line
            locus_match = re.search(r'LOCUS\s+\S+\s+(\d+)\s+bp', genbank_data)
            if locus_match:
                metadata['length'] = int(locus_match.group(1))
            
            return metadata
            
        except Exception as e:
            st.write(f"❌ **Error parsing GenBank metadata:** {str(e)}")
            return {}
    
    def extract_origin_from_genbank(self, genbank_data: str) -> str:
        """Extract ORIGIN sequence from GenBank format"""
        try:
            # Try multiple ORIGIN patterns
            origin_patterns = [
                r'ORIGIN\s*(.*?)(?=//)',                    # Original pattern
                r'ORIGIN\s*(.*?)(?=\n//)',                  # With newline before //
                r'ORIGIN\s*(.*?)$',                         # Until end of string
                r'ORIGIN\s*\n(.*?)(?=//)',                  # With explicit newline after ORIGIN
                r'ORIGIN\s*\n(.*?)(?=\n//)',               # With newlines
                r'ORIGIN[^\n]*\n(.*?)(?=//)',              # Skip ORIGIN line, start from next line
            ]
            
            origin_text = None
            
            for pattern in origin_patterns:
                try:
                    match = re.search(pattern, genbank_data, re.DOTALL)
                    if match:
                        origin_text = match.group(1)
                        break
                except Exception:
                    continue
            
            if not origin_text:
                # Try a simple substring approach
                origin_pos = genbank_data.find('ORIGIN')
                if origin_pos != -1:
                    origin_section = genbank_data[origin_pos:]
                    end_pos = origin_section.find('//')
                    if end_pos != -1:
                        origin_text = origin_section[6:end_pos]  # Skip "ORIGIN"
                else:
                    return ""
            
            if not origin_text:
                return ""
            
            # Clean the sequence
            clean_sequence = ""
            lines = origin_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that don't look like sequence lines
                if not re.search(r'\d+', line):
                    continue
                    
                # Remove line numbers (first token) and keep DNA bases
                parts = line.split()
                if parts:  # Make sure there are parts
                    for part in parts[1:]:  # Skip first part (line number)
                        # Keep only DNA bases
                        dna_bases = re.sub(r'[^ATGCatgcNn]', '', part)
                        clean_sequence += dna_bases.upper()
            
            return clean_sequence
            
        except Exception as e:
            return ""
        



    def parse_cds_features_from_genbank(self, genbank_data: str, origin_sequence: str, accession: str, quoted_terms: List[str] = None) -> List[Dict]:
        """Parse CDS features from GenBank format with optional filtering"""
        cds_sequences = []
        
        try:
            # Find FEATURES section
            features_match = re.search(r'FEATURES\s+Location/Qualifiers\s*(.*?)(?=ORIGIN|CONTIG|//)', genbank_data, re.DOTALL)
            
            if not features_match:
                st.write("❌ **No FEATURES section found**")
                return []
            
            features_text = features_match.group(1)
            st.write(f"✅ **Found FEATURES section:** {len(features_text)} characters")
            
            # Find CDS features
            cds_pattern = r'^\s+CDS\s+(.*?)(?=^\s+\w+\s+|\Z)'
            cds_matches = re.finditer(cds_pattern, features_text, re.MULTILINE | re.DOTALL)
            
            total_cds_found = 0
            filtered_cds_count = 0
            
            for i, match in enumerate(cds_matches):
                try:
                    total_cds_found += 1
                    cds_block = match.group(1)
                    
                    # Extract location - handle simple ranges and joins
                    location_line = cds_block.split('\n')[0].strip()
                    
                    # Parse coordinates
                    coordinates = self.parse_cds_coordinates(location_line)
                    
                    if coordinates:
                        start_pos, end_pos = coordinates[0], coordinates[-1]
                        
                        # Extract sequence
                        if len(origin_sequence) >= end_pos:
                            # For simple ranges, extract directly
                            if len(coordinates) == 2:
                                cds_sequence = origin_sequence[start_pos-1:end_pos]
                            else:
                                # For complex joins, concatenate segments
                                cds_sequence = ""
                                for j in range(0, len(coordinates), 2):
                                    if j+1 < len(coordinates):
                                        seg_start, seg_end = coordinates[j], coordinates[j+1]
                                        cds_sequence += origin_sequence[seg_start-1:seg_end]
                            
                            if cds_sequence:
                                # Extract gene information
                                gene_info = self.extract_gene_info_from_cds_block(cds_block)
                                
                                cds_info = {
                                    'accession': accession,
                                    'protein_name': gene_info.get('protein_name', f"CDS_{i+1}"),
                                    'gene_name': gene_info.get('gene_name', ''),
                                    'product': gene_info.get('product', ''),
                                    'locus_tag': gene_info.get('locus_tag', ''),
                                    'start_position': start_pos,
                                    'end_position': end_pos,
                                    'header': f">{accession}:{start_pos}-{end_pos} {gene_info.get('protein_name', f'CDS_{i+1}')}",
                                    'sequence': cds_sequence,
                                    'length': len(cds_sequence),
                                    'url': f"{self.base_url}/nuccore/{accession}",
                                    'valid_dna': self.is_valid_dna_sequence(cds_sequence)
                                }
                                
                                # Apply filtering if quoted terms are provided
                                if quoted_terms:
                                    if self.matches_quoted_terms(cds_info, quoted_terms):
                                        cds_sequences.append(cds_info)
                                        filtered_cds_count += 1
                                        st.write(f"  ✅ **Matched:** {gene_info.get('protein_name', f'CDS_{i+1}')} ({len(cds_sequence)} bp)")
                                    else:
                                        st.write(f"  ⏭️ **Skipped:** {gene_info.get('protein_name', f'CDS_{i+1}')} (no match)")
                                else:
                                    # No filtering, add all CDS
                                    cds_sequences.append(cds_info)
                                    st.write(f"  ✅ **Added:** {gene_info.get('protein_name', f'CDS_{i+1}')} ({len(cds_sequence)} bp)")
                                    
                except Exception as e:
                    st.write(f"  ❌ **Error processing CDS {i+1}:** {str(e)}")
                    continue
            
            # Summary
            if quoted_terms:
                st.write(f"🎯 **Filtering summary:** {total_cds_found} total CDS → {filtered_cds_count} matching '{', '.join(quoted_terms)}'**")
            else:
                st.write(f"📊 **Total CDS found:** {len(cds_sequences)}")
            
            return cds_sequences
            
        except Exception as e:
            st.write(f"❌ **Error parsing CDS features:** {str(e)}")
            return []
        
    def parse_cds_coordinates(self, location_str: str) -> List[int]:
        """Parse CDS coordinates from location string"""
        try:
            coordinates = []
            
            # Handle simple range: "266..13483"
            simple_match = re.match(r'(\d+)\.\.(\d+)', location_str)
            if simple_match:
                start = int(simple_match.group(1))
                end = int(simple_match.group(2))
                return [start, end]
            
            # Handle join: "join(266..13483,13484..21555)"
            join_match = re.search(r'join\((.*?)\)', location_str)
            if join_match:
                segments = join_match.group(1).split(',')
                for segment in segments:
                    segment = segment.strip()
                    range_match = re.match(r'(\d+)\.\.(\d+)', segment)
                    if range_match:
                        coordinates.extend([int(range_match.group(1)), int(range_match.group(2))])
                return coordinates
            
            # Handle complement: "complement(266..13483)"
            comp_match = re.search(r'complement\((\d+)\.\.(\d+)\)', location_str)
            if comp_match:
                start = int(comp_match.group(1))
                end = int(comp_match.group(2))
                return [start, end]
            
            return []
            
        except Exception as e:
            return []
    
    def extract_gene_info_from_cds_block(self, cds_block: str) -> Dict:
        """Extract gene information from CDS feature block with enhanced fields"""
        gene_info = {
            'protein_name': '',
            'gene_name': '',
            'product': '',
            'locus_tag': '',
            'note': '',
            'gene_synonym': '',
            'function': ''
        }
        
        try:
            # Extract gene name
            gene_match = re.search(r'/gene="([^"]+)"', cds_block)
            if gene_match:
                gene_info['gene_name'] = gene_match.group(1)
                gene_info['protein_name'] = gene_match.group(1)
            
            # Extract product (preferred for protein name)
            product_match = re.search(r'/product="([^"]+)"', cds_block)
            if product_match:
                gene_info['product'] = product_match.group(1)
                gene_info['protein_name'] = product_match.group(1)
            
            # Extract locus tag
            locus_match = re.search(r'/locus_tag="([^"]+)"', cds_block)
            if locus_match:
                gene_info['locus_tag'] = locus_match.group(1)
            
            # Extract note field (often contains descriptive information)
            note_match = re.search(r'/note="([^"]+)"', cds_block)
            if note_match:
                gene_info['note'] = note_match.group(1)
            
            # Extract gene synonym
            synonym_match = re.search(r'/gene_synonym="([^"]+)"', cds_block)
            if synonym_match:
                gene_info['gene_synonym'] = synonym_match.group(1)
            
            # Extract function (if present)
            function_match = re.search(r'/function="([^"]+)"', cds_block)
            if function_match:
                gene_info['function'] = function_match.group(1)
            
            # If no product or gene, try protein_id
            if not gene_info['protein_name']:
                protein_id_match = re.search(r'/protein_id="([^"]+)"', cds_block)
                if protein_id_match:
                    gene_info['protein_name'] = protein_id_match.group(1)
            
            return gene_info
            
        except Exception as e:
            return gene_info
    
    def fallback_fasta_approach(self, accession: str) -> Dict:
        """Fallback approach using FASTA format"""
        try:
            # Try to get FASTA format
            fasta_urls = [
                f"{self.base_url}/nuccore/{accession}?report=fasta&format=text",
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
            ]
            
            for url in fasta_urls:
                try:
                    time.sleep(1)
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code == 200 and response.text.startswith('>'):
                        content = response.text
                        lines = content.split('\n')
                        header = lines[0]
                        sequence = ''.join(lines[1:])
                        
                        if sequence and self.is_valid_dna_sequence(sequence):
                            return {
                                'accession': accession,
                                'url': f"{self.base_url}/nuccore/{accession}",
                                'success': True,
                                'cds_sequences': [{
                                    'accession': accession,
                                    'protein_name': f"Full_sequence_{accession}",
                                    'gene_name': '',
                                    'product': 'Complete sequence',
                                    'locus_tag': '',
                                    'start_position': 1,
                                    'end_position': len(sequence),
                                    'header': header,
                                    'sequence': sequence,
                                    'length': len(sequence),
                                    'url': f"{self.base_url}/nuccore/{accession}",
                                    'valid_dna': True
                                }],
                                'organism': '',
                                'definition': header,
                                'length': len(sequence)
                            }
                            
                except Exception as e:
                    continue
            
            return {
                'accession': accession,
                'success': False,
                'error': 'Both GenBank and FASTA approaches failed',
                'cds_sequences': []
            }
            
        except Exception as e:
            return {
                'accession': accession,
                'success': False,
                'error': str(e),
                'cds_sequences': []
            }
    
    def is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases"""
        if not sequence:
            return False
        return all(base.upper() in 'ATGCN' for base in sequence)
    
    # Keep the existing AI ranking and download methods unchanged
    def ai_select_best_sequences(self, query: str, sequences_with_cds: List[Dict]) -> List[Dict]:
        """Use AI to select the most relevant sequences based on the query"""
        if not self.anthropic or not sequences_with_cds:
            return sequences_with_cds
        
        sequence_summaries = []
        for i, seq_data in enumerate(sequences_with_cds):
            cds_info = []
            for cds in seq_data.get('cds_sequences', []):
                cds_info.append(f"  - {cds.get('protein_name', 'Unknown')} ({cds.get('start_position', 0)}-{cds.get('end_position', 0)}, {cds.get('length', 0)} bp)")
            
            summary = f"""
Sequence {i+1}:
- Accession: {seq_data['accession']}
- Title: {seq_data.get('title', 'N/A')}
- Definition: {seq_data.get('definition', 'N/A')}
- Organism: {seq_data.get('organism', 'N/A')}
- Length: {seq_data.get('length', 0)} bp
- CDS Count: {len(seq_data.get('cds_sequences', []))}
- CDS Details:
{chr(10).join(cds_info) if cds_info else '  - No CDS found'}
"""
            sequence_summaries.append(summary)
        
        context = "\n".join(sequence_summaries)
        
        prompt = f"""
You are a bioinformatics expert. A user is searching for: "{query}"

Below are the available sequences with their CDS information:

{context}

Please analyze these sequences and rank them by relevance to the user's query. Consider:
1. How well the organism/title matches the query
2. The presence and quality of CDS sequences
3. The biological relevance to the query
4. The completeness of the sequence data

Return your response as a JSON list of accession numbers in order of relevance (most relevant first), with a brief explanation for each.

Format: {{"rankings": [{{"accession": "XXX", "rank": 1, "reason": "explanation"}}, ...]}}
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                rankings_data = json.loads(json_match.group())
                rankings = rankings_data.get('rankings', [])
                
                ordered_sequences = []
                for ranking in rankings:
                    accession = ranking['accession']
                    for seq_data in sequences_with_cds:
                        if seq_data['accession'] == accession:
                            seq_data['ai_rank'] = ranking['rank']
                            seq_data['ai_reason'] = ranking['reason']
                            ordered_sequences.append(seq_data)
                            break
                
                ranked_accessions = [r['accession'] for r in rankings]
                for seq_data in sequences_with_cds:
                    if seq_data['accession'] not in ranked_accessions:
                        ordered_sequences.append(seq_data)
                
                return ordered_sequences
            
        except Exception as e:
            logger.error(f"Error in AI ranking: {e}")
        
        return sequences_with_cds
    
    def create_cds_download_data(self, sequences_with_cds: List[Dict]) -> pd.DataFrame:
        """Create downloadable DataFrame with CDS sequences"""
        download_data = []
        
        for seq_data in sequences_with_cds:
            base_info = {
                'Accession': seq_data.get('accession', ''),
                'Title': seq_data.get('title', ''),
                'Definition': seq_data.get('definition', ''),
                'Organism': seq_data.get('organism', ''),
                'Sequence_Length_bp': seq_data.get('length', 0),
                'NCBI_URL': seq_data.get('url', ''),
                'Filtered_Terms': ', '.join(seq_data.get('filtered_terms', [])),
                'AI_Rank': seq_data.get('ai_rank', ''),
                'AI_Reason': seq_data.get('ai_reason', '')
            }
            
            if seq_data.get('cds_sequences'):
                for i, cds in enumerate(seq_data['cds_sequences']):
                    row = base_info.copy()
                    row.update({
                        'CDS_Number': i + 1,
                        'Gene_Name': cds.get('gene_name', ''),
                        'Protein_Name': cds.get('protein_name', ''),
                        'Product': cds.get('product', ''),
                        'Locus_Tag': cds.get('locus_tag', ''),
                        'Start_Position': cds.get('start_position', 0),
                        'End_Position': cds.get('end_position', 0),
                        'CDS_Header': cds.get('header', ''),
                        'CDS_Sequence': cds.get('sequence', ''),
                        'CDS_Length_bp': cds.get('length', 0),
                        'Valid_DNA': cds.get('valid_dna', False),
                        'CDS_URL': cds.get('url', '')
                    })
                    download_data.append(row)
            else:
                row = base_info.copy()
                row.update({
                    'CDS_Number': 0,
                    'Gene_Name': 'No matching CDS found',
                    'Protein_Name': 'No matching CDS found',
                    'Product': '',
                    'Locus_Tag': '',
                    'Start_Position': 0,
                    'End_Position': 0,
                    'CDS_Header': '',
                    'CDS_Sequence': '',
                    'CDS_Length_bp': 0,
                    'Valid_DNA': False,
                    'CDS_URL': ''
                })
                download_data.append(row)
        
        return pd.DataFrame(download_data)


class UniProtSearchEngine:
    """Enhanced UniProt search engine for protein sequences with CDS data"""
    
    def __init__(self):
        self.base_url = "https://www.uniprot.org"
        self.api_url = "https://rest.uniprot.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
    
    def search_protein_sequences(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search UniProt for protein sequences using REST API"""
        try:
            # Prepare search parameters
            params = {
                'query': query,
                'format': 'json',
                'size': max_results,
                'fields': 'accession,id,protein_name,gene_names,organism_name,length,reviewed,xref_refseq,xref_embl,sequence'
            }
            
            url = f"{self.api_url}/uniprotkb/search"
            
            st.write(f"🔍 **Searching UniProt for:** {query}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                st.error(f"UniProt API error: {response.status_code}")
                return []
            
            data = response.json()
            results = data.get('results', [])
            
            st.write(f"✅ **Found {len(results)} UniProt entries**")
            
            # Process results
            processed_results = []
            for result in results:
                try:
                    # Extract basic information
                    accession = result.get('primaryAccession', '')
                    protein_name = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                    if not protein_name:
                        protein_name = result.get('proteinDescription', {}).get('submissionNames', [{}])[0].get('fullName', {}).get('value', '')
                    
                    gene_names = []
                    if 'genes' in result:
                        for gene in result['genes']:
                            if 'geneName' in gene:
                                gene_names.append(gene['geneName']['value'])
                    
                    organism = result.get('organism', {}).get('scientificName', '')
                    sequence = result.get('sequence', {}).get('value', '')
                    length = result.get('sequence', {}).get('length', 0)
                    reviewed = result.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'
                    
                    # Extract cross-references to nucleotide databases
                    nucleotide_refs = []
                    if 'uniProtKBCrossReferences' in result:
                        for xref in result['uniProtKBCrossReferences']:
                            db_name = xref.get('database', '')
                            if db_name in ['EMBL', 'RefSeq']:
                                nucleotide_refs.append({
                                    'database': db_name,
                                    'id': xref.get('id', ''),
                                    'properties': xref.get('properties', [])
                                })
                    
                    processed_results.append({
                        'accession': accession,
                        'protein_name': protein_name,
                        'gene_names': ', '.join(gene_names),
                        'organism': organism,
                        'protein_sequence': sequence,
                        'length': length,
                        'reviewed': reviewed,
                        'nucleotide_refs': nucleotide_refs,
                        'uniprot_url': f"{self.base_url}/uniprotkb/{accession}"
                    })
                    
                except Exception as e:
                    st.write(f"❌ Error processing UniProt entry: {str(e)}")
                    continue
            
            return processed_results
            
        except Exception as e:
            st.error(f"Error searching UniProt: {str(e)}")
            return []
    
    def get_nucleotide_sequences_from_uniprot(self, uniprot_results: List[Dict], quoted_terms: List[str] = None) -> List[Dict]:
        """Extract nucleotide sequences from UniProt cross-references"""
        sequences_with_cds = []
        
        for uniprot_entry in uniprot_results:
            try:
                # Check if this entry matches quoted terms
                if quoted_terms and not self.matches_quoted_terms_uniprot(uniprot_entry, quoted_terms):
                    st.write(f"  ⏭️ **Skipped UniProt {uniprot_entry['accession']}:** No match for quoted terms")
                    continue
                
                st.write(f"🧬 **Processing UniProt {uniprot_entry['accession']}:** {uniprot_entry['protein_name']}")
                
                # Try to get nucleotide sequences from cross-references
                cds_sequences = []
                
                for nucleotide_ref in uniprot_entry.get('nucleotide_refs', []):
                    try:
                        db_name = nucleotide_ref['database']
                        nucleotide_id = nucleotide_ref['id']
                        
                        st.write(f"  🔗 **Checking {db_name} reference:** {nucleotide_id}")
                        
                        # Try to get nucleotide sequence
                        if db_name == 'EMBL':
                            nucleotide_seq = self.get_embl_sequence(nucleotide_id)
                        elif db_name == 'RefSeq':
                            nucleotide_seq = self.get_refseq_sequence(nucleotide_id)
                        else:
                            continue
                        
                        if nucleotide_seq:
                            cds_info = {
                                'accession': nucleotide_id,
                                'protein_name': uniprot_entry['protein_name'],
                                'gene_name': uniprot_entry['gene_names'],
                                'product': uniprot_entry['protein_name'],
                                'locus_tag': uniprot_entry['accession'],
                                'start_position': 1,
                                'end_position': len(nucleotide_seq),
                                'header': f">{nucleotide_id} {uniprot_entry['protein_name']}",
                                'sequence': nucleotide_seq,
                                'length': len(nucleotide_seq),
                                'url': f"https://www.ncbi.nlm.nih.gov/nuccore/{nucleotide_id}",
                                'valid_dna': self.is_valid_dna_sequence(nucleotide_seq),
                                'source_database': db_name,
                                'uniprot_accession': uniprot_entry['accession']
                            }
                            cds_sequences.append(cds_info)
                            st.write(f"    ✅ **Retrieved {db_name} sequence:** {len(nucleotide_seq)} bp")
                        else:
                            st.write(f"    ❌ **Failed to retrieve {db_name} sequence**")
                        
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        st.write(f"    ❌ **Error processing {db_name} reference:** {str(e)}")
                        continue
                
                # If we found nucleotide sequences, add this entry
                if cds_sequences:
                    sequences_with_cds.append({
                        'accession': uniprot_entry['accession'],
                        'title': uniprot_entry['protein_name'],
                        'definition': f"{uniprot_entry['protein_name']} [{uniprot_entry['organism']}]",
                        'organism': uniprot_entry['organism'],
                        'length': max(cds['length'] for cds in cds_sequences),
                        'url': uniprot_entry['uniprot_url'],
                        'success': True,
                        'cds_sequences': cds_sequences,
                        'filtered_terms': quoted_terms or [],
                        'source': 'UniProt'
                    })
                    st.write(f"  ✅ **Added {len(cds_sequences)} CDS sequences from UniProt entry**")
                else:
                    st.write(f"  ❌ **No nucleotide sequences found for UniProt entry**")
                    
            except Exception as e:
                st.write(f"❌ **Error processing UniProt entry {uniprot_entry.get('accession', 'Unknown')}:** {str(e)}")
                continue
        
        return sequences_with_cds
    
    def get_embl_sequence(self, embl_id: str) -> str:
        """Get nucleotide sequence from EMBL database"""
        try:
            # Try NCBI E-utilities (EMBL records are often in NCBI)
            urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={embl_id}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{embl_id}?report=fasta&format=text"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200 and response.text.startswith('>'):
                        lines = response.text.strip().split('\n')
                        sequence = ''.join(lines[1:]).upper()
                        sequence = re.sub(r'[^ATGCN]', '', sequence)
                        if sequence:
                            return sequence
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
    
    def get_refseq_sequence(self, refseq_id: str) -> str:
        """Get nucleotide sequence from RefSeq database"""
        try:
            # RefSeq is available through NCBI
            urls = [
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={refseq_id}&rettype=fasta&retmode=text",
                f"https://www.ncbi.nlm.nih.gov/nuccore/{refseq_id}?report=fasta&format=text"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200 and response.text.startswith('>'):
                        lines = response.text.strip().split('\n')
                        sequence = ''.join(lines[1:]).upper()
                        sequence = re.sub(r'[^ATGCN]', '', sequence)
                        if sequence:
                            return sequence
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
    
    def matches_quoted_terms_uniprot(self, uniprot_entry: Dict, quoted_terms: List[str]) -> bool:
        """Check if a UniProt entry matches quoted terms"""
        if not quoted_terms:
            return True
        
        try:
            searchable_fields = [
                uniprot_entry.get('protein_name', '').lower(),
                uniprot_entry.get('gene_names', '').lower(),
                uniprot_entry.get('organism', '').lower(),
                uniprot_entry.get('accession', '').lower()
            ]
            
            for term in quoted_terms:
                for field in searchable_fields:
                    if field and term in field:
                        return True
            
            return False
            
        except Exception:
            return True
    
    def is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if sequence contains only valid DNA bases"""
        if not sequence:
            return False
        return all(base.upper() in 'ATGCN' for base in sequence)
    
    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract terms in quotes from the search query"""
        try:
            quoted_terms = re.findall(r'"([^"]+)"', query)
            if not quoted_terms:
                quoted_terms = re.findall(r"'([^']+)'", query)
            
            cleaned_terms = []
            for term in quoted_terms:
                cleaned_term = term.strip().lower()
                if cleaned_term:
                    cleaned_terms.append(cleaned_term)
            
            return cleaned_terms
            
        except Exception as e:
            st.write(f"❌ **Error extracting quoted terms:** {str(e)}")
            return []
    
    def ai_select_best_sequences(self, query: str, sequences_with_cds: List[Dict]) -> List[Dict]:
        """Use AI to select the most relevant sequences based on the query"""
        if not self.anthropic or not sequences_with_cds:
            return sequences_with_cds
        
        sequence_summaries = []
        for i, seq_data in enumerate(sequences_with_cds):
            cds_info = []
            for cds in seq_data.get('cds_sequences', []):
                cds_info.append(f"  - {cds.get('protein_name', 'Unknown')} ({cds.get('start_position', 0)}-{cds.get('end_position', 0)}, {cds.get('length', 0)} bp)")
            
            summary = f"""
Sequence {i+1}:
- Accession: {seq_data['accession']}
- Title: {seq_data.get('title', 'N/A')}
- Definition: {seq_data.get('definition', 'N/A')}
- Organism: {seq_data.get('organism', 'N/A')}
- Source: {seq_data.get('source', 'Unknown')}
- Length: {seq_data.get('length', 0)} bp
- CDS Count: {len(seq_data.get('cds_sequences', []))}
- CDS Details:
{chr(10).join(cds_info) if cds_info else '  - No CDS found'}
"""
            sequence_summaries.append(summary)
        
        context = "\n".join(sequence_summaries)
        
        prompt = f"""
You are a bioinformatics expert. A user is searching for: "{query}"

Below are the available sequences with their CDS information from both NCBI and UniProt databases:

{context}

Please analyze these sequences and rank them by relevance to the user's query. Consider:
1. How well the organism/title matches the query
2. The presence and quality of CDS sequences
3. The biological relevance to the query
4. The completeness of the sequence data
5. The source database (NCBI vs UniProt)

Return your response as a JSON list of accession numbers in order of relevance (most relevant first), with a brief explanation for each.

Format: {{"rankings": [{{"accession": "XXX", "rank": 1, "reason": "explanation"}}, ...]}}
"""

        try:
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                rankings_data = json.loads(json_match.group())
                rankings = rankings_data.get('rankings', [])
                
                ordered_sequences = []
                for ranking in rankings:
                    accession = ranking['accession']
                    for seq_data in sequences_with_cds:
                        if seq_data['accession'] == accession:
                            seq_data['ai_rank'] = ranking['rank']
                            seq_data['ai_reason'] = ranking['reason']
                            ordered_sequences.append(seq_data)
                            break
                
                ranked_accessions = [r['accession'] for r in rankings]
                for seq_data in sequences_with_cds:
                    if seq_data['accession'] not in ranked_accessions:
                        ordered_sequences.append(seq_data)
                
                return ordered_sequences
            
        except Exception as e:
            logger.error(f"Error in AI ranking: {e}")
        
        return sequences_with_cds
    
    def create_cds_download_data(self, sequences_with_cds: List[Dict]) -> pd.DataFrame:
        """Create downloadable DataFrame with CDS sequences"""
        download_data = []
        
        for seq_data in sequences_with_cds:
            base_info = {
                'Source_Database': seq_data.get('source', 'Unknown'),
                'Accession': seq_data.get('accession', ''),
                'Title': seq_data.get('title', ''),
                'Definition': seq_data.get('definition', ''),
                'Organism': seq_data.get('organism', ''),
                'Sequence_Length_bp': seq_data.get('length', 0),
                'Database_URL': seq_data.get('url', ''),
                'Filtered_Terms': ', '.join(seq_data.get('filtered_terms', [])),
                'AI_Rank': seq_data.get('ai_rank', ''),
                'AI_Reason': seq_data.get('ai_reason', '')
            }
            
            if seq_data.get('cds_sequences'):
                for i, cds in enumerate(seq_data['cds_sequences']):
                    row = base_info.copy()
                    row.update({
                        'CDS_Number': i + 1,
                        'Gene_Name': cds.get('gene_name', ''),
                        'Protein_Name': cds.get('protein_name', ''),
                        'Product': cds.get('product', ''),
                        'Locus_Tag': cds.get('locus_tag', ''),
                        'Start_Position': cds.get('start_position', 0),
                        'End_Position': cds.get('end_position', 0),
                        'CDS_Header': cds.get('header', ''),
                        'CDS_Sequence': cds.get('sequence', ''),
                        'CDS_Length_bp': cds.get('length', 0),
                        'Valid_DNA': cds.get('valid_dna', False),
                        'Nucleotide_URL': cds.get('url', ''),
                        'Source_DB': cds.get('source_database', ''),
                        'UniProt_Accession': cds.get('uniprot_accession', '')
                    })
                    download_data.append(row)
            else:
                row = base_info.copy()
                row.update({
                    'CDS_Number': 0,
                    'Gene_Name': 'No matching CDS found',
                    'Protein_Name': 'No matching CDS found',
                    'Product': '',
                    'Locus_Tag': '',
                    'Start_Position': 0,
                    'End_Position': 0,
                    'CDS_Header': '',
                    'CDS_Sequence': '',
                    'CDS_Length_bp': 0,
                    'Valid_DNA': False,
                    'Nucleotide_URL': '',
                    'Source_DB': '',
                    'UniProt_Accession': ''
                })
                download_data.append(row)
        
        return pd.DataFrame(download_data)
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

def test_serper_connection(api_key: str) -> Dict:
    """Test SERPER API connection with a simple search"""
    try:
        url = "https://google.serper.dev/search"
        payload = {"q": "test", "num": 1}
        headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "message": "Connection successful"}
        elif response.status_code == 403:
            return {"success": False, "error": "Invalid API key or insufficient permissions"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": "Connection error - unable to reach google.serper.dev"}
    except requests.exceptions.Timeout as e:
        return {"success": False, "error": "Connection timeout"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def test_uniprot_connection() -> Dict:
    """Test UniProt API connection"""
    try:
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": "insulin", "format": "json", "size": 1}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return {"success": True, "message": "UniProt connection successful"}
            else:
                return {"success": False, "error": "Unexpected response format"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": "Connection error - unable to reach UniProt API"}
    except requests.exceptions.Timeout as e:
        return {"success": False, "error": "Connection timeout"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

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
            optimized = nc_stop_codon_optimisation(clean_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
                'Method': method
            }
        #elif method == "JT Plus1 Stop Optimization":
            optimized = JT_Plus1_Stop_Optimized(clean_seq)
            weights, _ = get_codon_weights_row(optimized)
            result = {
                'Original_DNA': clean_seq,
                'Protein': protein_seq,
                'Optimized_DNA': optimized,
                'CAI_Weights': ','.join(f"{w:.4f}" for w in weights),
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

    # Callback for applying CDS from search to mRNA design tab
    def apply_cds_callback():
        if 'temp_selected_cds_for_apply' in st.session_state:
            selected_cds = st.session_state.temp_selected_cds_for_apply
            st.session_state.mrna_design_cds_paste = selected_cds.get('sequence', '')
            st.session_state.mrna_cds_input_method = "Paste Sequence"
            st.session_state.show_apply_success_message = True
            st.session_state.applied_cds_name = selected_cds.get('gene_name', 'Unknown')
            if 'temp_selected_cds_for_apply' in st.session_state:
                del st.session_state.temp_selected_cds_for_apply

    # Apply the selected theme CSS
    inject_app_theme()
    # Initialize research engines
    
    if 'ncbi_engine' not in st.session_state:
        st.session_state.ncbi_engine = NCBISearchEngine()
    if 'uniprot_engine' not in st.session_state:
        st.session_state.uniprot_engine = UniProtSearchEngine()
        
    st.title("🐎 Harmonized Optimization of Oligos and Frames")
    st.markdown("Welcome to HOOF: your DNA sequence optimization and analysis companion!")

    with st.expander("Read Me"):
        st.markdown('''
        ### Optimization Algorithms
        - **Standard Codon Optimization**: This method replaces each codon in your sequence with the most frequently used synonymous codon from the provided codon usage table. This is a straightforward way to potentially increase protein expression levels.
        - **Balanced Optimization**: This algorithm considers both codon usage frequency and the introduction of +1 frameshift-inducing stop codons. It tries to find a balance between using high-frequency codons and strategically placing codons that can terminate out-of-frame translation, which can be beneficial for mRNA vaccine design. The "Bias Weight" slider in the sidebar allows you to control how strongly the algorithm favors introducing these +1 stop codons.
        - **MaxStop**: This method specifically aims to introduce TAA or TAG stop codons in the +1 reading frame. It can perform double substitutions to create stop-stop motifs like TAATAA or TAGTAG.
        

        ### Analysis
        - **+1 Frame Analysis**: This analysis scans the sequence for stop codons in the +1 reading frame. It also includes a feature to scan for known immunogenic peptides in all three reading frames (+0, +1, -1). This is useful for identifying potential off-target immune responses from your translated sequence and its out-of-frame products. 
        - **In-frame Analysis**: This option analyzes the input sequence in its primary reading frame (0). It calculates the Codon Adaptation Index (CAI) for each codon, which is a measure of how well the codon is adapted to the codon usage of a reference organism (in this case, humans by default). It also calculates the GC content of the sequence. This analysis is useful for assessing the baseline quality of your sequence before optimization.

        ### Design Pages
        - **mRNA Design**: This page allows you to design a full mRNA sequence by providing a CDS (Coding Sequence). You can add 5' and 3' UTRs (Untranslated Regions) and a signal peptide. The tool will then generate a complete mRNA sequence and provide a detailed analysis, including a visualization of the sequence, GC content, and a summary of important features.
        - **Cancer Vaccine Design**: This page is a more specialized tool for designing cancer vaccines. It allows you to input a neoantigen sequence and will design a vaccine construct around it. This includes features for optimizing the expression of the neoantigen and adding other elements to enhance the immune response.
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
            # Initialize current selection if not exists
            if 'selected_codon_file' not in st.session_state:
                st.session_state.selected_codon_file = list(available_files.keys())[0]  # First available
            
            # Make sure current selection is still available
            if st.session_state.selected_codon_file not in available_files:
                st.session_state.selected_codon_file = list(available_files.keys())[0]
            
            selected_codon_species = st.selectbox(
                "Select organism codon usage:",
                list(available_files.keys()),
                index=list(available_files.keys()).index(st.session_state.selected_codon_file),
                key="codon_species_selector"
            )
            
            # Check if selection changed
            if selected_codon_species != st.session_state.selected_codon_file:
                st.session_state.selected_codon_file = selected_codon_species
                # Clear existing codon data to force reload
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
        
        # Algorithm settings
        st.subheader("Algorithm Settings")
        bias_weight = st.slider(
            "Bias Weight (Balanced Optimization)", 
            min_value=0.01, 
            max_value=1.0, 
            value=float(st.session_state.config.get("bias_weight", BIAS_WEIGHT_DEFAULT)),
            step=0.1,
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
            optimization_method = st.selectbox(
    "Choose Optimization Method",
    [
        "In-Frame Analysis",           # 1st
        "+1 Frame Analysis",           # 2nd
        "Standard Codon Optimization", # 3rd
        "MaxStop",  # 4th
        "Balanced Optimization",       # 5th
       
    ],
    help="Choose the optimization algorithm to apply"
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
            
            run_optimization_button = st.button("Run Optimization", type="primary")
        
        # Results section - using full width outside of columns
        if run_optimization_button:
            if not sequence_input.strip():
                st.error("Please enter a DNA sequence")
            else:
                with st.spinner("Processing sequence..."):
                    result, error = run_single_optimization(sequence_input, optimization_method, bias_weight)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Optimization completed successfully - Scroll down to see some magical results!")
                    
                    # Full-width results section
                    st.divider()
                    
                    # Display results using full page width
                    if optimization_method == "In-Frame Analysis":
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
                        
                    elif optimization_method == "+1 Frame Analysis":
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
                        st.subheader("Optimization Results")
                        
                        # Show sequence comparison for optimization methods using full width
                        if 'Optimized_DNA' in result:
                            st.subheader("Sequence Comparison")
                            
                            
                            
                            seq_col1, seq_col2 = st.columns(2)
                            
                            with seq_col1:
                                display_copyable_sequence(result['Original_DNA'], "Original Sequence", "orig")
                            with seq_col2:
                                display_copyable_sequence(result['Optimized_DNA'], "Optimized Sequence", "opt")
                            
                            
                        else:
                            # For methods without optimization (like pure analysis)
                            result_data = []
                            for key, value in result.items():
                                if key != 'Method':
                                    result_data.append({'Field': key.replace('_', ' ').title(), 'Value': str(value)})
                            
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(result_df, use_container_width=True)
                            
                         # Use optimized sequence if available, otherwise fallback to original
                        optimized_seq = result['Optimized_DNA'] if 'Optimized_DNA' in result else sequence_input

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
                
                batch_method = st.selectbox(
    "Batch Optimization Method",
    [
        "In-Frame Analysis",           # 1st
        "+1 Frame Analysis",           # 2nd
        "Standard Codon Optimization", # 3rd
        "MaxStop",  # 4th
        "Balanced Optimization",       # 5th
        
    ]
)
                
                if st.button("Process Batch", type="primary"):
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

   #with tab4:
        st.header("mRNA Design")
        st.markdown("Design a full-length mRNA sequence by providing a coding sequence (CDS) and adding UTRs.")

        # Define UTR constants
        JT_5_UTR = "TCGAGCTCGGTACCTAATACGACTCACTATAAGGGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC"
        JT_3_UTR = "CTCGAGCTGGTACTGCATGCACGCAATGCTAGCTGCCCCTTTCCCGTCCTGGGTACCCCGAGTCTCCCCCGACCTCGGGTCCCAGGTATGCTCCCACCTCCACCTGCCCCACTCACCACCTCTGCTAGTTCCAGACACCTCCCAAGCACGCAGCAATGCAGCTCAAAACGCTTAGCCTAGCCACACCCCCACGGGAAACAGCAGTGATTAACCTTTAGCAATAAACGAAAGTTTAACTAAGCTATACTAACCCCAGGGTTGGTCAATTTCGTGCCAGCCACACCCTGGAGCTAGCAAACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTATCATGTCTGGATC"

        SIGNAL_PEPTIDES_DATA = {
            "tPA Signal Peptide": {
                "common_use": "Directs proteins to secretory pathway",
                "sequence_aa": "MDAMKRGLCCVLLLCGAVFVS"
            },
            "IL-2 Signal Peptide": {
                "common_use": "Enhances secretion of cytokines, antigens",
                "sequence_aa": "MYRMQLLSCIALSLALVTNS"
            },
            "Ig κ-chain Signal Peptide": {
                "common_use": "Common in antibody or fusion protein expression",
                "sequence_aa": "METDTLLLWVLLLWVPGSTG"
            },
            "Albumin Signal Peptide": {
                "common_use": "Used for liver-targeted or plasma-secreted proteins",
                "sequence_aa": "MKWVTFISLLLLFSSAYSRGV"
            },
            "Gaussia Luciferase SP": {
                "common_use": "Used in reporter constructs for secreted luciferase",
                "sequence_aa": "MKTIIALSYIFCLVFA"
            },
            "BM40/SPARC Signal Peptide": {
                "common_use": "Common in mRNA vaccines for targeting secretory pathway",
                "sequence_aa": "MGSFSLWLLLLLQSLVAIQG"
            },
            "CD33 Signal Peptide": {
                "common_use": "Used in immune cell-targeted expression",
                "sequence_aa": "MDMVLKVAAVLAGLVSLLVRA"
            },
            "HSA Signal Peptide": {
                "common_use": "Used for hepatocyte-specific mRNA delivery",
                "sequence_aa": "MKWVTFISLLLLFSSAYSRGVFRR"
            },
            "EPO Signal Peptide": {
                "common_use": "Common for erythropoietin or glycoprotein secretion",
                "sequence_aa": "MGVHECPAWLWLLLSLLSLPLGL"
            },
            "Tissue Plasminogen Activator (tPA)": {
                "common_use": "Strong secretory signal; Frequently used in mRNA vaccines (e.g., for spike protein)",
                "sequence_aa": "MDAMKRGLCCVLLLCGAVFVS"
            }
        }

        st.subheader("1. Provide Coding Sequence (CDS)")

        # Initialize session state for mrna_cds_input_method if not already set
        if 'mrna_cds_input_method' not in st.session_state:
            st.session_state.mrna_cds_input_method = "Paste Sequence"

        # Determine the initial index for the radio button
        initial_index = 0 if st.session_state.mrna_cds_input_method == "Paste Sequence" else 1
        cds_input_method = st.radio("Choose CDS input method:", ("Paste Sequence", "Search Database"), key="mrna_cds_input_method", index=initial_index)
        
        # Display success message if it was set by the callback
        if st.session_state.get('show_apply_success_message', False):
            st.success(f"✅ CDS from {st.session_state.get('applied_cds_name', 'Unknown')} applied!")
            st.session_state.show_apply_success_message = False # Reset after showing

        # Ensure cds_sequence always reflects the session state for the text area
        cds_sequence = st.session_state.mrna_design_cds_paste

        if cds_input_method == "Paste Sequence":
            st.text_area("Paste CDS here:", value=st.session_state.mrna_design_cds_paste, height=150, key="mrna_design_cds_paste")
        else:
            # st.markdown("#### Step 1: Search UniProt for Proteins")

            protein_query_mrna = st.text_input(
                "Enter protein search (e.g., 'SARS-CoV-2 spike protein', 'human insulin'):",
                placeholder="Full Name of Protein Makes me Work Better - Yummy",
                key="mrna_protein_search_query"
            )
            
            col_search1_mrna, col_search2_mrna = st.columns([2, 1])
            with col_search1_mrna:
                search_protein_btn_mrna = st.button("🔍 Search UniProt", type="primary", key="mrna_search_protein_btn")
            with col_search2_mrna:
                max_uniprot_results_mrna = st.slider("Max results", 5, 20, 10, key="mrna_max_uniprot_cds")
            
            # Initialize session state for this tab's CDS workflow
            if 'mrna_uniprot_results' not in st.session_state:
                st.session_state.mrna_uniprot_results = []
            if 'mrna_selected_uniprot_entry' not in st.session_state:
                st.session_state.mrna_selected_uniprot_entry = None
            if 'mrna_ncbi_details' not in st.session_state:
                st.session_state.mrna_ncbi_details = None
            if 'mrna_cds_options' not in st.session_state:
                st.session_state.mrna_cds_options = []
            
            # Step 1: UniProt Search Results
            if search_protein_btn_mrna and protein_query_mrna.strip():
                with st.spinner("Searching UniProt..."):
                    try:
                        results = st.session_state.uniprot_engine.search_protein_sequences(protein_query_mrna, max_uniprot_results_mrna)
                        st.session_state.mrna_uniprot_results = results
                        # Reset other states
                        st.session_state.mrna_selected_uniprot_entry = None
                        st.session_state.mrna_ncbi_details = None
                        st.session_state.mrna_cds_options = []
                        
                        if results:
                            st.success(f"✅ Found {len(results)} UniProt entries")
                        else:
                            st.warning("No UniProt entries found. Try different search terms.")
                            
                    except Exception as e:
                        st.error(f"Error searching UniProt: {str(e)}")
            
            # Step 2: Display UniProt Results and Selection
            if st.session_state.mrna_uniprot_results:
                st.markdown("#### Step 2: Select a Protein Entry")
                
                # Create selection dropdown
                uniprot_options_mrna = []
                for i, entry in enumerate(st.session_state.mrna_uniprot_results):
                    option_text = f"{entry['accession']} - {entry['protein_name']} [{entry['organism']}]"
                    uniprot_options_mrna.append(option_text)
                
                selected_uniprot_idx_mrna = st.selectbox(
                    "Choose a UniProt entry:",
                    range(len(uniprot_options_mrna)),
                    format_func=lambda x: uniprot_options_mrna[x],
                    key="mrna_uniprot_selection"
                )
                
                selected_entry_mrna = st.session_state.mrna_uniprot_results[selected_uniprot_idx_mrna]
                st.session_state.mrna_selected_uniprot_entry = selected_entry_mrna
                
                # Display selected entry details
                with st.expander("📋 Selected Entry Details", expanded=True):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**UniProt ID:** {selected_entry_mrna['accession']}")
                        st.write(f"**Protein:** {selected_entry_mrna['protein_name']}")
                        st.write(f"**Genes:** {selected_entry_mrna['gene_names']}")
                    with col_info2:
                        st.write(f"**Organism:** {selected_entry_mrna['organism']}")
                        st.write(f"**Length:** {selected_entry_mrna['length']} aa")
                        st.write(f"**Reviewed:** {'Yes' if selected_entry_mrna['reviewed'] else 'No'}")
                    
                    # Show nucleotide cross-references
                    if selected_entry_mrna['nucleotide_refs']:
                        st.markdown("**🔗 Nucleotide Cross-References:**")
                        for ref in selected_entry_mrna['nucleotide_refs']:
                            st.write(f"- **{ref['database']}:** {ref['id']}")
                
                
                # Step 3: Choose CDS Source
                st.markdown("#### Step 3: Choose CDS Source")

                cds_source_method_mrna = st.radio(
                    "Select how to get the CDS:",
                    ("Search NCBI Databases", "Use UniProt Protein (Reverse Translate - ITS BETTER FOR THE ENVIRONMENT & MY POCKET)"),
                    key="mrna_cds_source_selection"
                )

                if cds_source_method_mrna == "Search NCBI Databases":
                    if selected_entry_mrna['nucleotide_refs']:
                        st.markdown("**🧬 NCBI Database Search**")
                        st.info("This will search for native genomic sequences from NCBI cross-references.")
                        
                        get_ncbi_btn_mrna = st.button("🧬 Get CDS from NCBI", type="primary", key="mrna_get_ncbi_btn")
                        
                        if get_ncbi_btn_mrna:
                            # Try each nucleotide reference
                            ncbi_success = False
                            st.session_state.mrna_cds_options = [] # Clear previous options
                            for ref in selected_entry_mrna['nucleotide_refs']:
                                if ref['database'] in ['EMBL', 'RefSeq']:
                                    accession = ref['id']
                                    
                                    with st.spinner(f"Retrieving CDS information for {accession}..."):
                                        try:
                                            ncbi_details = st.session_state.ncbi_engine.scrape_ncbi_page(accession, protein_query_mrna)
                                            
                                            if ncbi_details['success'] and ncbi_details['cds_sequences']:
                                                st.session_state.mrna_ncbi_details = ncbi_details
                                                st.session_state.mrna_cds_options.extend(ncbi_details['cds_sequences'])
                                                ncbi_success = True
                                            else:
                                                st.warning(f"⚠️ No CDS found in {accession}")
                                                
                                        except Exception as e:
                                            st.error(f"Error retrieving {accession}: {str(e)}")
                                            continue
                            
                            if st.session_state.mrna_cds_options:
                                st.success(f"✅ Retrieved {len(st.session_state.mrna_cds_options)} native CDS features from all cross-references.")
                            else:
                                st.error("❌ Could not retrieve CDS information from any cross-reference.")
                                st.info("💡 **Tip:** Try the 'Use UniProt Protein' option above as an alternative.")
                    
                    else:
                        st.warning("⚠️ **No nucleotide cross-references available** for this UniProt entry.")
                        st.info("💡 **Suggestion:** Use the 'Use UniProt Protein (Reverse Translate - ITS BETTER FOR THE ENVIRONMENT & MY POCKET)' option above.")

                elif cds_source_method_mrna == "Use UniProt Protein (Reverse Translate - ITS BETTER FOR THE ENVIRONMENT & MY POCKET)":
                    st.markdown("**🔄 UniProt Protein Reverse Translation**")
                    st.warning("⚠️ **Note:** This creates a synthetic sequence optimized for human codon usage, not a native genomic sequence.")
                    
                    if selected_entry_mrna.get('protein_sequence'):
                        st.info(f"**Available:** UniProt protein sequence ({selected_entry_mrna['length']} amino acids)")
                        
                        # Show protein sequence preview
                        with st.expander("👀 Preview Protein Sequence", expanded=False):
                            protein_preview = selected_entry_mrna['protein_sequence'][:200]
                            if len(selected_entry_mrna['protein_sequence']) > 200:
                                protein_preview += f"... [+{len(selected_entry_mrna['protein_sequence'])-200} more amino acids]"
                            st.text_area("Protein Sequence:", protein_preview, height=100, key="mrna_protein_preview")
                        
                        use_protein_btn_mrna = st.button("🔄 Reverse Translate Protein Sequence", type="primary", key="mrna_use_protein_btn")
                        
                        if use_protein_btn_mrna:
                            with st.spinner("Reverse translating protein sequence..."):
                                try:
                                    protein_seq = selected_entry_mrna['protein_sequence']
                                    reverse_translated_dna = reverse_translate_highest_cai(protein_seq)
                                    
                                    # Create synthetic CDS entry
                                    synthetic_cds = {
                                        'accession': f"{selected_entry_mrna['accession']}_RT",
                                        'protein_name': selected_entry_mrna['protein_name'],
                                        'gene_name': selected_entry_mrna['gene_names'],
                                        'product': f"{selected_entry_mrna['protein_name']} (reverse translated)",
                                        'locus_tag': selected_entry_mrna['accession'],
                                        'start_position': 1,
                                        'end_position': len(reverse_translated_dna),
                                        'header': f">{selected_entry_mrna['accession']}_RT {selected_entry_mrna['protein_name']} (reverse translated from UniProt)",
                                        'sequence': reverse_translated_dna,
                                        'length': len(reverse_translated_dna),
                                        'url': selected_entry_mrna['uniprot_url'],
                                        'valid_dna': True,
                                        'is_reverse_translated': True,
                                        'original_protein': protein_seq,
                                        'source_type': 'UniProt Reverse Translation'
                                    }
                                    
                                    st.session_state.mrna_cds_options = [synthetic_cds]
                                    st.session_state.mrna_ncbi_details = {
                                        'accession': selected_entry_mrna['accession'],
                                        'success': True,
                                        'cds_sequences': [synthetic_cds],
                                        'is_fallback': True,
                                        'source_type': 'UniProt Reverse Translation'
                                    }
                                    
                                    st.success("✅ Successfully reverse-translated protein sequence!")
                                    
                                except Exception as e:
                                    st.error(f"Error during reverse translation: {str(e)}")
                    
                    else:
                        st.error("❌ No protein sequence available from UniProt entry")

            # Step 4: CDS Selection and Application
            if st.session_state.mrna_cds_options:
                st.markdown("#### Step 4: Select CDS to Use")
                
                cds_dropdown_options_mrna = [f"{cds.get('gene_name', 'Unknown')} | {cds.get('product', 'Unknown')} | {cds.get('length', 0)} bp" for cds in st.session_state.mrna_cds_options]
                selected_cds_idx_mrna = st.selectbox(
                    "Choose a CDS feature:",
                    range(len(cds_dropdown_options_mrna)),
                    format_func=lambda x: cds_dropdown_options_mrna[x],
                    key="mrna_cds_selection"
                )
                selected_cds_mrna = st.session_state.mrna_cds_options[selected_cds_idx_mrna]
                
                # Temporarily store the selected CDS for the callback
                st.session_state.temp_selected_cds_for_apply = selected_cds_mrna

                st.button(
                    "Apply this CDS to mRNA Design", 
                    key="mrna_apply_cds",
                    on_click=apply_cds_callback
                )

        st.subheader("2. Add Signal Peptide (Optional)")
        add_signal_peptide = st.checkbox("Add Signal Peptide to mRNA", key="add_signal_peptide_checkbox")

        if add_signal_peptide:
            signal_peptide_names = list(SIGNAL_PEPTIDES_DATA.keys())
            selected_signal_peptide_name = st.selectbox(
                "Select a Signal Peptide:",
                signal_peptide_names,
                key="signal_peptide_selection"
            )
            
            selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
            st.info(f"**Common Use:** {selected_sp_info['common_use']}\n\n**Amino Acid Sequence:** {selected_sp_info['sequence_aa']}")

        st.subheader("3. Sequence Processing Options")

        # GC Content Correction
        st.markdown("**GC Content Correction**")
        gc_correction_enabled = st.checkbox("Enable GC content correction (if > 70%)")
        st.info("This will attempt to lower the GC content of the CDS to between 55-70% using synonymous codons.")
        
        # Codon Optimization
        st.markdown("**Codon Optimization**")
        optimization_method_mrna = st.selectbox(
            "Choose an optimization method for the CDS:",
            ["None", "Standard Codon Optimization", "Balanced Optimization", "MaxStop"],
            key="mrna_design_optimization"
        )

        # Stop Codon Selection
        st.markdown("**Stop Codon Selection**")
        STOP_CODONS = ["TAA", "TAG", "TGA"]
        selected_stop_codon = st.selectbox(
            "Select Stop Codon to append:",
            STOP_CODONS,
            key="mrna_design_stop_codon"
        )

        # ADD THIS ENTIRE SECTION HERE - RIGHT AFTER STOP CODON SELECTION:
        st.markdown("**3' Peptide Tag (Optional)**")
        add_3prime_tag = st.checkbox("Add 3' peptide tag to mRNA", key="add_3prime_tag")

        if add_3prime_tag:
            TAG_PEPTIDES = {
                "His-Tag (6x)": {
                    "sequence_aa": "HHHHHH",
                    "purpose": "Protein purification and detection"
                },
                "His-Tag (8x)": {
                    "sequence_aa": "HHHHHHHH", 
                    "purpose": "Enhanced protein purification"
                },
                "FLAG Tag": {
                    "sequence_aa": "DYKDDDDK",
                    "purpose": "Protein detection and purification"
                },
                "V5 Tag": {
                    "sequence_aa": "GKPIPNPLLGLDST",
                    "purpose": "Protein detection and immunoprecipitation"
                },
                "Myc Tag": {
                    "sequence_aa": "EQKLISEEDL", 
                    "purpose": "Protein detection and localization"
                },
                "HA Tag": {
                    "sequence_aa": "YPYDVPDYA",
                    "purpose": "Protein detection and purification"
                },
                "Strep-Tag II": {
                    "sequence_aa": "WSHPQFEK",
                    "purpose": "Gentle protein purification"
                },
                "Custom Peptide": {
                    "sequence_aa": "",
                    "purpose": "User-defined peptide sequence"
                }
            }
            
            selected_tag_name = st.selectbox(
                "Select 3' peptide tag:",
                list(TAG_PEPTIDES.keys()),
                key="tag_selection"
            )
            
            selected_tag_info = TAG_PEPTIDES[selected_tag_name]
            
            if selected_tag_name == "Custom Peptide":
                custom_tag_sequence = st.text_area(
                    "Enter custom peptide sequence (amino acids):",
                    placeholder="EXAMPLE",
                    key="custom_tag_sequence"
                )
                tag_sequence_aa = custom_tag_sequence.strip().upper()
            else:
                tag_sequence_aa = selected_tag_info["sequence_aa"]
                st.info(f"**Purpose:** {selected_tag_info['purpose']}\n\n**Amino Acid Sequence:** {tag_sequence_aa}")
            
            # Tag linker options
            use_tag_linker = st.checkbox("Add linker before 3' tag", value=False, key="use_tag_linker")
            
            if use_tag_linker:
                tag_linker_options = {
                    "Flexible (GGGGS)": "GGGGS",
                    "Flexible (GGS)": "GGS", 
                    "Rigid (EAAAK)": "EAAAK",
                    "Short (GG)": "GG"
                }
                
                selected_tag_linker = st.selectbox(
                    "Select tag linker:",
                    list(tag_linker_options.keys()),
                    key="tag_linker_selection"
                )
                
                tag_linker_aa = tag_linker_options[selected_tag_linker]
            else:
                tag_linker_aa = ""


        st.subheader("4. Design mRNA")
        if st.button("Design mRNA Sequence", type="primary"):
            if not cds_sequence.strip():
                st.error("Please provide a CDS sequence first.")
            else:
                is_valid, clean_cds, error_msg = validate_dna_sequence(cds_sequence)
                if not is_valid:
                    st.error(error_msg)
                else:
                    processed_cds = clean_cds
                    
                    # Step 1: GC Content Correction
                    if gc_correction_enabled:
                        with st.spinner("Adjusting GC content..."):
                            initial_gc = calculate_gc_content(processed_cds)
                            if initial_gc > 70.0:
                                processed_cds = adjust_gc_content(processed_cds, max_gc=75.0, min_gc=55.0)
                            else:
                                st.info(f"Initial GC content ({initial_gc:.1f}%) is not above 70%. No correction applied.")
                            
                            with st.spinner("Enforcing local GC content (10bp windows)..."):
                                processed_cds = enforce_local_gc_content(processed_cds, target_max_gc=75.0, window_size=25, step_size=1)

                    # Step 2: Codon Optimization
                    if optimization_method_mrna != "None":
                        with st.spinner(f"Applying {optimization_method_mrna}..."):
                            protein_seq = translate_dna(processed_cds)
                            if optimization_method_mrna == "Standard Codon Optimization":
                                processed_cds = codon_optimize(protein_seq)
                            elif optimization_method_mrna == "Balanced Optimization":
                                processed_cds = balanced_optimisation(processed_cds)
                            elif optimization_method_mrna == "MaxStop":
                                processed_cds = nc_stop_codon_optimisation(processed_cds)
                            #elif optimization_method_mrna == "JT Plus1 Stop Optimization":
                                processed_cds = JT_Plus1_Stop_Optimized(processed_cds)
                            st.success(f"Successfully applied {optimization_method_mrna}.")

                # In the "Design mRNA Sequence" button logic, after assembling the main CDS:
                    # Step 3: Assemble the full CDS, handling the signal peptide correctly
                    dna_signal_peptide = ""
                    main_cds = processed_cds

                    if add_signal_peptide:
                        selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
                        selected_sp_aa_seq = selected_sp_info["sequence_aa"]
                        dna_signal_peptide = reverse_translate_highest_cai(selected_sp_aa_seq)
                        
                        # Remove the ATG from the main CDS only if a signal peptide is added
                        if main_cds.upper().startswith("ATG"):
                            main_cds = main_cds[3:]
                            st.info("Removed ATG start codon from main CDS because a signal peptide was added.")

                    # Step 3.5: Add 3' peptide tag if selected
                    tag_sequence_dna = ""
                    if add_3prime_tag and tag_sequence_aa:
                        if use_tag_linker and tag_linker_aa:
                            full_tag_aa = tag_linker_aa + tag_sequence_aa
                        else:
                            full_tag_aa = tag_sequence_aa
                        
                        # Reverse translate the tag
                        tag_sequence_dna = reverse_translate_highest_cai(full_tag_aa)
                        st.info(f"Added 3' peptide tag: {full_tag_aa}")

                    # The full coding sequence is: signal peptide + main CDS + 3' tag
                    full_cds = dna_signal_peptide + main_cds + tag_sequence_dna

                    # Step 4: Handle stop codons
                    STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}
                    last_codon = full_cds[-3:].upper()

                    if last_codon in STANDARD_STOP_CODONS:
                        # If there's already a stop codon, replace it with the selected double stop
                        cds_with_stops = full_cds[:-3] + (selected_stop_codon * 2)
                        st.info(f"Replaced existing stop codon with selected double stop codon: {selected_stop_codon * 2}")
                    else:
                        # If there's no stop codon, append the selected double stop
                        cds_with_stops = full_cds + (selected_stop_codon * 2)
                        st.info(f"Added selected double stop codon: {selected_stop_codon * 2}")
                        # ✅ Run validation on the final full CDS before adding UTRs
                        validation_problems = validate_final_coding_sequence(cds_with_stops)

                        st.subheader("🧪 mRNA Design Validation")
                        if validation_problems:
                            for problem in validation_problems:
                                st.error(problem)
                            st.warning("⚠️ Please fix the issues above before using this mRNA sequence in production.")
                        else:
                            st.success("✅ CDS passed all validation checks: proper start codon, no internal stops, ends with exactly two stop codons.")

                    

                    # Step 5: Assemble and display the final mRNA sequence
                    final_mrna_sequence = JT_5_UTR + cds_with_stops + JT_3_UTR
                    st.subheader("✅ Final mRNA Sequence")

                    # For display, we pass the components separately to be colored correctly
                    main_cds_for_display = main_cds  # Just the main CDS without signal peptide and tag
                    tag_for_display = tag_sequence_dna + (selected_stop_codon * 2)  # Include stop codons with tag

                    display_colored_mrna_sequence(
                        utr5_seq=JT_5_UTR, 
                        cds_seq=main_cds_for_display,
                        utr3_seq=JT_3_UTR, 
                        signal_peptide_seq=dna_signal_peptide, 
                        tag_sequence_seq=tag_for_display,  # Now includes the stop codons
                        key_suffix="final_mrna"
                    )

                    # Step 4: Assemble and display the final mRNA sequence
                    final_mrna_sequence = JT_5_UTR + cds_with_stops + JT_3_UTR
                    st.subheader("✅ Final mRNA Sequence")
                    
                    # For display, we pass the components separately to be colored correctly
                    
                    # For display, we need to separate the components
                    main_cds_for_display = main_cds  # Just the main CDS
                    
                  
       


                    # Step 5: Perform and display the final analysis
                    st.subheader("📊 Final Analysis")

                    # The context for frame analysis must include the 5' UTR to find the junctional ACCATG
                    analysis_context_sequence = JT_5_UTR + full_cds

                    # Detailed stats table (using the full, correct CDS)
                    summary_df = generate_detailed_mrna_summary(full_cds, final_mrna_sequence, JT_5_UTR, JT_3_UTR)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    epitope_df = load_immunogenic_peptides()

                    if not epitope_df.empty:
                        st.subheader("🔬 Immunogenic Peptide Scanning (mRNA Design)")
                        
                        # Translate +1 and -1 frames of the full CDS
                        plus1_protein = translate_frame(full_cds, 1)  # +1 frame
                        minus1_protein = translate_frame(full_cds, 2)  # -1 frame (offset by 2 to get -1)
                        
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
                            st.warning(f"⚠️ **mRNA DESIGN ALERT**: Found {total_findings} immunogenic peptides in alternative reading frames!")
                            
                            # Create detailed summary
                            summary_df_epitopes = create_immunogenic_peptide_summary(plus1_findings, minus1_findings)
                            if summary_df_epitopes is not None:
                                st.subheader("📋 Detailed Epitope Findings")
                                st.dataframe(summary_df_epitopes, use_container_width=True, hide_index=True)
                                
                                # Show frame-specific details
                                if plus1_findings:
                                    with st.expander(f"🔍 +1 Frame Epitopes ({len(plus1_findings)} found)", expanded=True):
                                        for i, finding in enumerate(plus1_findings, 1):
                                            st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}")
                                
                                if minus1_findings:
                                    with st.expander(f"🔍 -1 Frame Epitopes ({len(minus1_findings)} found)", expanded=True):
                                        for i, finding in enumerate(minus1_findings, 1):
                                            st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}")
                                
                                # Download button for epitope findings
                                excel_data = create_download_link(summary_df_epitopes, f"mRNA_Design_Immunogenic_Peptides_{len(summary_df_epitopes)}.xlsx")
                                st.download_button(
                                    label="📥 Download mRNA Epitope Findings (Excel)",
                                    data=excel_data,
                                    file_name=f"mRNA_Design_Immunogenic_Peptides_{len(summary_df_epitopes)}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Download complete list of found immunogenic peptides in mRNA design"
                                )
                        else:
                            st.success("✅ **Good news**: No known immunogenic peptides found in +1 or -1 reading frames of your mRNA design!")

                    else:
                        st.info("ℹ️ Immunogenic peptide scanning disabled - epitope_table_export.xlsx not found")

                    # Top row: CAI/GC and +1 Stop Pie Chart
                    col_chart1, col_chart2 = st.columns([3, 1])

                    # Find this section in tab4 (mRNA Design):
                    with col_chart1:
                        st.markdown("##### CDS CAI and GC Content")
                        cai_result, cai_error = run_single_optimization(full_cds, "In-Frame Analysis")
                        if not cai_error and cai_result:
                            cai_df = pd.DataFrame(cai_result)
                            
                            # Replace the existing chart with this:
                            positions = cai_df['Position'].tolist()
                            cai_weights = cai_df['CAI_Weight'].tolist()
                            amino_acids = cai_df['Amino_Acid'].tolist()
                            plus1_stop_positions = get_plus1_stop_positions(full_cds)
                            minus1_stop_positions = get_minus1_stop_positions(full_cds)
                            slippery_positions = get_slippery_motif_positions(full_cds)
                            
                            display_stateful_overlay_chart(
                                positions=positions,
                                cai_weights=cai_weights,
                                amino_acids=amino_acids,
                                sequence=full_cds,
                                seq_name="Processed CDS",
                                plus1_stop_positions=plus1_stop_positions,
                                minus1_stop_positions=minus1_stop_positions,
                                slippery_positions=slippery_positions
                            )
                        else:
                            st.warning("Could not generate CAI/GC plot.")

                    with col_chart2:
                        st.markdown("##### CDS +1 Stop Codons")
                        plus1_stops = number_of_plus1_stops(full_cds)
                        if plus1_stops['total'] > 0:
                            stop_labels = ['TAA', 'TAG', 'TGA']
                            stop_values = [plus1_stops['TAA'], plus1_stops['TAG'], plus1_stops['TGA']]
                            fig_pie = create_interactive_pie_chart(stop_values, stop_labels, "+1 Stop Codon Distribution")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("No +1 stop codons found in the processed CDS.")

                    # Bottom row: full-width visualization
                    st.markdown("---")
                    st.markdown("##### Final mRNA Visualisation")
                    create_geneious_like_visualization(
                        utr5_seq=JT_5_UTR, 
                        cds_seq=main_cds_for_display,
                        utr3_seq=JT_3_UTR, 
                        signal_peptide_seq=dna_signal_peptide,
                        tag_sequence_seq=tag_sequence_dna,  # ADD THIS LINE
                        key_suffix="final_mrna"
                    )
                    
             

                            
   # with tab5:
            st.header("Cancer Vaccine Design")
            st.markdown("Design a personalized cancer vaccine by combining multiple peptides with appropriate linkers")
            
            # Step 1: Select number of peptides
            st.subheader("1. Define Vaccine Components")
            num_peptides = st.number_input("Number of peptides in vaccine", min_value=1, max_value=10, value=3)
            
            # Initialize peptide inputs dictionary if not in session state
            if 'cancer_vaccine_peptides' not in st.session_state:
                st.session_state.cancer_vaccine_peptides = {}
            
            # Step 2: Input peptide sequences
            st.markdown("#### Peptide Sequences")
            peptide_sequences = []
            
            for i in range(1, int(num_peptides) + 1):
                key = f"peptide_{i}"
                if key not in st.session_state.cancer_vaccine_peptides:
                    st.session_state.cancer_vaccine_peptides[key] = ""
                
                st.session_state.cancer_vaccine_peptides[key] = st.text_area(
                    f"Peptide {i} (amino acid sequence):",
                    value=st.session_state.cancer_vaccine_peptides[key],
                    height=80,
                    key=f"cancer_peptide_{i}"
                )
                
                if st.session_state.cancer_vaccine_peptides[key]:
                    peptide_sequences.append(st.session_state.cancer_vaccine_peptides[key])
            
            # Step 3: Select linker
            st.markdown("#### Linker Selection")
            
            LINKER_OPTIONS = {
                "(G₄S)n linker": {
                    "sequence_aa": "GGGGS",
                    "type": "Flexible",
                    "purpose": "Most widely used linker; adds flexibility between domains to reduce steric clash"
                },
                "EAAAK linker": {
                    "sequence_aa": "EAAAK",
                    "type": "Rigid (helical)",
                    "purpose": "Promotes α-helix formation; keeps domains structurally separate"
                },
                "HE Linker": {
                    "sequence_aa": "HEHEHE",
                    "type": "Rigid",
                    "purpose": "Promotes hydrophilic spacing; sometimes used for helical separation"
                },
                "AP linker": {
                    "sequence_aa": "AEAAAKA",
                    "type": "Rigid",
                    "purpose": "Engineered rigid helix; used for mechanical separation of domains"
                },
                "(XP)n linker": {
                    "sequence_aa": "GPGPG",
                    "type": "Flexible/Spacer",
                    "purpose": "T cell epitope spacers (seen in multi-epitope vaccines)"
                },
                "AAY linker": {
                    "sequence_aa": "AAY",
                    "type": "Cleavable",
                    "purpose": "Used in epitope fusion vaccines; recognized by immunoproteasome"
                },
                "GPGPG linker": {
                    "sequence_aa": "GPGPG",
                    "type": "Flexible",
                    "purpose": "Used in multi-epitope vaccine constructs; promotes better MHC presentation"
                },
                "RRRRRR linker": {
                    "sequence_aa": "RRRRRR",
                    "type": "Cell-penetrating",
                    "purpose": "Enhances delivery, e.g., in peptide-based vaccines or intracellular targeting"
                },
                "KFERQ linker": {
                    "sequence_aa": "KFERQ",
                    "type": "Degron motif",
                    "purpose": "Targets proteins for lysosomal degradation (used in autophagy or clearance therapy)"
                },
                "ENLYFQG (TEV site)": {
                    "sequence_aa": "ENLYFQG",
                    "type": "Protease site",
                    "purpose": "Cleavable linker for conditional release (TEV protease)"
                },
                "LVPRGS (Thrombin site)": {
                    "sequence_aa": "LVPRGS",
                    "type": "Protease site",
                    "purpose": "Used to cleave fusion tags (e.g., tag–protein constructs)"
                }
            }
            
            # Format linker options for display
            linker_names = list(LINKER_OPTIONS.keys())
            
            selected_linker_name = st.selectbox(
                "Select Linker:",
                linker_names,
                key="cancer_vaccine_linker_selection"
            )
            
            selected_linker_info = LINKER_OPTIONS[selected_linker_name]
            st.info(f"**Type:** {selected_linker_info['type']}\n\n**Purpose:** {selected_linker_info['purpose']}\n\n**Amino Acid Sequence:** {selected_linker_info['sequence_aa']}")
            
            linker_repeats = 1
            if selected_linker_name in ["(G₄S)n linker", "EAAAK linker", "(XP)n linker"]:
                linker_repeats = st.slider("Number of linker repeats:", 1, 5, 1, key="cancer_linker_repeats")
            
            # Step 4: Signal Peptide Selection
            st.markdown("#### Signal Peptide Selection")
            
            signal_peptide_names = list(SIGNAL_PEPTIDES_DATA.keys())
            selected_signal_peptide_name = st.selectbox(
                "Select a Signal Peptide:",
                signal_peptide_names,
                key="cancer_signal_peptide_selection"
            )
            
            selected_sp_info = SIGNAL_PEPTIDES_DATA[selected_signal_peptide_name]
            st.info(f"**Common Use:** {selected_sp_info['common_use']}\n\n**Amino Acid Sequence:** {selected_sp_info['sequence_aa']}")
            
            # Step 5: MITD Option
            st.markdown("#### MITD Option")
            add_mitd = st.checkbox("Add Membrane-Interacting Transport Domain (MITD)", key="cancer_add_mitd")
            if add_mitd:
                st.info("MITD sequence (STQALNTVYTKLNIRLRQGRTLYTILNLA) will be added after the last peptide")
            
            # Step 6: Optimization Options
            st.subheader("2. Sequence Processing Options")
            
            # GC Content Correction
            st.markdown("**GC Content Correction**")
            gc_correction_enabled = st.checkbox("Enable GC content correction (if > 70%)", key="cancer_gc_correction")
            
            # Codon Optimization
            st.markdown("**Codon Optimization**")
            optimization_method_cancer = st.selectbox(
                "Choose an optimization method for the CDS:",
                ["None", "Standard Codon Optimization", "Balanced Optimization", "MaxStop"],
                key="cancer_optimization"
            )
            
            # Stop Codon Selection
            st.markdown("**Stop Codon Selection**")
            STOP_CODONS = ["TAA", "TAG", "TGA"]
            selected_stop_codon = st.selectbox(
                "Select Stop Codon to append:",
                STOP_CODONS,
                key="cancer_stop_codon"
            )
            
            
            
            # Step 7: Design Vaccine Button
            st.subheader("3. Design Cancer Vaccine")
            design_vaccine_btn = st.button("Design Cancer Vaccine", type="primary", key="design_cancer_vaccine_btn")
            
            if design_vaccine_btn:
                if len(peptide_sequences) < 1:
                    st.error("Please provide at least one peptide sequence.")
                else:
                    with st.spinner("Designing cancer vaccine..."):
                        # Generate full amino acid sequence
                        # 1. Start with signal peptide
                        full_aa_sequence = selected_sp_info['sequence_aa']
                        
                        # 2. Add peptides with linkers
                        linker_aa_sequence = selected_linker_info['sequence_aa'] * linker_repeats
                        
                        for i, peptide in enumerate(peptide_sequences):
                            # Clean peptide (remove spaces, etc.)
                            clean_peptide = peptide.strip().upper()
                            
                            # Add peptide
                            full_aa_sequence += clean_peptide
                            
                            # Add linker after all peptides except the last one
                            if i < len(peptide_sequences) - 1:
                                full_aa_sequence += linker_aa_sequence
                        
                        # 3. Add MITD if selected
                        if add_mitd:
                            mitd_aa_sequence = "STQALNTVYTKLNIRLRQGRTLYTILNLA"
                            full_aa_sequence += mitd_aa_sequence
                        
                        # Reverse translate to nucleotide sequence
                        full_cds = reverse_translate_highest_cai(full_aa_sequence)
                        
                        # Process CDS (GC content correction and optimization)
                        processed_cds = full_cds
                        
                        # Step 1: GC Content Correction
                        if gc_correction_enabled:
                            initial_gc = calculate_gc_content(processed_cds)
                            if initial_gc > 70.0:
                                processed_cds = adjust_gc_content(processed_cds, max_gc=75.0, min_gc=55.0)
                            else:
                                st.info(f"Initial GC content ({initial_gc:.1f}%) is not above 70%. No correction applied.")
                            
                            # Enforce local GC content
                            processed_cds = enforce_local_gc_content(processed_cds, target_max_gc=75.0, window_size=25, step_size=1)
                        
                        # Step 2: Codon Optimization
                        if optimization_method_cancer != "None":
                            protein_seq = translate_dna(processed_cds)
                            if optimization_method_cancer == "Standard Codon Optimization":
                                processed_cds = codon_optimize(protein_seq)
                            elif optimization_method_cancer == "Balanced Optimization":
                                processed_cds = balanced_optimisation(processed_cds)
                            elif optimization_method_cancer == "MaxStop":
                                processed_cds = nc_stop_codon_optimisation(processed_cds)
                            #elif optimization_method_cancer == "JT Plus1 Stop Optimization":
                                processed_cds = JT_Plus1_Stop_Optimized(processed_cds)
                        
                        # Add stop codons
                        STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}
                        last_codon = processed_cds[-3:].upper() if len(processed_cds) >= 3 else ""
                        if last_codon in STANDARD_STOP_CODONS:
                            # If the sequence already ends in a stop, replace it with the selected double stop
                            cds_with_stops = processed_cds[:-3] + (selected_stop_codon * 2)
                        else:
                            # Otherwise, append the double stop codon
                            cds_with_stops = processed_cds + (selected_stop_codon * 2)
                        
                        # Assemble final mRNA sequence
                        final_mrna_sequence = JT_5_UTR + cds_with_stops + JT_3_UTR
                        
                        # Display the results
                        st.subheader("✅ Cancer Vaccine mRNA Sequence")
                        
                        # For display only - calculate the signal peptide DNA sequence
                        signal_peptide_dna = reverse_translate_highest_cai(selected_sp_info['sequence_aa'])
                        
                        # Display colored sequence
                        display_colored_mrna_sequence(
                            utr5_seq=JT_5_UTR,
                            cds_seq=cds_with_stops[len(signal_peptide_dna):],  # Main CDS without signal peptide
                            utr3_seq=JT_3_UTR,
                            signal_peptide_seq=signal_peptide_dna,
                            key_suffix="cancer_vaccine"
                        )
                        
                        # Display components
                        st.subheader("📋 Vaccine Components")
                        
                        components_data = {
                            "Component": ["5' UTR", "Signal Peptide"],
                            "Type": ["Regulatory Element", "Targeting Sequence"],
                            "Length (aa)": ["N/A", len(selected_sp_info['sequence_aa'])],
                            "Length (bp)": [len(JT_5_UTR), len(signal_peptide_dna)]
                        }
                        
                        # Add peptides and linkers
                        for i, peptide in enumerate(peptide_sequences):
                            clean_peptide = peptide.strip().upper()
                            components_data["Component"].append(f"Peptide {i+1}")
                            components_data["Type"].append("Antigen")
                            components_data["Length (aa)"].append(len(clean_peptide))
                            components_data["Length (bp)"].append(len(clean_peptide) * 3)  # Approximate
                            
                            if i < len(peptide_sequences) - 1:
                                components_data["Component"].append(f"Linker {i+1}")
                                components_data["Type"].append(selected_linker_info['type'])
                                components_data["Length (aa)"].append(len(selected_linker_info['sequence_aa']) * linker_repeats)
                                components_data["Length (bp)"].append(len(selected_linker_info['sequence_aa']) * linker_repeats * 3)  # Approximate
                        
                        # Add MITD if selected
                        if add_mitd:
                            components_data["Component"].append("MITD")
                            components_data["Type"].append("Transport Domain")
                            components_data["Length (aa)"].append(30)  # Length of MITD
                            components_data["Length (bp)"].append(90)  # 30 aa * 3 bp/aa
                        
                        # Add 3' UTR
                        components_data["Component"].append("3' UTR")
                        components_data["Type"].append("Regulatory Element")
                        components_data["Length (aa)"].append("N/A")
                        components_data["Length (bp)"].append(len(JT_3_UTR))
                        
                        # Display components table
                        components_df = pd.DataFrame(components_data)
                        st.dataframe(components_df, use_container_width=True, hide_index=True)
                        
                        # Perform final analysis
                        st.subheader("📊 Final Analysis")

                        # The context for frame analysis must include the 5' UTR to find the junctional ACCATG
                        analysis_context_sequence = JT_5_UTR + processed_cds

                        # Detailed stats table (using the full, correct CDS)
                        summary_df = generate_detailed_mrna_summary(processed_cds, final_mrna_sequence, JT_5_UTR, JT_3_UTR)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                        # ADD IMMUNOGENIC PEPTIDE SCANNING HERE
                        epitope_df = load_immunogenic_peptides()

                        if not epitope_df.empty:
                            st.subheader("🔬 Immunogenic Peptide Scanning (Cancer Vaccine)")
                            
                            # Translate +1 and -1 frames of the processed CDS
                            plus1_protein = translate_frame(processed_cds, 1)  # +1 frame
                            minus1_protein = translate_frame(processed_cds, 2)  # -1 frame (offset by 2 to get -1)
                            
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
                                st.warning(f"⚠️ **VACCINE DESIGN ALERT**: Found {total_findings} immunogenic peptides in alternative reading frames!")
                                
                                # Show which input peptides might be problematic
                                st.markdown("**🔍 Analysis of Input Peptides vs Found Epitopes:**")
                                
                                # Check if any of the found epitopes overlap with the input peptides
                                input_peptides_text = " ".join([peptide.strip().upper() for peptide in peptide_sequences])
                                overlapping_epitopes = []
                                
                                for finding in plus1_findings + minus1_findings:
                                    epitope = finding['epitope']
                                    if epitope in input_peptides_text:
                                        overlapping_epitopes.append(finding)
                                
                                if overlapping_epitopes:
                                    st.error(f"🚨 **CRITICAL**: {len(overlapping_epitopes)} epitopes found that match your input peptides!")
                                    for finding in overlapping_epitopes:
                                        st.write(f"- `{finding['epitope']}` found in {finding['frame']} at position {finding['position']}")
                                else:
                                    st.info("✅ Found epitopes do not directly match your input cancer peptides")
                                
                                # Create detailed summary
                                summary_df_epitopes = create_immunogenic_peptide_summary(plus1_findings, minus1_findings)
                                if summary_df_epitopes is not None:
                                    st.subheader("📋 Detailed Epitope Findings")
                                    st.dataframe(summary_df_epitopes, use_container_width=True, hide_index=True)
                                    
                                    # Show frame-specific details
                                    if plus1_findings:
                                        with st.expander(f"🔍 +1 Frame Epitopes ({len(plus1_findings)} found)", expanded=True):
                                            for i, finding in enumerate(plus1_findings, 1):
                                                st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}")
                                    
                                    if minus1_findings:
                                        with st.expander(f"🔍 -1 Frame Epitopes ({len(minus1_findings)} found)", expanded=True):
                                            for i, finding in enumerate(minus1_findings, 1):
                                                st.write(f"**{i}.** `{finding['epitope']}` at position {finding['position']}-{finding['end_position']}")
                                    
                                    # Download button for epitope findings
                                    excel_data = create_download_link(summary_df_epitopes, f"Cancer_Vaccine_Immunogenic_Peptides_{len(summary_df_epitopes)}.xlsx")
                                    st.download_button(
                                        label="📥 Download Vaccine Epitope Findings (Excel)",
                                        data=excel_data,
                                        file_name=f"Cancer_Vaccine_Immunogenic_Peptides_{len(summary_df_epitopes)}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        help="Download complete list of found immunogenic peptides in cancer vaccine design"
                                    )
                            else:
                                st.success("✅ **Excellent**: No known immunogenic peptides found in +1 or -1 reading frames of your cancer vaccine!")

                        else:
                            st.info("ℹ️ Immunogenic peptide scanning disabled - epitope_table_export.xlsx not found")

                        # Continue with existing CAI/GC analysis...
                        # CAI/GC Plot and +1 Stop Pie Chart
                        col_chart1, col_chart2 = st.columns([3, 1])
                        # ... rest of your existing code
                        
                        # CAI/GC Plot and +1 Stop Pie Chart
                        col_chart1, col_chart2 = st.columns([3, 1])
                        
                        # Find this section in tab5 (Cancer Vaccine Design):
                        with col_chart1:
                            st.markdown("##### CDS CAI and GC Content")
                            cai_result, cai_error = run_single_optimization(processed_cds, "In-Frame Analysis")
                            if not cai_error and cai_result:
                                cai_df = pd.DataFrame(cai_result)
                                
                                # Replace the existing chart with this:
                                positions = cai_df['Position'].tolist()
                                cai_weights = cai_df['CAI_Weight'].tolist()
                                amino_acids = cai_df['Amino_Acid'].tolist()
                                plus1_stop_positions = get_plus1_stop_positions(processed_cds)
                                minus1_stop_positions = get_minus1_stop_positions(processed_cds)
                                slippery_positions = get_slippery_motif_positions(processed_cds)
                                
                                display_stateful_overlay_chart(
                                    positions=positions,
                                    cai_weights=cai_weights,
                                    amino_acids=amino_acids,
                                    sequence=processed_cds,
                                    seq_name="Processed CDS",
                                    plus1_stop_positions=plus1_stop_positions,
                                    minus1_stop_positions=minus1_stop_positions,
                                    slippery_positions=slippery_positions
                                )
                            else:
                                st.warning("Could not generate CAI/GC plot.")
                            
                        with col_chart2:
                            st.markdown("##### CDS +1 Stop Codons")
                            plus1_stops = number_of_plus1_stops(full_cds)
                            if plus1_stops['total'] > 0:
                                stop_labels = ['TAA', 'TAG', 'TGA']
                                stop_values = [plus1_stops['TAA'], plus1_stops['TAG'], plus1_stops['TGA']]
                                fig_pie = create_interactive_pie_chart(stop_values, stop_labels, "+1 Stop Codon Distribution")
                                st.plotly_chart(fig_pie, use_container_width=True, key="cancer_vaccine_stop_pie_chart")
                            else:
                                st.info("No +1 stop codons found in the processed CDS.")

                        # Bottom row: full-width visualization
                        st.markdown("---")
                        st.markdown("##### Final mRNA Visualisation")
                        create_geneious_like_visualization(
                                utr5_seq=JT_5_UTR,
                                cds_seq=cds_with_stops[len(signal_peptide_dna):],
                                utr3_seq=JT_3_UTR,
                                signal_peptide_seq=signal_peptide_dna,
                                key_suffix=f"cancer_vaccine_{id(cds_with_stops)}"  # Using a unique ID
                            ) 
                    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
