Harmonized Optimization of Oligos and Frames (HOOF)
A comprehensive DNA sequence optimization and analysis tool for bioinformatics research, with integrated database search and patent research capabilities.
Features
Core Optimization Methods

Standard Codon Optimization: Uses most frequent codons for each amino acid
In-Frame Analysis: Calculates Codon Adaptation Index (CAI) with interactive 10bp GC content windows
Balanced Optimization: Advanced algorithm considering codon usage and +1 frame effects
NC Stop Codon Optimization: Specialized for alternative reading frame stop codon creation
JT Plus1 Stop Optimization: Creates specific stop motifs in +1 frame
+1 Frame Analysis: Comprehensive analysis including slippery motifs and immunogenic peptide scanning

Database Search & Integration

NCBI CDS Extraction: Automated search and extraction of coding sequences
UniProt Integration: Search protein database and retrieve linked nucleotide sequences
Patent Search: AI-powered search of Google Patents for molecular biology technologies
Multi-Database Search: Simultaneous search across multiple databases with intelligent ranking

mRNA Design Tools

Full mRNA Design: Create complete mRNA sequences with 5' UTR, CDS, and 3' UTR
Signal Peptide Integration: Library of common signal peptides for protein targeting
Cancer Vaccine Design: Multi-peptide vaccine construction with customizable linkers
GC Content Optimization: Automatic GC content correction and local window enforcement

Interactive Analysis

Real-time Visualizations: Interactive Plotly charts with hover details and zoom capabilities
Dual-axis Plots: CAI weights and GC content displayed simultaneously
Immunogenic Peptide Scanning: Automated detection of known immunogenic sequences
Batch Processing: Process multiple sequences with comparative analysis

Installation
Requirements
bashCopypip install streamlit pandas numpy matplotlib plotly biopython openpyxl requests beautifulsoup4 python-dotenv anthropic
Required Files

HumanCodons.xlsx: Codon usage frequency data (auto-loaded if present)
epitope_table_export.xlsx: Immunogenic peptide database (optional)
.env: API configuration file

API Configuration
Create a .env file in your application directory:
envCopySERPER_API_KEY=your_serper_api_key_here
ANTHROPIC_API=your_anthropic_api_key_here
Usage
Running the Application
bashCopystreamlit run main.py
Basic Workflow

Single Sequence Analysis: Paste DNA sequence and select optimization method
Database Search: Search for specific proteins and extract CDS sequences
Batch Processing: Upload multiple sequences for comparative analysis
mRNA Design: Create full mRNA constructs with UTRs and signal peptides
Patent Research: Search for relevant technologies and extract sequences

File Formats

Input: DNA sequences (plain text, FASTA), Excel files for batch processing
Output: Excel files with comprehensive analysis, FASTA sequences

Key Features
Immunogenic Peptide Scanning

Automatic detection of known immunogenic peptides in alternative reading frames
Database of epitopes for vaccine safety assessment
Frame-specific analysis (+1 and -1 reading frames)

Interactive Visualizations

CAI Analysis: Interactive plots showing codon adaptation index trends
GC Content Windows: Real-time GC content analysis in sliding windows
Stop Codon Distribution: Pie charts and bar graphs for stop codon analysis
Batch Comparisons: Side-by-side optimization comparisons

Database Integration

Targeted Search: Find specific proteins using descriptive terms
Cross-Reference Mining: Automatic retrieval of nucleotide sequences from protein databases
Quality Validation: Checks for valid DNA sequences before optimization
Seamless Transfer: Direct integration between search and optimization tools

Advanced Analysis

Slippery Motif Detection: Identification of ribosomal frameshifting sites
Alternative Frame Analysis: Comprehensive +1 and -1 frame stop codon analysis
Protein Translation: Automatic translation and validation
Sequence Metrics: GC content, CAI scores, sequence length analysis

File Structure
Copy├── main.py                          # Main application file
├── HumanCodons.xlsx                 # Codon usage data
├── epitope_table_export.xlsx        # Immunogenic peptide database
├── .env                             # API configuration
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
API Requirements
SERPER API (Required for database searches)

Used for Google-based searches (NCBI, Patents)
Sign up at: https://serper.dev/
Free tier available with usage limits

Anthropic API (Optional, for AI analysis)

Used for intelligent sequence ranking and analysis
Sign up at: https://console.anthropic.com/
Provides enhanced search result ranking

UniProt REST API

No API key required
Used for protein database searches
Free public access

Use Cases
Research Applications

Codon Optimization: Optimize sequences for protein expression
Vaccine Development: Design multi-epitope vaccines with safety analysis
Comparative Genomics: Find and analyze homologous sequences
Patent Research: Discover existing technologies and extract sequences

Educational Applications

Sequence Analysis Training: Interactive tools for learning bioinformatics
Protein Expression Projects: Source sequences for cloning experiments
Database Mining: Learn to search and extract biological data

Industrial Applications

Biotech R&D: Sequence optimization for commercial applications
IP Research: Patent landscape analysis for molecular biology
Quality Control: Batch processing and validation of sequences

Troubleshooting
Common Issues

API Connection Errors: Check your .env file configuration
File Not Found: Ensure HumanCodons.xlsx is in the application directory
Memory Issues: Reduce batch size for large datasets
Slow Performance: Check internet connection for database searches

Support

Check the "About" tab in the application for detailed feature descriptions
Test API connections using the built-in connection test buttons
Review the troubleshooting tips provided in error messages

Version Information
Current Version: v2.5 (Interactive Visualizations & Enhanced Analysis)
Recent Updates

Interactive Plotly visualizations with hover details
10bp GC content window analysis
Enhanced immunogenic peptide scanning
Cancer vaccine design workflow
Improved user interface and experience

License
This software is provided for research and educational purposes. Please ensure compliance with relevant terms of service for external APIs and databases.

