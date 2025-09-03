# ArteFact — Art History AI Research Platform

**ArteFact** is a sophisticated web application that bridges visual art and textual scholarship using AI. By automatically linking visual elements in artworks to scholarly descriptions, it empowers researchers, students, and art enthusiasts to discover new connections and understand artworks in their broader academic context.

##  What ArteFact Does

- **Upload or select artwork images** and find scholarly passages that describe similar visual elements
- **Search by region** - crop specific areas of paintings to find text about those visual details  
- **Filter results** by art historical topics or specific creators
- **Switch AI models** between CLIP and PaintingCLIP for different analysis approaches
- **Access scholarly sources** with full citations, DOI links, and BibTeX references
- **Generate heatmaps** showing which image regions contribute to text similarity using Grad-ECLIP
- **Interactive grid analysis** - click on 7×7 grid cells to analyze specific image regions

## 🏗️ Architecture Overview

### **Backend: Flask API with ML Pipeline**
- **Flask server** (`backend/runner/app.py`) serving the SPA from `frontend/`
- **ML Models**: CLIP base model + PaintingCLIP LoRA fine-tuned adapter
- **Inference Engine**: Region-aware analysis with 7×7 grid overlay and patch-level attention
- **Background Processing**: Thread-based task queue for ML inference with progress tracking
- **Caching System**: Intelligent caching of model components and embeddings for performance

### **Frontend: Interactive Web Application**
- **Single-page application** with responsive Bootstrap design
- **Image Tools**: Upload, crop, edit, undo, and analyze specific regions
- **Grid Analysis**: Click-to-analyze 7×7 grid cells for spatial understanding
- **Model Selection**: Dropdown to switch between CLIP and PaintingCLIP models
- **Academic Integration**: Full citation management, source verification, and BibTeX export
- **Real-time Feedback**: Progress indicators and status updates during processing

### **Data Architecture: Distributed Hugging Face Datasets**
- **`artefact-embeddings`**: Pre-computed sentence embeddings (12.8GB total)
  - `clip_embeddings.safetensors` (6.39GB) - CLIP model embeddings for 3.1M sentences
  - `paintingclip_embeddings.safetensors` (6.39GB) - PaintingCLIP embeddings for 3.1M sentences
  - `clip_sentence_ids.json` & `paintingclip_sentence_ids.json` - Sentence ID mappings
- **`artefact-json`**: Metadata and structured data
  - `sentences.json` - 3.1M sentence metadata with work associations
  - `works.json` - 7,200 work records with DOI and citation information
  - `creators.json` - Artist/creator mappings for filtering
  - `topics.json` - Topic classifications for content filtering
  - `topic_names.json` - Human-readable topic names
- **`artefact-markdown`**: Source documents and images (239,996 files)
  - 7,200 work directories with markdown files and associated images
  - Organized by work ID for efficient retrieval and context display
- **Local Models**: PaintingCLIP LoRA weights in `data/models/PaintingCLIP/`

## 🚀 Getting Started

### **Prerequisites**
- Python 3.9+
- Docker (for containerized deployment)
- Access to Hugging Face datasets (public access)

### **Local Development**
```bash
# Clone the repository
git clone https://github.com/sammwaughh/artefact-context.git
cd artefact-context

# Install backend dependencies
cd backend
pip install -e .

# Set environment variables
export STUB_MODE=1  # Use stub responses for development
export DATA_ROOT=./data

# Run the Flask development server
python -m backend.runner.app
```

### **Hugging Face Spaces Deployment**
```bash
# Add HF Spaces remote
git remote add hf https://huggingface.co/spaces/samwaugh/ArteFact

# Deploy to Space
git push hf main:main

# Force rebuild if needed (use HF Space settings → Factory Reset)
```

## ⚙️ Configuration

### **Environment Variables**
- `STUB_MODE`: Set to `1` for stub responses, `0` for real ML inference
- `DATA_ROOT`: Data directory path (default: `/data` for HF Spaces)
- `PORT`: Server port (set by Hugging Face Spaces)
- `MAX_WORKERS`: Thread pool size for ML inference (default: 2)
- `ARTEFACT_JSON_DATASET`: HF dataset name for JSON metadata (default: `samwaugh/artefact-json`)
- `ARTEFACT_EMBEDDINGS_DATASET`: HF dataset name for embeddings (default: `samwaugh/artefact-embeddings`)
- `ARTEFACT_MARKDOWN_DATASET`: HF dataset name for markdown files (default: `samwaugh/artefact-markdown`)

### **Data Sources**
The application automatically connects to distributed Hugging Face datasets:
- **Embeddings**: `samwaugh/artefact-embeddings` for fast similarity search using safetensors format
- **Metadata**: `samwaugh/artefact-json` for sentence, work, and topic information
- **Documents**: `samwaugh/artefact-markdown` for source documents and context
- **Models**: Local `data/models/` directory for ML model weights with fallback to base CLIP

## 📊 Data Processing Pipeline

### **ArtContext Research Pipeline**
ArteFact processes a massive corpus of art historical texts:

- **Scale**: 3.1 million sentences from scholarly articles across 7,200 works
- **Processing**: Executed on Durham University's Bede HPC cluster
- **GPU**: NVIDIA H100 with 32GB memory
- **Processing Time**: ~12 minutes for full corpus embedding generation
- **Output**: Structured embeddings and metadata for real-time analysis

### **Data Organization**
```
data/
├── models/
│   └── PaintingClip/          # LoRA fine-tuned weights
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── README.md
└── artifacts/                 # Uploaded images
└── outputs/                   # Inference results

# Data hosted on Hugging Face Hub:
# - samwaugh/artefact-embeddings: 12.8GB embeddings in safetensors format
# - samwaugh/artefact-json: 5 JSON metadata files
# - samwaugh/artefact-markdown: 239,996 files across 7,200 work directories
```

## 🧠 AI Models & Features

### **Core Models**
- **CLIP**: OpenAI's CLIP-ViT-B/32 for general image-text understanding
- **PaintingCLIP**: Fine-tuned version specialized for art historical content using LoRA adapters
- **Model Switching**: Users can choose between models for different analysis types
- **Fallback System**: Graceful degradation to base CLIP if LoRA adapter is unavailable

### **Advanced AI Features**
- **Region-Aware Analysis**: 7×7 grid overlay for spatial understanding of image regions
- **Grad-ECLIP Heatmaps**: Visual explanations of AI decision-making with attention visualization
- **Smart Filtering**: Topic and creator-based result filtering with real-time updates
- **Patch-Level Attention**: ViT patch embeddings for detailed analysis of image components
- **Batch Processing**: Efficient processing of large embedding datasets with memory optimization
- **Direct File Loading**: Fast loading of consolidated safetensors files for optimal performance

## 🎨 User Interface Features

### **Image Analysis Tools**
- **Drag & Drop Upload**: Easy image input with preview and validation
- **Interactive Grid**: Click-to-analyze specific image regions with visual feedback
- **Crop & Edit**: Built-in image manipulation tools with undo functionality
- **Image History**: Track and compare different analyses with thumbnail navigation
- **Example Gallery**: Pre-loaded historical artworks for quick testing

### **Academic Integration**
- **Citation Management**: One-click BibTeX copying with formatted output
- **Source Verification**: Direct links to scholarly articles and DOI resolution
- **Context Preservation**: Full paragraph context for matched sentences
- **Work Exploration**: Browse related images and metadata from the same scholarly work
- **Modal Documentation**: Detailed work information with embedded PDF previews

### **User Experience**
- **Real-time Progress**: Loading indicators and status updates during processing
- **Responsive Design**: Mobile-friendly interface with Bootstrap components
- **Error Handling**: Graceful error messages and recovery options
- **Performance Optimization**: Caching and efficient data loading for fast responses

## 🔬 Research & Development

### **Technical Innovations**
- **Efficient Embedding Storage**: Safetensors format for fast loading and memory efficiency
- **Memory-Optimized Inference**: Intelligent caching and batch processing
- **Real-Time Analysis**: Sub-second response times for similarity search
- **Scalable Architecture**: Designed for production deployment with distributed data
- **Distributed Data**: Hugging Face datasets for scalable data management and collaboration
- **Robust Error Handling**: Fallback mechanisms and graceful degradation

### **Academic Applications**
- **Art Historical Research**: Discover connections across large scholarly corpora
- **Digital Humanities**: Computational analysis of visual-textual relationships
- **Educational Tools**: Interactive learning for art history students and researchers
- **Scholarly Discovery**: AI-powered literature review and citation analysis
- **Cross-Reference Analysis**: Find related works and themes across different time periods

## 🛠️ Technical Implementation

### **Backend Architecture**
- **Flask Application**: RESTful API with async task processing
- **Thread Pool**: Background processing for ML inference tasks
- **Caching Layer**: Intelligent caching of model components and embeddings
- **Error Recovery**: Robust error handling with user-friendly messages
- **Data Validation**: Input validation and sanitization for security

### **Frontend Architecture**
- **Single Page Application**: jQuery-based interactive interface
- **Bootstrap UI**: Responsive design with modern components
- **Real-time Updates**: WebSocket-like polling for task status
- **State Management**: Client-side state for user interactions and preferences
- **Accessibility**: Keyboard navigation and screen reader support

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest backend/tests/`
5. Submit a pull request

### **Data Contributions**
- **Embeddings**: Process new art historical texts and generate embeddings
- **Models**: Improve fine-tuning and model performance
- **Documentation**: Enhance user guides and API documentation
- **Testing**: Add test cases for new features and edge cases

## 📄 License & Acknowledgments

**License**: MIT License

**Created by**: [Samuel Waugh](https://www.linkedin.com/in/samuel-waugh-31903b1bb/)

**Supervised by**: [Dr. Stuart James](https://stuart-james.com), Department of Computer Science, Durham University

**Supported by**: [N8 Centre of Excellence in Computationally Intensive Research (N8 CIR)](https://n8cir.org.uk/themes/internships/internships-2025/)

**Special Thanks**: Durham University's Bede HPC cluster for providing computational resources needed to process the large-scale art history corpus using Grace Hopper GPUs.

This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is coordinated by the Universities of Durham, Manchester and York.

## 🔗 Links

- **Live Application**: [ArteFact on Hugging Face Spaces](https://huggingface.co/spaces/samwaugh/ArteFact)
- **Source Code**: [GitHub Repository](https://github.com/sammwaughh/artefact-context)
- **Research Paper**: [Download PDF](paper/waugh2025artcontext.pdf)
- **Embeddings Dataset**: [artefact-embeddings on HF](https://huggingface.co/datasets/samwaugh/artefact-embeddings)
- **JSON Dataset**: [artefact-json on HF](https://huggingface.co/datasets/samwaugh/artefact-json)
- **Markdown Dataset**: [artefact-markdown on HF](https://huggingface.co/datasets/samwaugh/artefact-markdown)

---

*ArteFact represents a significant contribution to computational art history, making large-scale scholarly resources accessible through AI-powered visual analysis while maintaining academic rigor and providing transparent explanations of AI decision-making. The application leverages Hugging Face's distributed data infrastructure for scalable and collaborative research, enabling researchers worldwide to explore the connections between visual art and textual scholarship.*
