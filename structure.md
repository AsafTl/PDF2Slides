# Project Structure

```
PDF2Slides/
├── venv/                       # Python virtual environment
├── requirements.txt            # Dependency tracker
├── src/                        # Core application code
│   ├── pipeline.py             # Orchestration script (Main entry point)
│   ├── pdf_processor.py        # PDF extraction and rasterization
│   ├── analyzer.py             # AI Layout parsing and OCR engine
│   └── builder.py              # PPTX Reconstruction engine
├── scripts/                    # Helper scripts for slide manipulation
│   └── generate_slide_plots.py # Plot reconstruction utility
├── tests/                      # Automated test scripts
│   └── test_pdf_processor.py   # Tests for pdf_processor functionality
└── README.md                   # Documentation for GitHub
```
