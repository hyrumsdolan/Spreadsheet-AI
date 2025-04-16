<div align="center">

# 📊 Spreadsheet AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI-412991.svg)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/AI-Anthropic-0B3C8D.svg)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/powered%20by-Streamlit-FF4B4B.svg)](https://streamlit.io)

**Effortlessly process CSV/Excel files with AI - row by row**

[Features](#-features) • [Setup](#-setup) • [Usage](#-usage) • [Examples](#-examples)

</div>

---

## ✨ Features

- 🔄 **Upload CSV or Excel files** - Process your tabular data without format restrictions
- 🧠 **AI-Powered Processing** - Choose between OpenAI or Anthropic models
- 🎯 **Selective Processing** - Specify which columns to feed into the AI
- 📝 **Custom Instructions** - Create tailored prompts for your specific use case
- ⚡ **Configurable Concurrency** - Adjust processing speed for large files
- 💾 **Flexible Output** - Download results as standalone CSV or append to the original file

## 🚀 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spreadsheet-ai.git
   cd spreadsheet-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   ```

4. **Launch the application**
   ```bash
   streamlit run main.py
   ```

## 🔍 Usage

1. Open the app in your browser (typically at `http://localhost:8501`)
2. Upload your CSV or Excel file using the file uploader
3. Select columns to include as context for the AI
4. Write custom instructions for how the AI should process each row
5. Configure concurrency settings based on your needs
6. Start processing and monitor progress
7. Download your results when processing is complete

## 📊 Examples

- **Data Enrichment**: Add missing information to your dataset
- **Sentiment Analysis**: Analyze customer feedback at scale
- **Content Generation**: Create descriptions from product attributes
- **Data Classification**: Categorize entries based on text content
- **Translation**: Convert text between languages

---

<div align="center">
Made with ❤️ by <a href="https://github.com/yourusername">Your Name</a>
</div> 