# ⚖️ Project Gavel  
An intelligent **legal co-pilot for Indian jurisprudence** using **Retrieval-Augmented Generation (RAG)**.  
This tool lets you upload **legal case files (PDF/TXT)**, automatically indexes them using **FAISS**, and allows you to query them using **local open-source LLMs** (via [Ollama](https://ollama.com/)).  


---

##  Features  
-  Upload **PDF / TXT legal documents**  
-  Automatic document indexing using **FAISS**  
-  Query documents using **local LLMs (phi-3, gemma, mistral, llama3, etc.)**  
-  100% private (no data leaves your computer)  
-  Designed for **Indian legal use-cases**  

---

## Installation  

### 1. Clone this repo  
```bash
git clone https://github.com/<your-username>/Project-Gavel.git
cd Project-Gavel
```
 ### 2. Create a Virtual Environment
 ```bash
python -m venv gavel_env
```
Activate it:
```bash
gavel_env\Scripts\activate
```
### 3.Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
Download from [Ollama's Website](https://ollama.com/download)



###  Usage
### 1.Run the Model
```bash
ollama run llama3:8b
```


### 2. Run the streamlit
```bash
streamlit run app.py
```
