import os
from unstructured.partition.auto import partition
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

def extract_and_chunk(input_folder):
    """
    Extracts text from all supported files in a folder, chunks it, and returns a list of chunks.
    a list of llamaindex document objects
    """
    documents = []

    print(f"Checking folder: {os.path.abspath(input_folder)}")
    files_in_folder = os.listdir(input_folder)
    print(f"Found files: {files_in_folder}")

    for filename in files_in_folder:
        file_path = os.path.join(input_folder, filename)
        
        if os.path.isdir(file_path):
            continue
        
        print(f"Processing file: {filename}")
        
        # --- REMOVE THE TRY...EXCEPT BLOCK ---
        elements = partition(filename=file_path)
        
        if not elements:
            print(f"-> WARNING: Unstructured returned no elements for '{filename}'.")
            continue 

        full_text = "\n\n".join([str(el) for el in elements])
        doc = Document(text=full_text, metadata={"file_name": filename})
        documents.append(doc)
        print(f"-> Successfully extracted text from '{filename}'")
            
            
    print(f"Extracted {len(documents)} documents. Now starting chunking.")
    
    #chunking logic
    print("\nStarting chunking process...")
    # splitter with desired chunk size
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # splitter processes the list of documents and creates nodes
    nodes = splitter.get_nodes_from_documents(documents)
    print("Chunking complete")

    return nodes

# folder path
input_directory = "data"

# getting the nodes
chunks = extract_and_chunk(input_directory)


print(f"\nCreated {len(chunks)} chunks.")
if chunks:
    print(f"Example chunk from '{chunks[0].metadata['file_name']}':")
    print("-------------------------------------------------------")
    print(chunks)
    print("-------------------------------------------------------")
