import os
from unstructured.partition.auto import partition

def extract_text_from_folder(input_folder, output_folder):
    """
    Extracts text from all supported files in a folder and saves it to an output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        print(f"Processing file: {filename}")
        
        try:
            # Use unstructured.partition to handle the file
            # This function automatically uses OCR for image-based PDFs
            elements = partition(filename=file_path)
            
            # Combine the elements into a single string
            full_text = "\n\n".join([str(el) for el in elements])
            
            # Create the output filename
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Write the extracted text to a new .txt file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            print(f"Successfully extracted text from '{filename}' to '{output_filename}'")
            
        except Exception as e:
            print(f"Failed to process '{filename}'. Error: {e}")

# --- Set your folder paths here ---
input_directory = "C:\\Project-Gavel\\Project-Gavel\\data"
output_directory = "C:\\Project-Gavel\\Project-Gavel\\extracted_text"

extract_text_from_folder(input_directory, output_directory)