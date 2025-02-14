import cv2  
from mistralai.client import MistralClient
import time
import pytesseract
import numpy as np
import os
from docx import Document
#from langchain.prompts import PromptTemplate #incase of further optimizing this agent we can use langchain or if the guideline doc is very big also


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



#opening the guidelines file and reading file
doc_path = "F:\intern Synera AI\intern-ai\Fastener_Types_Manual.docx"
doc = Document(doc_path)

guidelines = "\n".join([para.text for para in doc.paragraphs])


#Open AI requesting and API
client = MistralClient(api_key = "5UDx2tPbp37EYvtRhLGo7MLvOiMi3Ecs")

# Function to get fastener recommendations
def get_fastener_recommendation(image_description, guidelines):
    prompt = f"""You are an AI assistant that provides fastener recommendations based on the following guidelines:

    {guidelines}

    Image analysis for fastner recomendation:
    {image_description}

    Based on the guidelines and image analysis, suggest the most suitable fastner type and explain why:
    """

    
    print("Sending request to Mistral AI...")  # Debugging print
    response = client.chat(
        model="mistral-tiny",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Response received!")  # Debugging print
    return response.choices[0].message.content

    
    
# Load the image folder
image_folder = "F:\intern Synera AI\intern-ai\Sample Images"

#creating a list of .png files
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

for image_file in image_files:
    image_path = os.path.join(image_folder,image_file)
    image = cv2.imread(image_path)

    # Check if the image loaded correctly
    if image is None:
        print("Error: Could not load image. Please check the file path!")

    else:
        #converting to gray scale
        gray = cv2.cvtColor(image,cv2.COLOR_BAYER_BGGR2GRAY)

        #extract text from gray image
        text = pytesseract.image_to_string(gray)

        #edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.count_nonzero(edges)


        # Display the image in a new window
        cv2.imshow("Edges", edges)
        print("Number of edges", edge_count)

        # Wait for a key press/2 sec and close the window
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        image_description = f"Detected text: {text.strip()}. Edge count: {edge_count}."





answer = get_fastener_recommendation(image_description, guidelines)

print("Fastener Recommendation:", answer)