import cv2  
import openai
import time
import pytesseract
import numpy as np
import os
from docx import Document
#from langchain.prompts import PromptTemplate #incase of further optimizing this agent we can use langchain or if the guideline doc is very big also


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



#opening the guidelines file and reading file
doc_path = r"F:\intern Synera AI\intern-ai\Fastener_Types_Manual.docx"
doc = Document(doc_path)

guidelines = "\n".join([para.text for para in doc.paragraphs])


#Open AI requesting and API
client = openai.Client(api_key = "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")

# Function to get fastener recommendations
def get_fastener_recommendation(image_description, guidelines):
    prompt = f"""You are an AI assistant that provides fastener recommendations based on the following guidelines:

    {guidelines}

    Image analysis for fastner recomendation:
    {image_description}

    Based on the guidelines and image analysis, suggest the most suitable fastner type and explain why:
    """

    try:
        print("Sending request to OpenAI...")  # Debugging print
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print("Response received!")  # Debugging print
        return response.choices[0].message.content

    except openai.RateLimitError:
        print("❌ Rate limit exceeded. Waiting for 30 seconds...")
        time.sleep(30)  
        return get_fastener_recommendation(image_description, guidelines)

    except Exception as e:  # Catch all errors
        print("❌ Error:", str(e))  
        return "An error occurred."
    
# Load the image folder
image_folder = r"F:\intern Synera AI\intern-ai\Sample Images"

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
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

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