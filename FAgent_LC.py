import cv2  
import pytesseract
import numpy as np
import os
from docx import Document
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import SystemMessage, HumanMessage

# Load API key securely
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

#mistral_api_key="Cc8p5DVZUnwOnRk5C6YekMhcQTTHXGIk"
# Initialize LangChain OpenAI-compatible model
llm = ChatMistralAI(model="mistral-small", temperature=0.3, openai_api_key=mistral_api_key)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the Fastener and Machine Selection Guidelines
doc_path_fastener = r"F:\intern Synera AI\intern-ai\Fastener_Types_Manual.docx"
doc_path_machine = r"F:\intern Synera AI\intern-ai\Manufacturing Expert Manual.docx"

def load_guidelines(doc_path):
    doc = Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs])

fastener_guidelines = load_guidelines(doc_path_fastener)
machine_guidelines = load_guidelines(doc_path_machine)

# Function to get fastener and machine recommendations
def get_recommendations(image_description, fastener_guidelines, machine_guidelines):
    messages = [
        SystemMessage(content="You are an AI assistant that provides both fastener and machine recommendations."),
        HumanMessage(content=f"""
        Fastener Selection Guidelines:
        {fastener_guidelines}

        Machine Selection Guidelines:
        {machine_guidelines}

        Image Analysis:
        {image_description}

        Based on the guidelines and image analysis, suggest:
        1. The most suitable fastener type.
        2. The best manufacturing machine for this geometry.
        Explain your reasoning clearly.
        """)
    ]
    
    print("\nSending request to Mistral AI via LangChain...\n")
    response = llm.invoke(messages)
    print("\nResponse received!\n")
    return response.content

# Function to process image and extract details
def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    #converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #extract text from gray image
    text = pytesseract.image_to_string(gray)

    #edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # Bounding Box & Aspect Ratio
    if num_contours > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = round(float(w) / h, 2) if h != 0 else 0
    else:
        aspect_ratio = None

    # Convexity & Solidity (How "solid" the object is)
    solidity = None
    if num_contours > 0:
        area = cv2.contourArea(contours[0])
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = round(float(area) / hull_area, 2) if hull_area != 0 else None    
    
    # Color Histogram (for material detection)
    avg_color = cv2.mean(image)[:3]  # Average BGR color
    
    cv2.imshow("Edges", edges)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    return f"Edge count: {edge_count}, Contours: {num_contours}, Aspect ratio: {aspect_ratio}, Solidity: {solidity}, Avg Color (BGR): {avg_color}, Text from image: {text}"

# User selects an image to analyze
image_folder = r"F:\intern Synera AI\intern-ai\Sample Images"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

print("\nAvailable Images:")
for idx, file in enumerate(image_files):
    print(f"{idx }. {file}")

while True:
    try:
        choice = int(input("\nEnter the number of the image you want to analyze: ")) 
        if 0 <= choice < len(image_files):
            selected_image = os.path.join(image_folder, image_files[choice])
            break
        else:
            print("Invalid choice! Please enter a valid number.")
    except ValueError:
        print("Invalid input! Please enter a number.")


image_description = analyze_image(selected_image)

answer = get_recommendations(image_description, fastener_guidelines, machine_guidelines)
print(f"\nFastener & Machine Recommendation for {image_files[choice]}: {answer}")
print("\n"*10)
