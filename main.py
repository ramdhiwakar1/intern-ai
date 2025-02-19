import os
import sys
from docx import Document
from core.feature_extraction import FeatureExtraction
from core.fastenerguidelines import FastenerGuideline
from core.agent import FastenerAgent

# Define the manual path as a constant relative to the current file
MANUAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Manufacturing Expert Manual.docx")

class ManufacturingAgent:
    def __init__(self):
        """Initialize the Manufacturing Agent with manufacturing guidelines from the manual."""
        self.guidelines = self.load_guidelines()

    def load_guidelines(self):
        """Loads manufacturing guidelines from the predefined DOCX manual."""
        guidelines = {}
        try:
            doc = Document(MANUAL_PATH)
            for para in doc.paragraphs:
                if "-" in para.text:
                    key, value = para.text.split("-", 1)
                    guidelines[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error loading manufacturing manual: {str(e)}")
        return guidelines

    def find_best_machine(self, features):
        """Determine the best manufacturing process based on extracted features."""
        hole_diameter = features.get("hole_diameter", 0)
        curvature = features.get("curvature", 0)
        geometry_type = features.get("geometry_type", "unknown")

        # Machine selection logic based on features
        if geometry_type == "flat" and hole_diameter > 10:
            machine = "Laser Cutting or Waterjet Cutting"
        elif geometry_type == "cylindrical":
            machine = "CNC Lathe or CNC Turning"
        elif geometry_type == "complex":
            machine = "CNC Milling or 5-Axis Machining"
        elif hole_diameter < 5:
            machine = "EDM (Electrical Discharge Machining)"
        else:
            machine = "3D Printing or Injection Molding"

        reason = self.guidelines.get(machine, "Default machine recommendation based on part geometry.")
        return {"machine": machine, "reason": reason}

def main(image_path_2d, depth_image_path_3d, *args):  # Modified to accept variable arguments
    """
    Main function to process images and recommend fasteners and manufacturing processes.
    
    Args:
        image_path_2d (str): Path to 2D image
        depth_image_path_3d (str): Path to 3D depth image
    
    Returns:
        tuple: (features, fastener_decision, manufacturing_decision)
    """
    try:
        # Validate file paths
        if not os.path.exists(image_path_2d):
            raise FileNotFoundError(f"2D image not found: {image_path_2d}")
        if not os.path.exists(depth_image_path_3d):
            raise FileNotFoundError(f"3D depth image not found: {depth_image_path_3d}")

        # Step 1: Extract features
        feature_extractor = FeatureExtraction(image_path_2d, depth_image_path_3d)
        features = feature_extractor.extract_features()

        # Step 2: Load Fastener Guidelines
        fastener_guideline = FastenerGuideline().get_guideline()

        # Step 3: Get fastener recommendation
        agent = FastenerAgent(fastener_guideline)
        fastener_decision = agent.find_best_fastener(features)

        # Step 4: Get manufacturing recommendation
        manufacturing_agent = ManufacturingAgent()
        manufacturing_decision = manufacturing_agent.find_best_machine(features)
        
        return features, fastener_decision, manufacturing_decision

    except Exception as e:
        error_msg = str(e)
        print(f"Error in main process: {error_msg}", file=sys.stderr)
        return {}, {"fastener_type": "error", "explanation": error_msg}, {"machine": "error", "reason": error_msg}