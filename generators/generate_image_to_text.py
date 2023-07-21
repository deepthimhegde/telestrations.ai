import os
import secrets

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials


class TextGenerator:
    """This class provides functions to generate a text caption based on the given
    image prompt using Azure Cognitive Services image description model."""

    def generate_text(self):
        # authentication
        computervision_client = ComputerVisionClient(
            secrets.Secrets.ENDPOINT, CognitiveServicesCredentials(secrets.Secrets.SUBSCRIPTION_KEY))
        # Open local image file
        local_image_path = os.path.join("./", "tmp.jpg")
        local_image = open(local_image_path, "rb")
        # Call API
        description_result = computervision_client.describe_image_in_stream(local_image)
        # Get the captions (descriptions) from the response, with confidence level
        if (len(description_result.captions) == 0):
            print("No description detected.")
        else:
            for caption in description_result.captions:
                print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
