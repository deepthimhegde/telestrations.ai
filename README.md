Fun AI implementation of [Telestrations](https://www.ultraboardgames.com/telestrations/game-rules.php). In the real game, each player takes turns to either caption a drawing, or draw a picture for a given phrase/caption and passes it along to the next person.

## Data source:
[MS COCO dataset](https://cocodataset.org/#download) contains image-text pairs, either of which can be used as an initial prompt.

## Implementation details:
 1. The program contains 2 components:
  * Image-to-text piece that uses [image descriptions](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-describing-images) feature of Azure Cognitive services to generate captions.
  * Text-to-image piece that uses OpenAI's [DALL-E](https://openai.com/blog/dall-e/) type model to generate image from the generated caption from step 1. Huggingface library hosts smaller versions of the model such as [dalle-mini](https://huggingface.co/flax-community/dalle-mini) that is used here.
 2. The game is played by alternately calling the 2 components by feeding in the output of the previous step as input to the next.
