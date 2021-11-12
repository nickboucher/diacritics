import pickle
from glob import glob
from sys import argv
from textdistance import levenshtein
from toxic.core.model import ModelWrapper
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch import cuda
from PIL import Image, ImageDraw, ImageFont

toxic_model = ModelWrapper()
if cuda.is_available():
    toxic_model.model.cuda()
device = 'cuda:0' if cuda.is_available() else 'cpu'
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)

def load_font(font: str, font_size: int) -> None:
  _d = ImageDraw.Draw(Image.new("RGB", (0, 0), 'white'))
  font = ImageFont.truetype(font, font_size)    
  global draw
  def draw(text: str) -> Image:
    x, y = _d.textsize(text, font=font)
    img = Image.new("RGB", (x+40, y+20), 'white')
    d = ImageDraw.Draw(img)
    d.text((20, 5), text, font=font, fill=0)
    return img

load_font('arialuni.ttf', 32)

def ocr(text: str) -> str:
    image = draw(text)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    del pixel_values # remove from GPU
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

for filename in glob(argv[1]):
    print(f"Processing {filename}...")

    with open(filename, 'rb') as f:
        output = pickle.load(f)
        for exp_name, budgets in output.items():
            for budget, exp in budgets.items():
                for id, adv_example in exp.items():
                    adv_example['adv_example_ocr_adv_distance'] = levenshtein.distance(adv_example['adv_example_ocr'], adv_example['adv_example'])
                if exp_name == "toxic-diacriticals":
                    for id, adv_example in exp.items():
                            adv_example['adv_logit_toxic'] = float(toxic_model.predict([ocr(adv_example['adv_example'])])[0]['toxic'])
                            adv_example['adv_label_toxic'] = round(adv_example['adv_logit_toxic']) == 1

    with open(filename, 'wb') as f:
        pickle.dump(output, f)