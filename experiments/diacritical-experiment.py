#!/usr/bin/env python3
import pickle
from abc import ABC
from typing import Callable, List, Dict, Tuple
from textdistance import levenshtein
from torch import cuda
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from time import process_time, sleep
from argparse import ArgumentParser
from scipy.optimize import differential_evolution
from os.path import exists
from tqdm.auto import tqdm

# Diacritical marks supported by Arial Unicode MS
# Full range is 768 - 879
diacriticals = list(map(chr, list(range(768, 838)) + [864, 865]))

distance = levenshtein.distance

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

def load_glue_data(start_index: int, end_index: int,):
  return load_dataset('glue', 'cola', split='validation') \
    .filter(lambda x: x['label'] == 1) \
    .remove_columns(['label']) \
    .select(range(start_index, end_index))

def serialize(adv_example: str, adv_example_ocr: str, input: str, input_ocr: str, adv_generation_time: int, budget: int, maxiter: int, popsize: int):
  return  {
    'adv_example': adv_example,
    'adv_example_ocr': adv_example_ocr,
    'adv_example_ocr_input_distance': distance(adv_example_ocr, input),
    'input': input,
    'input_ocr': input_ocr,
    'input_ocr_input_distance': distance(input_ocr, input),
    'adv_generation_time': adv_generation_time,
    'budget': budget,
    'maxiter': maxiter,
    'popsize': popsize
  }


class OcrObjective(ABC):
  """ Objective targeting OCR models using Unicode diacriticals. """

  def __init__(self, input: str, budget: int):
    self.input = input
    self.budget = budget
    self.cache: Dict[str,str] = dict()

  def bounds(self) -> List[Tuple[float, float]]:
    return [(0,len(diacriticals)-1), (-1, len(self.input)-1)] * self.budget

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]
    for i in range(0, len(perturbations), 2):
      inp_index = round(perturbations[i+1])
      if inp_index >= 0:
        diacritical = diacriticals[round(perturbations[i])]
        candidate = candidate[:inp_index] + [diacritical] + candidate[inp_index:]
    return ''.join(candidate)

  def ocr(self, text: str) -> str:
      raise NotImplementedError()

  def objective(self) -> Callable[[List[float]], float]:
    def _objective(perturbations: List[float]) -> float:
      candidate = self.candidate(perturbations)
      if candidate in self.cache:
        output = self.cache[candidate]
      else:
        output = self.ocr(candidate)
        self.cache[candidate] = output
      return -distance(output, self.input)
    return _objective

  def differential_evolution(self, maxiter: int, popsize: int) -> Dict:
    if self.budget > 0:
      start = process_time()
      result = differential_evolution(self.objective(), self.bounds(),
                                      disp=False, maxiter=maxiter,
                                      popsize=popsize, polish=False)
      end = process_time()
      generation_time = end - start
      adv_example = self.candidate(result.x)
      adv_example_ocr = self.ocr(adv_example)
      input_ocr = self.ocr(self.input)
    else:
      adv_example = self.input
      input_ocr = self.ocr(self.input)
      adv_example_ocr = input_ocr
      generation_time = 0
    return serialize(adv_example, adv_example_ocr, self.input, input_ocr, generation_time, self.budget, maxiter, popsize)


class TrOcrObjective(OcrObjective):

  def __init__(self, input: str, budget: int, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, device: str):
    super().__init__(input, budget)
    self.processor = processor
    self.model = model
    self.device = device

  def ocr(self, text: str) -> str:
    image = draw(text)
    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
    generated_ids = self.model.generate(pixel_values)
    return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def trocr_experiment(start_index: int, end_index: int, min_budget: int, max_budget: int, pkl_file: str, maxiter: int, popsize: int, overwrite: bool, cpu: bool, **kwargs):
  # Load resources
  label = "trocr-diacriticals"
  dataset = load_glue_data(start_index, end_index)
  print(f"Performing experiments against the TrOCR model for {len(dataset)} examples, from index {start_index} to {end_index}, with budgets {min_budget} to {max_budget}.")
  # The "printed" models output all uppercase, so we use the "handwritten" models
  # which appear to perform well on computer-generated text as well
  device = 'cuda:0' if not cpu and cuda.is_available() else 'cpu'
  processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
  model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
  print(f"TrOCR configured to use device {device}.")
  budgets = range(min_budget, max_budget+1)
  # Create or load pickle
  if overwrite or not exists(pkl_file):
    adv_examples = { label: {} }
  else:
    with open(pkl_file, 'rb') as f:
      adv_examples = pickle.load(f)
    if label not in adv_examples:
      adv_examples[label] = {}
  # Run experiments
  with tqdm(total=len(dataset)*len(budgets), desc="Adv. Examples") as pbar:
    for budget in budgets:
      if budget not in adv_examples[label]:
        adv_examples[label][budget] = {}
      for data in dataset:
        if data['idx'] not in adv_examples[label][budget]:
          objective = TrOcrObjective(data['sentence'], budget, processor, model, device)
          adv_examples[label][budget][data['idx']] = objective.differential_evolution(maxiter, popsize)
          with open(pkl_file, 'wb') as f:
            pickle.dump(adv_examples, f)
        pbar.update(1)


if __name__ == '__main__':

  parser = ArgumentParser(description='Adversarial NLP Experiments.')
  model = parser.add_mutually_exclusive_group(required=True)
  model.add_argument('-t', '--trocr', action='store_true', help="Target Microsoft TrOCR model.")
  parser.add_argument('-c', '--cpu', action='store_true', help="Use CPU for ML inference instead of CUDA.")
  parser.add_argument('pkl_file', help="File to contain Python pickled output.")
  parser.add_argument('-e', '--end-index', type=int, default=500, help="The upper bound of the items in the dataset to use in experiments.")
  parser.add_argument('-s', '--start-index', type=int, default=0, help="The lower bound of the items in the dataset to use in experiments.")
  parser.add_argument('-l', '--min-budget', type=int, default=1, help="The lower bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-u', '--max-budget', type=int, default=5, help="The upper bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-a', '--maxiter', type=int, default=10, help="The maximum number of iterations in the genetic algorithm.")
  parser.add_argument('-p', '--popsize', type=int, default=32, help="The size of the population in the genetic algorithm.")
  parser.add_argument('-o', '--overwrite', action='store_true', help="Overwrite existing results file instead of resuming.")
  parser.add_argument('-f', '--font', default="arialuni.ttf", help="TTF font file to use for image rendering.")
  parser.add_argument('-F', '--font-size', type=int, default=32, help="Font size for image rendering.")
  args = parser.parse_args()

  load_font(args.font, args.font_size)

  if args.trocr:
    trocr_experiment(**vars(args))