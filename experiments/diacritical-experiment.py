#!/usr/bin/env python3
import pickle
import ast
import pandas as pd
import numpy as np
from abc import ABC
from typing import Callable, List, Dict, Tuple
from textdistance import levenshtein
from torch import cuda, hub, device as torchdevice
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from time import process_time
from argparse import ArgumentParser
from scipy.optimize import differential_evolution
from os.path import exists
from tqdm.auto import tqdm
from toxic.core.model import ModelWrapper
from logging import getLogger, WARNING
from fairseq.hub_utils import GeneratorHubInterface
from sacrebleu import sentence_chrf
from bs4 import BeautifulSoup
from fairseq import utils, tasks, checkpoint_utils, options
from collections import namedtuple
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

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

def load_glue_data(start_index: int, end_index: int):
  return load_dataset('glue', 'cola', split='validation') \
    .filter(lambda x: x['label'] == 1) \
    .remove_columns(['label']) \
    .select(range(start_index, end_index))

def load_toxic_data(start_index: int, end_index: int):
    # Load toxic comments data set
    getLogger('numexpr.utils').setLevel(WARNING)
    comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
    annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')
    # labels a comment as toxic if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['toxicity'].mean() > 0.5
    # join labels and comments
    comments['toxicity'] = labels
    # remove newline and tab tokens
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    test_comments = comments.query("split=='test'").query("toxicity==True")
    examples = test_comments.reset_index().to_dict('records')
    return examples[start_index:end_index]

def load_translation_data(start_index: int, end_index: int):
  # Build source and target mappings for BLEU scoring
  source = dict()
  target = dict()
  with open('newstest2014-fren-src.en.sgm', 'r') as f:
    source_doc = BeautifulSoup(f, 'html.parser')
  with open('newstest2014-fren-ref.fr.sgm', 'r') as f:
    target_doc = BeautifulSoup(f, 'html.parser')
  for doc in source_doc.find_all('doc'):
    source[str(doc['docid'])] = dict()
    for seg in doc.find_all('seg'):
      source[str(doc['docid'])][str(seg['id'])] = str(seg.string)
  for docid, doc in source.items():
    target[docid] = dict()
    for segid in doc:
      node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
      target[docid][segid] = str(node.string)
  # Sort the examples in order of length to improve runtime
  source_list = []
  for docid, doc in source.items():
    for segid, seg in doc.items():
      source_list.append({ docid: { segid: seg }})
  source_list.sort(key=lambda x: len(str(list(list(x.values())[0].values())[0])))
  source_list = source_list[start_index:end_index]
  output = []
  for example in source_list:
    for docid, doc in example.items():
      for segid, seg in doc.items():
        output.append({
            'docid': docid,
            'segid': segid,
            'english': seg,
            'french': target[docid][segid]
          })
  return output

def load_de_translation_data(start_index: int, end_index: int):
  # Build source and target mappings for BLEU scoring
  source = dict()
  target = dict()
  with open('newstest2020-deen-src.de.sgm', 'r') as f:
    source_doc = BeautifulSoup(f, 'html.parser')
  with open('newstest2020-deen-ref.en.sgm', 'r') as f:
    target_doc = BeautifulSoup(f, 'html.parser')
  for doc in source_doc.find_all('doc'):
    source[str(doc['docid'])] = dict()
    for seg in doc.find_all('seg'):
      source[str(doc['docid'])][str(seg['id'])] = str(seg.string)
  for docid, doc in source.items():
    target[docid] = dict()
    for segid in doc:
      node = target_doc.select_one(f'doc[docid="{docid}"] > p > seg[id="{segid}"]')
      target[docid][segid] = str(node.string)
  # Sort the examples in order of length to improve runtime
  source_list = []
  for docid, doc in source.items():
    for segid, seg in doc.items():
      source_list.append({ docid: { segid: seg }})
  source_list.sort(key=lambda x: len(str(list(list(x.values())[0].values())[0])))
  source_list = source_list[start_index:end_index]
  output = []
  for example in source_list:
    for docid, doc in example.items():
      for segid, seg in doc.items():
        output.append({
            'docid': docid,
            'segid': segid,
            'german': seg,
            'english': target[docid][segid]
          })
  return output

def serialize_trocr(adv_example: str, adv_example_ocr: str, input: str, input_ocr: str, adv_generation_time: int, budget: int, maxiter: int, popsize: int, **kwargs):
  base = {
    'adv_example': adv_example,
    'input': input,
    'adv_generation_time': adv_generation_time,
    'budget': budget,
    'maxiter': maxiter,
    'popsize': popsize
  }
  if input_ocr is not None and adv_example_ocr is not None:
    base = sorted({
      'adv_example_ocr': adv_example_ocr,
      'adv_example_ocr_input_distance': distance(adv_example_ocr, input),
      'adv_example_ocr_adv_distance': distance(adv_example_ocr, adv_example),
      'input_ocr': input_ocr,
      'input_ocr_input_distance': distance(input_ocr, input),
      **base
    })
  return base

def serialize_toxic(adv_label_toxic: bool, gold_label_toxic: bool, adv_logit_toxic: float, **kwargs):
  base = serialize_trocr(**kwargs)
  return {
    'adv_label_toxic': adv_label_toxic,
    'gold_label_toxic': gold_label_toxic,
    'adv_logit_toxic': adv_logit_toxic,
    **base
  }

def serialize_translation(adv_translation: str, gold_translation: str, adv_bleu: float, **kwargs):
  base = serialize_trocr(**kwargs)
  return {
    'adv_translation': adv_translation,
    'gold_translation': gold_translation,
    'adv_bleu': adv_bleu,
    **base
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
    return serialize_trocr(adv_example, adv_example_ocr, self.input, input_ocr, generation_time, self.budget, maxiter, popsize)


class TrOcrObjective(OcrObjective):

  def __init__(self, input: str, budget: int, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, device: str, ocr_line_len: int):
    super().__init__(input, budget)
    self.processor = processor
    self.model = model
    self.device = device
    self.ocr_line_len = ocr_line_len

  def trocr(self, text: str) -> str:
    image = draw(text)
    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
    generated_ids = self.model.generate(pixel_values)
    del pixel_values # remove from GPU
    return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  def ocr(self, text: str) -> str:
    tokens = text.split()
    buffer = ''
    output = ''
    for token in tokens:
      if buffer == '':
        buffer = token
      elif len(buffer) + len(token) < self.ocr_line_len:
        buffer += ' ' + token
      else:
        output += self.trocr(buffer) + ' '
        buffer = token
    if buffer != '':
      output += self.trocr(buffer)
    elif output != '':
      output = output[:-1] # Remove trailing whitespace
    return output


class ToxicOcrObjective(TrOcrObjective):

  def __init__(self, input: str, budget: int, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, device: str, toxic_model: ModelWrapper, gold_label_toxic: bool, ocr_line_len: int):
      super().__init__(input, budget, processor, model, device, ocr_line_len)
      self.toxic_model = toxic_model
      self.gold_label_toxic = gold_label_toxic

  def objective(self) -> Callable[[List[float]], float]:
    ''' Tries to make all inputs classify as non-toxic. '''
    def _objective(perturbations: List[float]) -> float:
      candidate = self.candidate(perturbations)
      if candidate in self.cache:
        output = self.cache[candidate]
      else:
        output = self.toxic_model.predict([self.ocr(candidate)])[0]['toxic']
        self.cache[candidate] = output
      return output
    return _objective

  def differential_evolution(self, maxiter: int, popsize: int) -> Dict:
    adv_example = super().differential_evolution(maxiter, popsize)
    adv_logit_toxic = float(self.toxic_model.predict([self.ocr(adv_example['adv_example'])])[0]['toxic'])
    adv_label_toxic = round(adv_logit_toxic) == 1
    return serialize_toxic(adv_label_toxic, self.gold_label_toxic, adv_logit_toxic, **adv_example)

class TranslationOcrObjective(TrOcrObjective):

  def __init__(self, input: str, budget: int, processor: TrOCRProcessor, model: VisionEncoderDecoderModel, device: str, translation_model: GeneratorHubInterface, gold_translation: str, ocr_line_len: int):
      super().__init__(input, budget, processor, model, device, ocr_line_len)
      self.translation_model = translation_model
      self.gold_translation = gold_translation

  def objective(self) -> Callable[[List[float]], float]:
    ''' Tries to minimize BLEU score. '''
    def _objective(perturbations: List[float]) -> float:
      candidate = self.candidate(perturbations)
      if candidate in self.cache:
        output = self.cache[candidate]
      else:
        translation = self.translation_model.translate(self.ocr(candidate))
        output = sentence_chrf(translation, [self.gold_translation]).score
        self.cache[candidate] = output
      return output
    return _objective

  def differential_evolution(self, maxiter: int, popsize: int) -> Dict:
    adv_example = super().differential_evolution(maxiter, popsize)
    adv_translation = self.translation_model.translate(self.ocr(adv_example['adv_example']))
    adv_bleu = sentence_chrf(adv_translation, [self.gold_translation]).score
    return serialize_translation(adv_translation, self.gold_translation, adv_bleu, **adv_example)


class VisrepOcrObjective(OcrObjective):

  def __init__(self, input: str, budget: int, translate: Callable[[str],str], gold_translation: str):
      super().__init__(input, budget)
      self.translate = translate
      self.gold_translation = gold_translation

  def objective(self) -> Callable[[List[float]], float]:
    ''' Tries to minimize BLEU score. '''
    def _objective(perturbations: List[float]) -> float:
      candidate = self.candidate(perturbations)
      if candidate in self.cache:
        output = self.cache[candidate]
      else:
        translation = self.translate(candidate)
        output = sentence_chrf(translation, [self.gold_translation]).score
        self.cache[candidate] = output
      return output
    return _objective

  def ocr(self, text: str):
    return None # This model doesn't use OCR

  def differential_evolution(self, maxiter: int, popsize: int) -> Dict:
    adv_example = super().differential_evolution(maxiter, popsize)
    adv_translation = self.translate(adv_example['adv_example'])
    adv_bleu = sentence_chrf(adv_translation, [self.gold_translation]).score
    return serialize_translation(adv_translation, self.gold_translation, adv_bleu, adv_example_ocr=None, input_ocr=None, **adv_example)


def create_or_load_pickle(pkl_file: str, label: str, overwrite: bool) -> Dict:
  if overwrite or not exists(pkl_file):
    adv_examples = { label: {} }
  else:
    with open(pkl_file, 'rb') as f:
      adv_examples = pickle.load(f)
    if label not in adv_examples:
      adv_examples[label] = {}
  return adv_examples

def load_trocr(cpu: bool):
  # The "printed" models output all uppercase, so we use the "handwritten" models
  # which appear to perform well on computer-generated text as well
  device = 'cuda:0' if not cpu and cuda.is_available() else 'cpu'
  processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
  model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
  print(f"TrOCR configured to use device {device}.")
  return device, processor, model

def load_visrep(checkpoint: str, tgt_dict: str, font: str, cpu: bool) -> Callable[[str],str]:
  
  def make_batches(lines, cfg, task, max_positions, encode_fn):
    constraints_tensor = None
    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
      dataset=task.build_dataset_for_inference(
        tokens, lengths, constraints=constraints_tensor
      ),
      max_tokens=cfg.dataset.max_tokens,
      max_sentences=cfg.dataset.batch_size,
      max_positions=max_positions,
      ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
      ids = batch["id"]
      src_tokens = batch["net_input"]["src_tokens"]
      src_lengths = batch["net_input"]["src_lengths"]
      constraints = batch.get("constraints", None)

      yield Batch(
        ids=ids,
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        constraints=constraints,
      )

  parser = options.get_interactive_generation_parser()
  args = options.parse_args_and_arch(parser, input_args=['./', '--task', 'visual_text', '--path', checkpoint, '-s', 'de', '-t', 'en', '--target-dict', tgt_dict, '--image-font-path', font, '--beam', '5'])
  cfg = convert_namespace_to_omegaconf(args)

  # Fix seed for stochastic decoding
  np.random.seed(cfg.common.seed)
  utils.set_torch_seed(cfg.common.seed)

  use_cuda = not cpu and cuda.is_available()

  # Setup task, e.g., translation
  task = tasks.setup_task(cfg.task)

  # Load ensemble
  overrides = ast.literal_eval(cfg.common_eval.model_overrides)

  models, _ = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path),
    arg_overrides=overrides,
    task=task,
    suffix=cfg.checkpoint.checkpoint_suffix,
    strict=(cfg.checkpoint.checkpoint_shard_count == 1),
    num_shards=cfg.checkpoint.checkpoint_shard_count,
  )

  # Set dictionaries
  src_dict = task.source_dictionary
  tgt_dict = task.target_dictionary

  # Optimize ensemble for generation
  for model in models:
    if model is None:
      continue
    if cfg.common.fp16:
      model.half()
    if use_cuda:
      model.cuda()
    model.prepare_for_inference_(cfg)

  # Initialize generator
  generator = task.build_generator(models, cfg.generation)

  # Handle tokenization and BPE
  tokenizer = task.build_tokenizer(cfg.tokenizer)
  bpe = task.build_bpe(cfg.bpe)

  def encode_fn(x):
    if tokenizer is not None:
      x = tokenizer.encode(x)
    if bpe is not None:
      x = bpe.encode(x)
    return x

  def decode_fn(x):
    if bpe is not None:
      x = bpe.decode(x)
    if tokenizer is not None:
      x = tokenizer.decode(x)
    return x

  # Load alignment dictionary for unknown word replacement
  # (None if no unknown word replacement, empty if no path to align dictionary)
  align_dict = utils.load_align_dict(cfg.generation.replace_unk)

  max_positions = utils.resolve_max_positions(
    task.max_positions(), *[model.max_positions() for model in models]
  )

  Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
  
  def translate(input: str) -> str:
    results = []
    for batch in make_batches([input], cfg, task, max_positions, encode_fn):
      bsz = batch.src_tokens.size(0)
      src_tokens = batch.src_tokens
      src_lengths = batch.src_lengths
      constraints = batch.constraints
      if use_cuda:
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()
        if constraints is not None:
          constraints = constraints.cuda()

      sample = {
        "net_input": {
          "src_tokens": src_tokens,
          "src_lengths": src_lengths,
        },
      }

      translations = task.inference_step(
        generator, models, sample, constraints=constraints
      )

      list_constraints = [[] for _ in range(bsz)]
      for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
        constraints = list_constraints[i]
        results.append(
          (
            id,
            src_tokens_i,
            hypos
          )
        )

    # sort output to match input order
    for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
      src_str = ''
      if src_dict is not None:
        src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
      # Process top predictions
      for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
          hypo_tokens=hypo["tokens"].int().cpu(),
          src_str=src_str,
          alignment=hypo["alignment"],
          align_dict=align_dict,
          tgt_dict=tgt_dict,
          remove_bpe=cfg.common_eval.post_process,
          extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
        )
        detok_hypo_str = decode_fn(hypo_str)
        output = detok_hypo_str.replace(' ', '').replace('▁', ' ').strip()
        return output
      
  return translate

def trocr_experiment(start_index: int, end_index: int, min_budget: int, max_budget: int, pkl_file: str, maxiter: int, popsize: int, overwrite: bool, cpu: bool, ocr_line_len: int, **kwargs):
  # Load resources
  label = "trocr-diacriticals"
  dataset = load_glue_data(start_index, end_index)
  print(f"Performing experiments against the TrOCR model for {len(dataset)} examples, from index {start_index} to {end_index}, with budgets {min_budget} to {max_budget}.")
  device, processor, model = load_trocr(cpu)
  budgets = range(min_budget, max_budget+1)
  adv_examples = create_or_load_pickle(pkl_file, label, overwrite)
  # Run experiments
  with tqdm(total=len(dataset)*len(budgets), desc="Adv. Examples") as pbar:
    for budget in budgets:
      if budget not in adv_examples[label]:
        adv_examples[label][budget] = {}
      for data in dataset:
        if data['idx'] not in adv_examples[label][budget]:
          objective = TrOcrObjective(data['sentence'], budget, processor, model, device, ocr_line_len)
          adv_examples[label][budget][data['idx']] = objective.differential_evolution(maxiter, popsize)
          with open(pkl_file, 'wb') as f:
            pickle.dump(adv_examples, f)
        pbar.update(1)

def toxic_experiment(start_index: int, end_index: int, min_budget: int, max_budget: int, pkl_file: str, maxiter: int, popsize: int, overwrite: bool, cpu: bool, ocr_line_len: int, **kwargs):
  # Load resources
  label = "toxic-diacriticals"
  dataset = load_toxic_data(start_index, end_index)
  print(f"Performing experiments against the IBM MaxToxic model for {len(dataset)} examples, from index {start_index} to {end_index}, with budgets {min_budget} to {max_budget}.")
  device, processor, model = load_trocr(cpu)
  getLogger().setLevel(WARNING)
  toxic_model = ModelWrapper()
  if not cpu and cuda.is_available():
    toxic_model.model.cuda()
    toxic_model.device = torchdevice("cuda")
    device = "cuda"
  else:
    device = "cpu"
  print(f"MaxToxic model configured to use device {device}.")
  budgets = range(min_budget, max_budget+1)
  adv_examples = create_or_load_pickle(pkl_file, label, overwrite)
  # Run experiments
  with tqdm(total=len(dataset)*len(budgets), desc="Adv. Examples") as pbar:
    for budget in budgets:
      if budget not in adv_examples[label]:
        adv_examples[label][budget] = {}
      for data in dataset:
        if str(data['rev_id']) not in adv_examples[label][budget]:
          objective = ToxicOcrObjective(data['comment'], budget, processor, model, device, toxic_model, data['toxicity'], ocr_line_len)
          adv_examples[label][budget][str(data['rev_id'])] = objective.differential_evolution(maxiter, popsize)
          with open(pkl_file, 'wb') as f:
            pickle.dump(adv_examples, f)
        pbar.update(1)

def translation_experiment(start_index: int, end_index: int, min_budget: int, max_budget: int, pkl_file: str, maxiter: int, popsize: int, overwrite: bool, cpu: bool, ocr_line_len: int, **kwargs):
  # Load resources
  label = "translation-diacriticals"
  dataset = load_translation_data(start_index, end_index)
  print(f"Performing experiments against the Fairseq WMT14 EN->FR translation model for {len(dataset)} examples, from index {start_index} to {end_index}, with budgets {min_budget} to {max_budget}.")
  device, processor, model = load_trocr(cpu)
  getLogger('fairseq').setLevel(WARNING)
  en2fr = hub.load('pytorch/fairseq',
                   'transformer.wmt14.en-fr',
                   tokenizer='moses',
                   bpe='subword_nmt',
                   verbose=False).eval()
  if not cpu and cuda.is_available():
    en2fr.cuda()
    device = "cuda"
  else:
    en2fr.cpu()
    device = "cpu"
  print(f"Fairseq WMT14 EN->FR translation model configured to use device {device}.")
  budgets = range(min_budget, max_budget+1)
  adv_examples = create_or_load_pickle(pkl_file, label, overwrite)
  # Run experiments
  with tqdm(total=len(dataset)*len(budgets), desc="Adv. Examples") as pbar:
    for budget in budgets:
      if budget not in adv_examples[label]:
        adv_examples[label][budget] = {}
      for data in dataset:
        id = f"{data['docid']}-{data['segid']}"
        if id not in adv_examples[label][budget]:
          objective = TranslationOcrObjective(data['english'], budget, processor, model, device, en2fr, data['french'], ocr_line_len)
          adv_examples[label][budget][id] = objective.differential_evolution(maxiter, popsize)
          with open(pkl_file, 'wb') as f:
            pickle.dump(adv_examples, f)
        pbar.update(1)

def visrep_experiment(start_index: int, end_index: int, min_budget: int, max_budget: int, pkl_file: str, maxiter: int, popsize: int, overwrite: bool, cpu: bool, ocr_line_len: int, checkpoint: str, tgt_dict: str, font: str, **kwargs):
  # Load resources
  label = "visrep-diacriticals"
  dataset = load_de_translation_data(start_index, end_index)
  print(f"Performing experiments against the Visual Text Translation DE->EN model for {len(dataset)} examples, from index {start_index} to {end_index}, with budgets {min_budget} to {max_budget}.")
  translate = load_visrep(checkpoint, tgt_dict, font, cpu)
  print(f"Visual Text Translation model DE->EN configured to use device {'CPU' if cpu else 'GPU'}.")
  budgets = range(min_budget, max_budget+1)
  adv_examples = create_or_load_pickle(pkl_file, label, overwrite)
  # Run experiments
  with tqdm(total=len(dataset)*len(budgets), desc="Adv. Examples") as pbar:
    for budget in budgets:
      if budget not in adv_examples[label]:
        adv_examples[label][budget] = {}
      for data in dataset:
        id = f"{data['docid']}-{data['segid']}"
        if id not in adv_examples[label][budget]:
          objective = VisrepOcrObjective(data['german'], budget, translate, data['english'])
          adv_examples[label][budget][id] = objective.differential_evolution(maxiter, popsize)
          with open(pkl_file, 'wb') as f:
            pickle.dump(adv_examples, f)
        pbar.update(1)

if __name__ == '__main__':

  parser = ArgumentParser(description='Adversarial NLP Experiments.')
  model = parser.add_mutually_exclusive_group(required=True)
  model.add_argument('-t', '--trocr', action='store_true', help="Target Microsoft TrOCR model.")
  model.add_argument('-x', '--toxic', action='store_true', help="Target IBM's MaxToxic model defended by TrOCR.")
  model.add_argument('-r', '--translation', action='store_true', help="Target Facebook Fairseq's WMT 14 EN->FR translation model defended by TrOCR.")
  model.add_argument('-v', '--visrep', action='store_true', help="Target Visual Text Translation DE->EN model.")
  parser.add_argument('-c', '--cpu', action='store_true', help="Use CPU for ML inference instead of CUDA.")
  parser.add_argument('pkl_file', help="File to contain Python pickled output.")
  parser.add_argument('-s', '--start-index', type=int, default=0, help="The lower bound of the items in the dataset to use in experiments.")
  parser.add_argument('-e', '--end-index', type=int, default=500, help="The upper bound of the items in the dataset to use in experiments.")
  parser.add_argument('-l', '--min-budget', type=int, default=0, help="The lower bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-u', '--max-budget', type=int, default=5, help="The upper bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-a', '--maxiter', type=int, default=10, help="The maximum number of iterations in the genetic algorithm.")
  parser.add_argument('-p', '--popsize', type=int, default=32, help="The size of the population in the genetic algorithm.")
  parser.add_argument('-o', '--overwrite', action='store_true', help="Overwrite existing results file instead of resuming.")
  parser.add_argument('-f', '--font', default="arialuni.ttf", help="TTF font file to use for image rendering.")
  parser.add_argument('-F', '--font-size', type=int, default=32, help="Font size for image rendering.")
  parser.add_argument('-n', '--ocr-line-len', type=int, default=64, help="Max characters for words rendered to OCR image.")
  parser.add_argument('-k', '--checkpoint', default="WMT_de-en/checkpoint_best.pt", help="Model checkpoint filepath to use for inference. Visrep only.")
  parser.add_argument('-g', '--tgt-dict', default="WMT_de-en/dict.en.txt", help="Target dictionary filepath to use for inference. Visrep only.")
  args = parser.parse_args()

  load_font(args.font, args.font_size)

  if args.trocr:
    trocr_experiment(**vars(args))

  elif args.toxic:
    toxic_experiment(**vars(args))

  elif args.translation:
    translation_experiment(**vars(args))

  elif args.visrep:
    visrep_experiment(**vars(args))