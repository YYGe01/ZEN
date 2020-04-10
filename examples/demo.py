import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys

sys.path.append('../')

import logging
import torch
from tqdm import tqdm, trange

from ZEN import ZenNgramDict
from ZEN import BertTokenizer
from ZEN import ZenForTokenClassification
import torch.nn.functional as F
from utils_token_level_task import processors, convert_examples_to_features
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from ZEN import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME, NGRAM_DICT_NAME
import re
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.label = label


def combine_example(text):
    examples = []
    text = text.strip()
    label = ['S'] * len(text)
    text_a = ' '.join(list(text))

    examples.append(InputExample(text_a=text_a, label=label))
    return examples


def load_examples(tokenizer, ngram_dict, label_list, text):
    examples = combine_example(text)

    features = convert_examples_to_features(examples, label_list, 128, tokenizer, ngram_dict)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ngram_ids,
                         all_ngram_positions,
                         all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks, all_valid_ids, all_lmask_ids)

def _single_decode(sent,tag):
    # sent, tag = args
    # sent[1:-1], tag[1:-1] = args
    cur_sent= []
    t1 = []
    for i in range(len(sent)):
        word = sent[i]
        if tag[i] in 'SB':
            if len(t1) != 0:
                cur_sent.append(''.join(t1))
            t1 = [word]
        elif tag[i] in 'IE':
            t1.append(word)
    if len(t1) != 0:
        cur_sent.append(''.join(t1))
    return cur_sent
def predict(model, tokenizer, ngram_dict, label_list, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(label_list) + 1
    eval_dataset = load_examples(tokenizer, ngram_dict, label_list, text)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 64)

    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, ngram_ids, ngram_positions, \
        ngram_lengths, ngram_seg_ids, ngram_masks, valid_ids, l_mask = batch

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=valid_ids,
                           attention_mask_label=None, ngram_ids=ngram_ids, ngram_positions=ngram_positions)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()

        for i, label in enumerate(label_ids):
            for j, m in enumerate(label):
                if j == 0:
                    continue
                if label_ids[i][j] == num_labels - 1:
                    break
                y_true.append(label_map[label_ids[i][j]])
                y_pred.append(label_map[logits[i][j]])

    # print(y_pred)
    # print(len(y_pred))
    cur_sent = _single_decode(text,y_pred)
    return cur_sent
def split_text(text, delimiter="。"):
    pattern = re.compile(r"([(%s)])"%delimiter)
    line = re.split(pattern,text)
    line_text = line[0::2]
    line_sym = line[1::2]
    sentences = ["".join(i) for i in zip(line_text,line_sym)]
    if len(line_text) != len(sentences):
        sentences.append(line_text[-1])
    return sentences

def load_train_model():
    bert_model = '../models/checkpoint'
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
    processor = processors['cwspku']()
    label_list = processor.get_labels()
    model = ZenForTokenClassification.from_pretrained(bert_model,
                                                      cache_dir=cache_dir,
                                                      num_labels=len(label_list) + 1,
                                                      multift=False)
    # model.load_state_dict(torch.load('./results/result-tokenlevel-2020-03-31-15-52-51/checkpoint-27/pytorch_model.bin'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    ngram_dict = ZenNgramDict(bert_model, tokenizer=tokenizer)
    params = model, tokenizer, ngram_dict, label_list
    return params
def output_file(params,texts):
    model, tokenizer, ngram_dict, label_list = params
    max_seq_len = 128
    results = []
    def split_text(text, delimiter="。"):
        results = []
        pattern = re.compile(r"([(%s)])" % delimiter)
        line = re.split(pattern, text)
        line_text = line[0::2]
        line_sym = line[1::2]
        sentences = ["".join(i) for i in zip(line_text, line_sym)]
        if len(line_text) != len(sentences):
            sentences.append(line_text[-1])
        if len(max(sentences, key=len)) <= max_seq_len - 2:
            for t in sentences:
                if len(t) != 0:
                    seq = predict(model, tokenizer, ngram_dict, label_list, t)
                    f.write(' '.join(seq) + " ")
                    results.extend(seq)
            f.write("\n")

        return results
    with open("../output/pred_sent.txt","w") as f:
        for i,text in enumerate(texts):
            try:
                text = text.strip()
                results = []
                if not text.strip():
                    continue
                if len(text) > max_seq_len - 2:
                    if len(results) == 0:
                        results = split_text(text, delimiter="。")
                    if len(results) == 0:
                        results = split_text(text, delimiter="！")
                    if len(results) == 0:
                        results = split_text(text, delimiter="？")
                    if len(results) == 0:
                        results = split_text(text, delimiter="；")
                    if len(results) == 0:
                        results = split_text(text, delimiter="，")
                    if len(results) == 0:
                        results = split_text(text, delimiter="、")
                    if len(results) == 0:
                        results = split_text(text, delimiter="《")
                    if len(results) == 0:
                        results = split_text(text, delimiter="。，；？、！《")
                    if len(results) == 0:
                        print("Ignore line: " + text)
                else:
                    seq = predict(model, tokenizer, ngram_dict, label_list, text)
                    if not seq:
                        print(text)
                    f.write(' '.join(seq))
                    f.write("\n")
                    results.extend(seq)
            except:
                print(len(texts))
                print(text)
                print(len(text))
                print(i)
            # print(results)
            # return results
    return results
if __name__ == '__main__':
    params = load_train_model()

    # texts = ['根据签订的金融合作协议，国家开发银行将在“十五”期间为北京市城市轻轨、高速公路、公路联络线、信息网络、天然气、电力、大气治理和水资源环保等提供人民币500亿元左右的贷款支持；同意三年内为中关村科技园区建设提供人民币总量80亿至100亿元的贷款支持；对推荐的项目优先安排评审，实行评审“快通道”，对信用等级较高的企业将给予一定的授信；通过参与企业资本运作，防范信贷风险；受北京市政府委托为首都制定经济结构调整规划、整合创新资源、培育风险投资体系、产业结构调整和升级、重大项目决策和可行性研究、项目融资、银团贷款和资产重组等提供金融顾问服务等。北京市政府将积极帮助协调开发银行贷款项目的评审和管理、本息回收、资产保全等，促进开发银行信贷风险的防范和化解。']
    #        # '因有关日寇在京掠夺文物详情，藏界较为重视。',
    #        # '藏界较为重视，也是我们收藏北京史料中的要件之一。',
    #        # '因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。',
    #        # '因有关日寇在京掠夺文物详情，藏界较为重件之一。',
    #        # '因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件。']
    # with open("../datasets/pku_test.utf8") as f:
    #     texts = f.readlines()
    #     print(len(texts))
    # print(len(texts[0]))

    texts = ['根据签订的金融合作协议，国家开发银行将在“十五”期间为北京市城市轻轨、高速公路、公路联络线、信息网络、天然气、电力、大气治理和水资源环保等提供人民币500亿元左右的贷款支持。']
    results = output_file(params, texts)
    print(results)
