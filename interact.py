# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import string
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import numpy as np
import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


def get_fw_dict():
    """
    Get dictionary of function words
    'm, 'll, 've and 'd have been replaced with m, ll, ve, d respectively
    :return: Dictionary of function words
     """
    d = {
        'conjunctions': ['and', 'but', 'or', 'as', 'if', 'when', 'because', 'while', 'where', 'although', 'whether',
                         'before', 'since', 'so', 'though', 'until', 'after', 'cos', 'for', '&', 'nor', 'unless',
                         'once', 'whereas', 'whilst', 'rather than', 'and/or', 'even when', 'albeit', 'given that',
                         'provided that'],
        'auxiliary_verbs': ['be', 'is', 'are', 'were', 'was', 'been', 'm', 'being', 'have', 'has', 'was', 'were',
                            'would', 'will', 'do', 'can', 'could', 'dare', 'does', 'did', 'had', 'having', 'may',
                            'might', 'must', 'need', 'ought', 'shall', 'should', "ll", "d", "ve", 's'],
        'personal_pronouns': ['i', 'you', 'he', 'they', 'she', 'we', 'who', 'them', 'him', 'me', 'her', 'us',
                              'himself', 'themselves', 'someone', 'herself', 'anyone', 'everyone', 'whom', 'myself',
                              'each other', 'yourself', 'no one', 'somebody', 'nobody', 'everybody', 'anybody',
                              'his', 'mine', 'ourselves', 'yours', 'one another', 'hers', 'no-one', 'ours',
                              'theirs', 'his', 'their', 'her', 'my', 'your', 'our'],
        'impersonal_pronouns': ['it', 'its', 'they', 'that', 'this', 'them', 'something', 'nothing', 'anything',
                                'itself', 'themselves', 'itself', 'everything', 'each other', 'everything',
                                'something', "'em"],
        'prepositions': ['of', 'in', 'to', 'for', 'with', 'on', 'by', 'at', 'from', 'as', 'into', 'about', 'like',
                         'after', 'between', 'through', 'over', 'against', 'under', 'without', 'within', 'during',
                         'before', 'towards', 'around', 'upon', 'including', 'among', 'across', 'off', 'behind',
                         'since', 'rather than', 'until', 'according to', 'up to', 'despite', 'near', 'above',
                         'per', 'along', 'away from', 'throughout', 'outside', 'round', 'beyond', 'worth', 'down',
                         'on to', 'up', 'due to', 'inside', 'plus'],
        'adverbs': ['so', 'up', 'then', 'out', 'now', 'only', 'just', 'more', 'also', 'very', 'well', 'how', 'down',
                    'back', 'on', 'there', 'still', 'even', 'too', 'here', 'where', 'however', 'over', 'in', 'as',
                    'most', 'again', 'never', 'why', 'off', 'really', 'always', 'about', 'when', 'quite', 'much',
                    'both', 'often', 'away', 'perhaps', 'right', 'already', 'yet', 'later', 'almost', 'of course',
                    'far', 'together', 'probably', 'today', 'actually', 'ever', 'at least', 'enough', 'less',
                    'for example', 'therefore', 'particularly', 'either', 'around', 'rather', 'else', 'sometimes',
                    'thus', 'ago', 'yesterday', 'home', 'all', 'usually'],
        'quantifiers': ['all', 'some', 'any', 'many', 'more', 'another', 'much', 'each', 'few', 'most', 'both',
                        'several', 'half', 'little', 'whatever', 'less', 'enough', 'either', 'fewer', 'neither',
                        'a lot', 'least', 'a bit', 'a great deal', 'plenty'], 'articles': ['a', 'an', 'the']}

    return d
def split_history(history, tokenizer):
    """
    This function splits the history of a dialogue into the history of the chatbot and the conversation partner.
    :param history: the history ID's of the history
    :param tokenizer: the Tokenizer
    :return: a string containing the history of the conversation partner and a list of string-utterances of the
    chatbot history
    """

    # split the history
    partner_hist = history[::2]
    chatbot_hist = history[1::2]
    partner_hist2 = []
    chatbot_hist2 = []

    # convert to strings
    for index, p_utt in enumerate(partner_hist):

        try:
            c_utt = chatbot_hist[index]
        except:
            c_utt = False

        dec_p_utt = tokenizer.decode(p_utt)
        partner_hist2.append(dec_p_utt)

        if c_utt != False:
            dec_c_utt = tokenizer.decode(c_utt)
            chatbot_hist2.append(dec_c_utt)

    # remove punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))

    # refine the string for counting substrings
    hist_p_string = (' ' + ' '.join(partner_hist2).lower() + ' ').replace("'", ' ').translate(table)

    return hist_p_string, chatbot_hist2

def wd_logits(logits_, differences, w, d, tokenizer):
    """
    This function applies the actual weighted decoding to the given output logits.
    :param logits: output logits
    :param differences: the percentual differences in function word use beteen chatbot and conversation partner
    :param w: hyperparameter w, the weight
    :param d: the function word dictiopary
    :param tokenizer: the used tokenizer
    :return: the logits after applying weighted decoding
    """

    for j, diff in enumerate(differences):
        fws = ' ' + ' '.join(list(d.values())[j])
        fws_ids = tokenizer.encode(fws)
        for j, fw in enumerate(fws_ids):
            logits_[fw] = logits_[fw] + (w * diff)
    return logits_

def calc_fw_perc_diffs(dictio, str1, str2):
    """
    This function calculates the percentual differences of function word use
    between two strings for each function word category.
    :param dictio: the function word dictionary
    :param str1: input string 1
    :param str2: input string 2
    :return: the percentual difference of function word use and the percentages of function word use for each string
    """
    percs1 = []
    percs2 = []
    for cat in dictio.values():
        cat_count1 = 0
        cat_count2 = 0
        for fw in cat:
            fw = ' ' + fw + ' '
            cat_count1 += str1.count(fw)
            cat_count2 += str2.count(fw)

        percs1.append(cat_count1 / len(str1.split()))

        if len(str2.split()) != 0:
            percs2.append(cat_count2 / len(str2.split()))
        else:
            percs2.append(0)

    return np.array(percs1) - np.array(percs2)

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, history_wd, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    # use weighted decoding when wd-argument is set to True
    if args.wd:
        d = get_fw_dict()
        table = str.maketrans(dict.fromkeys(string.punctuation))
        hist_p_string, hist_c = split_history(history_wd, tokenizer)


    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature

        # use pretrained tokenizer for Weighted decoding testing (otherwise latin-1 error)
        if args.wd:
            dec_curr_output = tokenizer.decode(current_output)
            hist_c.append(dec_curr_output)
            hist_c_string = (' ' + ' '.join(hist_c).lower() + ' ').replace("'", ' ').translate(table)

            cat_diffs = calc_fw_perc_diffs(d, hist_p_string, hist_c_string)

            w = args.wd_weight

            logits = wd_logits(logits, cat_diffs, w, d, tokenizer)

        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

        if args.wd:
            # reset hist
            hist_c = hist_c[:-1]

    if args.wd:
        hist_c.append(current_output)
    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--wd", type=bool, default=False)
    parser.add_argument("--wd_weight", type=float, default=0.0)
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))


    # added a history for wd, to avoid the max history to have influence during weighted decoding.
    history = []
    history_wd = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        history_wd.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, history_wd, tokenizer, model, args)
        history_wd.append(out_ids)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()
