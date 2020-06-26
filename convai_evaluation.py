# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
import logging
import numpy as np
from copy import deepcopy
from pprint import pformat
from collections import defaultdict
from functools import partial
from tqdm import trange
import string
import torch
import torch.nn.functional as F
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from projects.convai2.eval_hits import eval_hits, setup_args as setup_args_hits
from projects.convai2.eval_f1 import eval_f1, setup_args as setup_args_f1
from projects.convai2.eval_ppl import eval_ppl, setup_args as setup_args_ppl
from projects.convai2.build_dict import build_dict
from transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Tokenizer)

from train import build_input_from_segments, pad_dataset, SPECIAL_TOKENS, add_special_tokens_
from utils import download_pretrained_model, AttrDict

def get_fw_dict():
    """
    Get dictionary of function words
    'm, 'll, 've and 'd have been replaced with m, ll, ve, d respectively, 'its' is also added
    :return: Dictionary of function words
     """
    d = {
        'conjunctions': ['and', 'but', 'or', 'as', 'if', 'when', 'because', 'while', 'where', 'although', 'whether',
                         'before', 'since', 'so', 'though', 'until', 'after', 'cos', 'for', '&', 'nor', 'unless',
                         'once', 'whereas', 'whilst', 'rather than', 'and/or', 'even when', 'albeit', 'given that',
                         'provided that'],
        'auxiliary_verbs': ['be', 'is', 's', 'are', 'were', 'was', 'been', 'am', 'm', 've', 'its', 'being', 'have', 'has', 'was', 'were',
                            'would', 'will', 'do', 'can', 'could', 'dare', 'does', 'did', 'had', 'having', 'may',
                            'might', 'must', 'need', 'ought', 'shall', 'should', 'll', 'd'],
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
    This function applies the actual weighted decoding to the
    given output logits.
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

    return torch.tensor(np.array(percs1) - np.array(percs2)).cuda(), np.array(percs1), np.array(percs2)

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
    """
    This function not only samples a sequence, but additionally calculates the lsm metrics

    :return: samples sequence
    """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    # use weighted decoding
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

            cat_diffs, _, _ = calc_fw_perc_diffs(d, hist_p_string, hist_c_string)

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


# init lsm score lists
lsm_model_list = []
lsm_human_list = []

class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model. Must be OpenAIGPT.")
        agent_args.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        agent_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        agent_args.add_argument("--eval_type", type=str, default="hits@1", help="hits@1, ppl or f1")
        agent_args.add_argument("--no_sample", action='store_true')
        agent_args.add_argument("--max_length", type=int, default=20)
        agent_args.add_argument("--wd", type=bool, default=False)
        agent_args.add_argument("--wd_weight", type=float, default=0.0)
        agent_args.add_argument("--min_length", type=int, default=1)
        agent_args.add_argument("--seed", type=int, default=0)
        agent_args.add_argument("--temperature", type=int, default=0.7)
        agent_args.add_argument("--top_k", type=int, default=20)
        agent_args.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        args = AttrDict(opt)  # to keep most commands identical to the interact.py script
        self.args = args

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(args))

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if shared is None:
            self.logger.info("Get pretrained model and tokenizer")
            if args.model_checkpoint == "":
                args.model_checkpoint = download_pretrained_model()
            if 'gpt2' in args.model_checkpoint:
                self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
                model_class = GPT2DoubleHeadsModel if self.args.eval_type == "hits@1" else GPT2LMHeadModel
            else:
                self.tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
                model_class = OpenAIGPTDoubleHeadsModel if self.args.eval_type == "hits@1" else OpenAIGPTLMHeadModel

            self.model_checkpoint = model_class.from_pretrained(args.model_checkpoint)
            self.model_checkpoint.to(args.device)

            self.logger.info("Build BPE prefix dictionary")
            convai_dict = build_dict()
            assert len(convai_dict) == 19304
            self.prefix2words = self.get_prefix2words(convai_dict)
        else:
            self.model_checkpoint = shared['model']
            self.tokenizer = shared['tokenizer']
            self.prefix2words = shared['prefix2words']
        add_special_tokens_(self.model_checkpoint, self.tokenizer)
        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        self.persona = []
        self.history_wd = []
        self.history = []
        self.labels = []
        self.d = get_fw_dict()


        self.reset()

    def observe(self, observation):
        if self.episode_done:

            # print current LSM means, the last printed LSM mean is the final LSM mean.
            # can't create a final sum up, because we're using an ParlAI agent class.
            if lsm_model_list != []:
                print('Current LSM model mean:', torch.mean(torch.stack(lsm_model_list)))
                print('Current LSM human mean:', torch.mean(torch.stack(lsm_human_list)))
                print('-----------------------------------------------------------------')
            self.reset()

        if self.labels:

            # Add the previous response to the history
            self.history_wd.append(self.labels)
            self.history.append(self.labels)

        if 'labels' in observation or 'eval_labels' in observation:
            text = observation.get('labels', observation.get('eval_labels', [[]]))[0]
            self.labels = self.tokenizer.encode(text)

        if 'text' in observation:
            text = observation['text']
            for subtext in text.split('\n'):
                subtext = subtext.strip()
                if subtext.startswith('your persona:'):
                    subtext = subtext.replace('your persona:', '').strip()
                    self.persona.append(self.tokenizer.encode(subtext))
                else:
                    self.history.append(self.tokenizer.encode(subtext))
                    self.history_wd.append(self.tokenizer.encode(subtext))

        self.history = self.history[-(2*self.args.max_history+1):]

        candidates = []
        if 'label_candidates' in observation:
            for candidate in observation['label_candidates']:
                candidates.append((self.tokenizer.encode(candidate), candidate))
        self.candidates = candidates

        self.episode_done = observation['episode_done']
        self.observation = observation
        return observation

    def act(self):
        reply = {}

        if self.args.eval_type == "hits@1" and len(self.candidates) > 0:
            instances = defaultdict(list)
            for candidate, _ in self.candidates:
                instance = build_input_from_segments(self.persona, self.history, candidate, self.tokenizer)
                for input_name, input_array in instance.items():
                    instances[input_name].append(input_array)

            inputs = pad_dataset(instances, padding=self.special_tokens_ids[-1])

            tensor_inputs = {}
            for input_name in ["input_ids", "mc_token_ids", "token_type_ids"]:
                tensor = torch.tensor(inputs[input_name], device=self.args.device)
                tensor = tensor.view((-1, len(self.candidates)) + tensor.shape[1:])
                tensor_inputs[input_name] = tensor

            with torch.no_grad():
                mc_logits = self.model_checkpoint(**tensor_inputs)[1]

            val, ind = torch.sort(mc_logits[0], descending=True)

            ypred = self.candidates[ind[0].item()][1] # match
            tc = []
            for j in range(len(self.candidates)):
                tc.append(self.candidates[ind[j].item()][1])
            reply = {'text': ypred, 'text_candidates': tc}


        elif self.args.eval_type == 'f1':
            # We are in interactive of f1 evaluation mode => just sample
            with torch.no_grad():
                out_ids = sample_sequence(self.persona, self.history, self.history_wd, self.tokenizer, self.model_checkpoint, self.args)
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
            reply = {'text': out_text}

            # integrate it LSM Metrics, during sampling with F1 eval type (decreases validation commands).
            # For every response to a given dialogue history, we calculate the LSM score
            # This score is added to a total list of scores and meaned at the end
            # We separate a model response with a label response, so that results in 2 LSM scores.
            response = reply['text']
            dialogue_hist = self.history_wd

            table = str.maketrans(dict.fromkeys(string.punctuation))

            # split history into history of speaker 2 and speaker 1
            speaker2_hist_string, speaker1_chatbot_hist_list = split_history(dialogue_hist, self.tokenizer)

            # use deepcopy to avoid variable troubles
            label_c_hist = deepcopy(speaker1_chatbot_hist_list)

            # add response generated by model
            speaker1_chatbot_hist_list.append(response)

            # add label to chatbot hist
            label_c_hist.append(self.tokenizer.decode(self.labels))

            # convert to strings
            pred_c_hist_string = (' ' + ' '.join(speaker1_chatbot_hist_list).lower() + ' ').replace("'", ' ').translate(
                table)
            label_c_hist_string = (' ' + ' '.join(label_c_hist).lower() + ' ').replace("'", ' ').translate(table)

            # results in two vectors containing the function word usage percentage for each category
            # we use prediction and labeled, so we get 2 LSM score eventually at the end of evaluation
            _, p1_model, p2_model = calc_fw_perc_diffs(self.d, pred_c_hist_string, speaker2_hist_string)
            _, p1_human, p2_human = calc_fw_perc_diffs(self.d, label_c_hist_string, speaker2_hist_string)

            # calculate LSM score for the model response and calculate LSM score for the label response
            LSMs_model = torch.tensor([1-(abs(p1 - p2_model[i]) / (p1 + p2_model[i] + 0.00000001)) for i, p1 in enumerate(p1_model)]).cuda()
            LSMs_human = torch.tensor([1-(abs(p1 - p2_human[i]) / (p1 + p2_human[i] + 0.00000001)) for i, p1 in enumerate(p1_human)]).cuda()

            lsm_model_list.append(torch.mean(LSMs_model))
            lsm_human_list.append(torch.mean(LSMs_human))

        else:
            # We are in interactive of f1 evaluation mode => just sample
            with torch.no_grad():
                out_ids = sample_sequence(self.persona, self.history, self.history_wd, self.tokenizer,
                                          self.model_checkpoint, self.args)
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
            reply = {'text': out_text}


        return reply

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.
        """
        partial_out_ids = self.tokenizer.encode(' '.join(partial_out))
        instance = build_input_from_segments(self.persona, self.history, partial_out_ids,
                                             self.tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=self.args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=self.args.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model_checkpoint(input_ids, token_type_ids=token_type_ids)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        probs = F.softmax(logits[0, -1], dim=0)

        dist = {}
        for prefix_id, words in self.prefix2words.items():
            for word, ratio in words.items():
                dist[word] = probs[prefix_id].item() * ratio
        return dist

    def get_prefix2words(self, convai_dict, smoothing_freq=5):
        """ map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts) """
        prefix2words = defaultdict(dict)
        for i in trange(len(convai_dict)):
            word = convai_dict[i]
            freq = convai_dict.freq[word] + smoothing_freq
            bpe_tokens = self.tokenizer.bpe(word).split(' ')
            prefix_id = self.tokenizer.convert_tokens_to_ids(bpe_tokens[0])
            prefix2words[prefix_id].update(dict([(word, freq)]))

        for prefix_id, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix_id] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model_checkpoint
        shared['prefix2words'] = self.prefix2words
        return shared

    def reset(self):
        self.persona = []
        self.history = []
        self.history_wd = []
        self.labels = []
        self.candidates = []
        self.episode_done = True
        self.observation = None



if __name__ == '__main__':

    parser = base_setup_args(None)
    parser.set_params(
        model='convai_evaluation:TransformerAgent')
    opt = parser.parse_args(print_args=False)

    if opt['eval_type'] == "hits@1":
        setup_args = setup_args_hits(None)
        eval_fct = partial(eval_hits, print_parser=setup_args)
    elif opt['eval_type'] == "ppl":
        setup_args = setup_args_ppl(None)
        eval_fct = eval_ppl
    elif opt['eval_type'] == "f1":
        setup_args = setup_args_f1(None)
        eval_fct = partial(eval_f1, print_parser=setup_args)
    else:
        raise ValueError

    setup_args.set_params(
        model='convai_evaluation:TransformerAgent')
    opt = setup_args.parse_args(print_args=False)
    eval_fct(opt)


    if lsm_model_list != []:
        print('FINAL LSM MODEL SCORE:', sum(lsm_model_list)/len(lsm_model_list))
        print('FINAL LSM HUMAN SCORE:', sum(lsm_human_list)/len(lsm_human_list))
    else:
        print('Empty, no LSM measure done')