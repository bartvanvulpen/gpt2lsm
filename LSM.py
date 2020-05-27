import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)


class LSMLoss:

    def __init__(self, input_ids, output_logits, token_type_ids, tokenizer):
        """
        Initializing LSM Loss
        :param input_ids: input ids of the input batch
        :param output_logits: output logits of the batch
        :param token_type_ids: token type ids
        :param tokenizer: (GPT2) tokenizer object
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.output_logits = output_logits
        self.tokenizer = tokenizer
        self.s1 = self.input_ids.size()[0]
        self.s2 = self.input_ids.size()[1]
        self.s3 = self.input_ids.size()[2]
        self.s4 = len(tokenizer)

    def get_fw_dict(self):
        """
        Get dictionary of function words
        :return: Dictionary of function words
         """
        d = {
            'conjunctions': ['and', 'but', 'or', 'as', 'if', 'when', 'because', 'while', 'where', 'although', 'whether',
                             'before', 'since', 'so', 'though', 'until', 'after', 'cos', 'for', '&', 'nor', 'unless',
                             'once', 'whereas', 'whilst', 'rather than', 'and/or', 'even when', 'albeit', 'given that',
                             'provided that'],
            'auxiliary_verbs': ['be', 'is', 'are', 'were', 'was', 'been', 'am', 'being', 'have', 'has', 'was', 'were',
                                'would', 'will', 'do', 'can', 'could', 'dare', 'does', 'did', 'had', 'having', 'may',
                                'might', 'must', 'need', 'ought', 'shall', 'should', "'ll", "'d"],
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

    def encode_fw_dict(self, fw_d):
        """
        This function encodes the function word dictionary with the GPT2 tokenizer.
        Adds <bos> for correct encoding, then later removes it.
        :param fw_d: dictionary of function words for each category
        :return: encoded dictionary tensor + its length
        """
        encoded_d = []
        for key, value in fw_d.items():
            fw_ids = torch.tensor(self.tokenizer.encode(' '.join(['<bos>'] + value)))[1:].cuda()
            fw_distr_sm = F.softmax(torch.zeros([len(self.tokenizer)], dtype=torch.float).cuda().scatter(0, fw_ids, 1.0), dim=-1)
            encoded_d.append(fw_distr_sm)
        return torch.stack(encoded_d), len(encoded_d)

    def encode_input_ids(self, ids):
        """
        This function encodes the input ids into a one hot vector of the vocabulary.
        Then to create a smoothened distribution, without zeros (to avoid division by zero errors).
        :param ids: input_ids
        :return: Softmax of encoded input id's
        """
        encoded_ids = []
        for id in ids:
            encoded_ids.append(F.log_softmax(torch.zeros([len(self.tokenizer)], dtype=torch.float).cuda().scatter(0, id, 1.0), dim=-1))
        return torch.stack(encoded_ids)

    def mask_input(self):
        """
        This function masks the input, so the focus will only be on the Language Style of the
        conversation partner <speaker2> (aka the dialogue partner).
        All history utterances of <speaker1> (aka the chatbot) are padded with <pad>.
        This masked input is being used to calculate the use-of-function-words (Language Style)
        of <speaker2> (aka dialogue partner), which will be compared to the LSM of <speaker1> (aka chatbot).
        :param input: batch input tensor
        :return: masked input tensor
        """
        sp1 = self.tokenizer.convert_tokens_to_ids('<speaker1>')
        sp2 = self.tokenizer.convert_tokens_to_ids('<speaker2>')
        bos = self.tokenizer.convert_tokens_to_ids('<bos>')
        eos = self.tokenizer.convert_tokens_to_ids('<eos>')
        pad = self.tokenizer.convert_tokens_to_ids('<pad>')

        for i, choices in enumerate(self.input_ids):
            for j, ids in enumerate(choices):
                speaker2 = False
                for k, word in enumerate(ids):
                    word = int(word.cpu().numpy())
                    if word == eos:
                        continue
                    if speaker2 == False:
                        self.input_ids[i][j][k] = pad
                    if word == sp2:
                        speaker2 = True
                    if word == sp1:
                        self.input_ids[i][j][k] = pad
                        speaker2 = False
        return self.input_ids
    def mask_output(self):
        """
        This function adds the history of <speaker1> to the output distribution to calculate
        the KL divergence values of each word of the history of utterances of speaker1 as well and eventually
        the use-of-function-words of speaker 1 (aka the chatbot).
        :return: output filled with history, so use-of-function-words (Language Style) calculation of <speaker1>
        will also be based on its history.
        """
        return 1

    def reshape_input_output(self):
        """
        This function reshapes the tensor with input ids, then encodes it with encode_input_ids, that
        return the log-softmax of the input distribution, that will be used for the KL divergence
        It also reshapes the tensor with output logits and returns its log-softmax
        :return: reshaped input and output (log-softmax)
        """

        # shape input
        input_ids_reshaped = self.input_ids.view(-1)
        input = self.encode_input_ids(input_ids_reshaped)

        # shape output
        lm_logits_reshaped = self.output_logits.view(-1, self.output_logits.size(-1))
        output = F.log_softmax(lm_logits_reshaped, dim=-1)  # to be used for KL divergence

        return input, output



    def calc_LSM_loss(self, kl1, kl2):
        """
        LSM from paper: If seq1 has 40% articles and seq2 0%, the score is 0.0 in the old version,
        the same when seq1 has 30% articles and seq2 0%, the score is still 0.0.
        This LSM method: However, with this score, the score gradually decreases as more articles are
        in seq1 and 0 in seq2. But still, if they are equal, LSM = 1.0
        :param kl1: Inverted KL Divergence scores of sentence(s)
        :param kl2: Inverted KL Divergence scores of sentence(s)
        :return: Inverted LSM Score, so it becomes a loss.
        """

        cat = torch.cat((kl1, kl2), 0)
        p1, p2 = torch.split(cat / torch.max(cat), int(len(cat) / 2))
        m1, m2 = torch.mean(p1), torch.mean(p2)
        LSM_score = 1 - (abs(m1 - m2) / (m1 + m2))

        return (LSM_score ** -1) / len(kl1)

    def calc_kl_divs(self, d, distr):

        """
        This function calculates for each distribution of a word in the batch of sentences the
        inverted KL Divergence value, in order to give a measure of how likely the distribution
        of this word is to be from a specific function word category.
        :param d: dictionary with function words
        :param distr: distributions of the words in a batch of sentences (distr_size = size of vocab)
        :return: inverted kl divergence value for each word in the sentence/batch of sentences, for each category
        """


        # calculate KL Divergence values for given distribution
        measures_output = []
        for fw in d:
            m = torch.Tensor([F.kl_div(o, fw, reduction='sum') for o in distr]).cuda()
            measures_output.append(m.pow(-1))
        kl_divs = torch.stack(measures_output)

        # reshape
        kl_divs = kl_divs.view(len(d), self.s1 * self.s2, self.s3)

        return kl_divs


    def loss(self):
        """
        This function combines all functions above and calculates the LSM loss.
        :return: LSM loss and kl_div values (to trace during training)
        """

        print("Loading dict and encoding it...")
        # get load and encode dict of function words
        d = self.get_fw_dict()
        d, d_len = self.encode_fw_dict(d)

        print("Masking input and reshaping input+output...")
        # mask input_ids, reshape input_ids tensor, also reshape output logits tensor
        self.mask_input()
        input, output = self.reshape_input_output()

        print("Calculating KL divergence values...")
        # calc KL divs
        kl_divs_input = self.calc_kl_divs(d, input)
        kl_divs_output = self.calc_kl_divs(d, output)

        print("Calculating LMS scores...")
        # calculate LSM scores for each sentence
        losses = []
        for i, cat in enumerate(kl_divs_input):
            lsm_loss_cat = torch.mean(torch.Tensor([self.calc_LSM_loss(kl_divs_output[i][j], input_kls) for j, input_kls in enumerate(cat)]).cuda())
            losses.append(lsm_loss_cat)

        print("Taking mean of LSM scores...")
        # final loss is the mean of the LSM scores for each category
        final_lsm_loss = torch.mean(torch.stack(losses))

        print("DONE")
        return final_lsm_loss, kl_divs_input[0][0], kl_divs_output[0][0]


### TEST CODE ###

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

print("Initializing tokenizer...")
tokenizer_class = GPT2Tokenizer
tokenizer = tokenizer_class.from_pretrained("gpt2-medium")

print("Initializing model...")
model_class = GPT2DoubleHeadsModel
model = model_class.from_pretrained("gpt2-medium", output_hidden_states=True)
# Add special tokens if they are not already added
print("Adding special tokens...")
add_special_tokens_(model, tokenizer)

print("Loading test tensors of input (+token_type_ids) and output")
# load input ids and output logits
input = torch.load('transfer-learning-conv-ai/input1_0.pt')
token_type_ids = torch.load('transfer-learning-conv-ai/token_type_ids_0.pt')
output = torch.load('transfer-learning-conv-ai/output_distr_logits0.pt')

# calculate and print loss
print("Initializing loss...")
X = LSMLoss(input, output, token_type_ids, tokenizer)
print("Started calculating loss...")
loss, _, _ = X.loss()

print(loss)