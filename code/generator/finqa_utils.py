"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip

from config import parameters as conf


sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text


def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                num = text
            else:
                num = None
    return num


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            assert cur_num_idx != -1
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 question,
                 input_ids,
                 input_mask,
                 option_mask,
                 segment_ids,
                 options,
                 answer=None,
                 program=None,
                 program_ids=None,
                 program_weight=None,
                 program_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.question = question
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.option_mask = option_mask
        self.segment_ids = segment_ids
        self.options = options
        self.answer = answer
        self.program = program
        self.program_ids = program_ids
        self.program_weight = program_weight
        self.program_mask = program_mask


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if conf.pretrained_model in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens


def _detokenize(tokens):
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    if len(question_tokens) >  max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            if is_training == True:
                # print("\n")
                # print("################")
                # print("number not in input")
                # print(example.original_question)
                # print(tokens)
                # print(len(tokens))
                # print(example.numbers[ind])
                # print(offset)

                # invalid example, drop for training
                return features

            # assert is_training == False



    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        program_ids = prog_token_to_indices(program, numbers, number_indices,
                                            max_seq_length, op_list, op_list_size,
                                            const_list, const_list_size)
        program_mask = [1] * len(program_ids)
        program_ids = program_ids[:max_program_length]
        program_mask = program_mask[:max_program_length]
        if len(program_ids) < max_program_length:
            padding = [0] * (max_program_length - len(program_ids))
            program_ids.extend(padding)
            program_mask.extend(padding)
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    features.append(
        InputFeatures(
            unique_id=-1,
            example_index=-1,
            tokens=tokens,
            question=example.original_question,
            input_ids=input_ids,
            input_mask=input_mask,
            option_mask=option_mask,
            segment_ids=segment_ids,
            options=example.options,
            answer=example.answer,
            program=program,
            program_ids=program_ids,
            program_weight=1.0,
            program_mask=program_mask))
    return features


def read_mathqa_entry(entry, tokenizer):
    
    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""


    if conf.retrieve_mode == "single":
        for ind, each_sent in entry["qa"]["model_input"]:
            context += each_sent
            context += " "
    elif conf.retrieve_mode == "slide":
        if len(entry["qa"]["pos_windows"]) > 0:
            context = random.choice(entry["qa"]["pos_windows"])[0]
        else:
            context = entry["qa"]["neg_windows"][0][0]
    elif conf.retrieve_mode == "gold":
        for each_con in entry["qa"]["gold_inds"]:
            context += entry["qa"]["gold_inds"][each_con]
            context += " "

    elif conf.retrieve_mode == "none":
        # no retriever, use longformer
        table = entry["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        context = " ".join(entry["pre_text"]) + " " + " ".join(entry["post_text"]) + " " + table_text

    context = context.strip()
    # process "." and "*" in text
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")
        
    original_question = question + " " + tokenizer.sep_token + " " + context.strip()

    if "exe_ans" in entry["qa"]:
        options = entry["qa"]["exe_ans"]
    else:
        options = None

    original_question_tokens = original_question.split(' ')

    numbers = []
    number_indices = []
    question_tokens = []
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)

    if "exe_ans" in entry["qa"]:
        answer = entry["qa"]["exe_ans"]
    else:
        answer = None

    # table headers
    for row in entry["table"]:
        tok = row[0]
        if tok and tok in original_question:
            numbers.append(tok)
            tok_index = original_question.index(tok)
            prev_tokens = original_question[:tok_index]
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)

    if conf.program_mode == "seq":
        if 'program' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)
        else:
            program = None
            original_program = None
            
    elif conf.program_mode == "nest":
        if 'program_re' in entry["qa"]:
            original_program = entry["qa"]['program_re']
            program = program_tokenization(original_program)
        else:
            program = None
            original_program = None
        
    else:
        program = None
        original_program = None

    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)
