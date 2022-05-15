import time
import os
import sys
import shutil
import io
import subprocess
import re
import zipfile
import json
import copy
import torch
import random
import collections
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from config import parameters as conf
from transformers import BertTokenizer, BertModel, BertConfig
import finqa_utils as finqa_utils
from sympy import simplify

# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
print(os.popen('stty size', 'r').read())
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')


def get_current_git_version():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def read_txt(input_path, log_file):
    """Read a txt file into a list."""

    write_log(log_file, "Reading: %s" % input_path)
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


def read_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in tqdm(input_data):
        examples.append(finqa_utils.read_mathqa_entry(entry, tokenizer))
        program = examples[-1].program
        # for tok in program:
        #     if 'const_' in tok and not (tok in const_list):
        #         const_list.append(tok)
        #     elif '(' in tok and not (tok in op_list):
        #         op_list.append(tok)
    return input_data, examples, op_list, const_list


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 max_program_length,
                                 is_training,
                                 op_list,
                                 op_list_size,
                                 const_list,
                                 const_list_size,
                                 verbose=True):
    """Converts a list of DropExamples into InputFeatures."""
    unique_id = 1000000000
    res = []
    for (example_index, example) in enumerate(examples):
        features = example.convert_single_example(
            is_training=is_training,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_program_length=max_program_length,
            op_list=op_list,
            op_list_size=op_list_size,
            const_list=const_list,
            const_list_size=const_list_size,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        for feature in features:
            feature.unique_id = unique_id
            feature.example_index = example_index
            res.append(feature)
            unique_id += 1

    return res


RawResult = collections.namedtuple(
    "RawResult",
    "unique_id logits loss")


def compute_prog_from_logits(logits, max_program_length, example,
                             template=None):
    pred_prog_ids = []
    op_stack = []
    loss = 0
    for cur_step in range(max_program_length):
        cur_logits = logits[cur_step]
        cur_pred_softmax = _compute_softmax(cur_logits)
        cur_pred_token = np.argmax(cur_logits)
        loss -= np.log(cur_pred_softmax[cur_pred_token])
        pred_prog_ids.append(cur_pred_token)
        if cur_pred_token == 0:
            break
    return pred_prog_ids, loss


def compute_predictions(all_examples, all_features, all_results, n_best_size,
                        max_program_length, tokenizer, op_list, op_list_size,
                        const_list, const_list_size):
    """Computes final predictions based on logits."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "logits"
        ])

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            logits = result.logits
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    logits=logits))

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", "options answer program_ids program")

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            program = example.program
            pred_prog_ids, loss = compute_prog_from_logits(pred.logits,
                                                           max_program_length,
                                                           example)
            pred_prog = finqa_utils.indices_to_prog(pred_prog_ids,
                                                    example.numbers,
                                                    example.number_indices,
                                                    conf.max_seq_length,
                                                    op_list, op_list_size,
                                                    const_list, const_list_size
                                                    )
            nbest.append(
                _NbestPrediction(
                    options=example.options,
                    answer=example.answer,
                    program_ids=pred_prog_ids,
                    program=pred_prog))

        assert len(nbest) >= 1

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["id"] = example.id
            output["options"] = entry.options
            output["ref_answer"] = entry.answer
            output["pred_prog"] = [str(prog) for prog in entry.program]
            output["ref_prog"] = example.program
            output["question_tokens"] = example.question_tokens
            output["numbers"] = example.numbers
            output["number_indices"] = example.number_indices
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions["pred_programs"][example_index] = nbest_json[0]["pred_prog"]
        all_predictions["ref_programs"][example_index] = nbest_json[0]["ref_prog"]
        all_nbest[example_index] = nbest_json

    return all_predictions, all_nbest


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


class DataLoader:
    def __init__(self, is_training, data, reserved_token_size, batch_size=64, shuffle=True):
        """
        Main dataloader
        """
        self.data = data
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(data)
        self.reserved_token_size = reserved_token_size
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        if shuffle:
            self.shuffle_all_data()
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # drop last batch
        if self.is_training:
            bound = self.num_batches - 1
        else:
            bound = self.num_batches
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        random.shuffle(self.data)
        return

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        batch_data = {"unique_id": [],
                      "example_index": [],
                      "tokens": [],
                      "question": [],
                      "input_ids": [],
                      "input_mask": [],
                      "option_mask": [],
                      "segment_ids": [],
                      "options": [],
                      "answer": [],
                      "program": [],
                      "program_ids": [],
                      "program_weight": [],
                      "program_mask": []}
        for each_data in self.data[start_index: end_index]:

            batch_data["option_mask"].append(each_data.option_mask)
            batch_data["input_mask"].append(each_data.input_mask)

            batch_data["unique_id"].append(each_data.unique_id)
            batch_data["example_index"].append(each_data.example_index)
            batch_data["tokens"].append(each_data.tokens)
            batch_data["question"].append(each_data.question)
            batch_data["input_ids"].append(each_data.input_ids)
            batch_data["segment_ids"].append(each_data.segment_ids)
            batch_data["options"].append(each_data.options)
            batch_data["answer"].append(each_data.answer)
            batch_data["program"].append(each_data.program)
            batch_data["program_ids"].append(each_data.program_ids)
            batch_data["program_weight"].append(each_data.program_weight)
            batch_data["program_mask"].append(each_data.program_mask)

        return batch_data


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def str_to_num(text):

    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def process_row(row_in):

    row_out = []
    invalid_flag = 0

    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()

        num = str_to_num(num)

        if num == "n/a":
            invalid_flag = 1
            break

        row_out.append(num)

    if invalid_flag:
        return "n/a"

    return row_out


def reprog_to_seq(prog_in, is_gold):
    '''
    predicted recursive program to list program
    ["divide(", "72", "multiply(", "6", "210", ")", ")"]
    ["multiply(", "6", "210", ")", "divide(", "72", "#0", ")"]
    '''

    st = []
    res = []

    try:
        num = 0
        for tok in prog_in:
            if tok != ")":
                st.append(tok)
            else:
                this_step_vec = [")"]
                for _ in range(3):
                    this_step_vec.append(st[-1])
                    st = st[:-1]
                res.extend(this_step_vec[::-1])
                st.append("#" + str(num))
                num += 1
    except:
        if is_gold:
            raise ValueError

    return res


def eval_program(program, table):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program = "|".join(program)
        steps = program.split(")")[:-1]

        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":

                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break

                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"

                res_dict[ind] = this_res

            elif "table" in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]

                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break

                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)

                if num_row == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)

                res_dict[ind] = this_res
        if this_res != "yes" and this_res != "no" and this_res != "n/a":

            this_res = round(this_res, 5)

    except:
        invalid_flag = 1

    return invalid_flag, this_res


def equal_program(program1, program2):
    '''
    symbolic program if equal
    program1: gold
    program2: pred
    '''

    sym_map = {}

    program1 = program1[:-1]  # remove EOF
    program1 = "|".join(program1)
    steps = program1.split(")")[:-1]

    invalid_flag = 0
    sym_ind = 0
    step_dict_1 = {}

    # symbolic map
    for ind, step in enumerate(steps):

        step = step.strip()

        assert len(step.split("(")) <= 2

        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        step_dict_1[ind] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1

        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1

            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1

    # check program 2
    step_dict_2 = {}
    try:
        program2 = program2[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program2):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    print("structure error")
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    print("structure error")
                    return False

        program2 = "|".join(program2)
        steps = program2.split(")")[:-1]

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False

            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False

                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except:
        return False

    def symbol_recur(step, step_dict):

        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()

        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            # as var
            return sym_map[step]

        if "#" in arg1:
            arg1_ind = int(arg1.replace("#", ""))
            arg1_part = symbol_recur(step_dict[arg1_ind], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if "#" in arg2:
            arg2_ind = int(arg2.replace("#", ""))
            arg2_part = symbol_recur(step_dict[arg2_ind], step_dict)
        else:
            arg2_part = sym_map[arg2]

        if op == "add":
            return "( " + arg1_part + " + " + arg2_part + " )"
        elif op == "subtract":
            return "( " + arg1_part + " - " + arg2_part + " )"
        elif op == "multiply":
            return "( " + arg1_part + " * " + arg2_part + " )"
        elif op == "divide":
            return "( " + arg1_part + " / " + arg2_part + " )"
        elif op == "exp":
            return "( " + arg1_part + " ** " + arg2_part + " )"
        elif op == "greater":
            return "( " + arg1_part + " > " + arg2_part + " )"

    # # derive symbolic program 1
    steps = program1.split(")")[:-1]
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)

    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except:
        return False

    return sym_prog1 == sym_prog2


def evaluate_result(json_in, json_ori, all_res_file, error_file, program_mode):
    '''
    execution acc
    program acc
    '''
    correct = 0

    with open(json_in) as f_in:
        data = json.load(f_in)

    with open(json_ori) as f_in:
        data_ori = json.load(f_in)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["id"] not in data_dict
        data_dict[each_data["id"]] = each_data

    exe_correct = 0
    prog_correct = 0

    res_list = []
    all_res_list = []

    for tmp in data:
        each_data = data[tmp][0]
        each_id = each_data["id"]

        each_ori_data = data_dict[each_id]

        table = each_ori_data["table"]
        gold_res = each_ori_data["qa"]["exe_ans"]

        pred = each_data["pred_prog"]
        gold = each_data["ref_prog"]

        if program_mode == "nest":
            if pred[-1] == "EOF":
                pred = pred[:-1]
            pred = reprog_to_seq(pred, is_gold=False)
            pred += ["EOF"]
            gold = gold[:-1]
            gold = reprog_to_seq(gold, is_gold=True)
            gold += ["EOF"]

        invalid_flag, exe_res = eval_program(pred, table)

        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1

            if equal_program(gold, pred):
                if exe_res != gold_res:
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(each_ori_data["id"])
                assert exe_res == gold_res
                prog_correct += 1
                if "".join(gold) != "".join(pred):
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(each_ori_data["id"])

        each_ori_data["qa"]["predicted"] = pred

        if exe_res != gold_res:
            res_list.append(each_ori_data)
        all_res_list.append(each_ori_data)

    exe_acc = float(exe_correct) / len(data)
    prog_acc = float(prog_correct) / len(data)

    print("All: ", len(data))
    print("Correct: ", correct)
    print("Exe acc: ", exe_acc)
    print("Prog acc: ", prog_acc)

    with open(error_file, "w") as f:
        json.dump(res_list, f, indent=4)

    with open(all_res_file, "w") as f:
        json.dump(all_res_list, f, indent=4)

    return exe_acc, prog_acc


if __name__ == '__main__':

    root = "your_root_path"
    our_data = root + "dataset/"
