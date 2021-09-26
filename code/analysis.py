import csv
import re
import sys
import os
import json
import random
import copy
import math
from sympy import simplify
csv.field_size_limit(sys.maxsize)


all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]

const_list = [
    "const_1",
    "const_2",
    "const_3",
    "const_4",
    "const_5",
    "const_6",
    "const_7",
    "const_8",
    "const_9",
    "const_10",
    "const_100",
    "const_1000",
    "const_10000",
    "const_100000",
    "const_1000000",
    "const_10000000",
    "const_1000000000",
    "const_m1",
    "none",
    "#0",
    "#1",
    "#2",
    "#3",
    "#4",
    "#5"
]

const_units = [
    "const_10",
    "const_100",
    "const_1000",
    "const_10000",
    "const_100000",
    "const_1000000",
    "const_10000000",
    "const_1000000000",
]


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
        elif "const_" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


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


def eval_program(program, table):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"
    
    try:
        program = program[:-1] # remove EOF
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
        
        # print(program)
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":
                
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    # print(arg1)
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

                    
                # print("ind: ", ind)
                # print(this_res)
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
                    
                # this_res = round(this_res, 5)

                res_dict[ind] = this_res

            # print(this_res)

        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            # print(this_res)
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
    
    program1 = program1[:-1] # remove EOF
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
        program2 = program2[:-1] # remove EOF
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
            
            # print(args)
            # print(op)
            
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
        
        # print(op)
        # print(arg1)
        # print(arg2)
        
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
    # print(program1)
    steps = program1.split(")")[:-1]
    # print(steps)
    # print(steps)
    # print(sym_map)
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)
    # print("########")
    # print(sym_prog1)
    
    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
        # print(sym_prog2)
    except:
        return False

    return sym_prog1 == sym_prog2


def this_evaluate_result(data):
    '''
    execution acc
    program acc
    '''
        
    exe_correct = 0
    prog_correct = 0

    res_list = []
    all_res_list = []
    
    for each_data in data:

        each_id = each_data["id"]
        
        table = each_data["table"]
        gold_res = each_data["qa"]["exe_ans"]
        
        pred = each_data["qa"]["predicted"]
        gold = program_tokenization(each_data["qa"]["program"])

        # print("#########")
        # print(pred)
        # print(gold)
        
        # print("\n")
        # print("########")
        invalid_flag, exe_res = eval_program(pred, table)
        
        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1
                
        # else:
        #     if "".join(gold) == "".join(pred):
        #         print(each_id)
        #         print(gold)
        #         print(pred)
        #         print(gold_res)
        #         print(exe_res)
        #         print(each_ori_data["id"])
                
        
        if equal_program(gold, pred):
            # assert exe_res == gold_res
            # if exe_res != gold_res:
            #     print(each_id)
            #     print(gold)
            #     print(pred)
            #     print(gold_res)
            #     print(exe_res)
            #     print(each_ori_data["id"])
            assert exe_res == gold_res
            prog_correct += 1
            # if "".join(gold) != "".join(pred):
            #     print(each_id)
            #     print(gold)
            #     print(pred)
            #     print(gold_res)
            #     print(exe_res)
            #     print(each_ori_data["id"])

        # if "".join(gold) == "".join(pred):
        #     if not equal_program(gold, pred):
        #         print(each_id)
        #         print(gold)
        #         print(pred)
        #         print(gold_res)
        #         print(exe_res)
        #         print(each_ori_data["id"])
        #     prog_correct += 1
            
    exe_acc = float(exe_correct) / len(data)
    prog_acc = float(prog_correct) / len(data)
            
    print("All: ", len(data))
    print("Exe acc: ", exe_acc)
    print("Prog acc: ", prog_acc)

    return exe_acc, prog_acc
    

def eval_cat_steps(json_in):

    with open(json_in) as f_in:
        data = json.load(f_in)

    data1 = []
    data2 = []
    data3 = []

    for each_data in data:
        steps = each_data["qa"]["steps"]

        if len(steps) == 1:
            data1.append(each_data)

        elif len(steps) == 2:
            data2.append(each_data)

        else:
            # more than 2 steps
            data3.append(each_data)

    print("1 step")
    this_evaluate_result(data1)

    print("2 steps")
    this_evaluate_result(data2)

    print(">2 steps")
    this_evaluate_result(data3)


def eval_cat_types(json_in):

    with open(json_in) as f_in:
        data = json.load(f_in)

    data_text = []
    data_table = []
    data_both = []

    for each_data in data:
        gold_inds = each_data["qa"]["gold_inds"]

        has_table = 0
        has_text = 0
        for tmp in gold_inds:
            if "text" in tmp:
                has_text = 1
            if "table" in tmp:
                has_table = 1

        if has_text == 1 and has_table == 0:
            # text only
            data_text.append(each_data)

        if has_text == 0 and has_table == 1:
            # table only
            data_table.append(each_data)

        if has_text == 1 and has_table == 1:
            # table text
            data_both.append(each_data)

    print("text only")
    this_evaluate_result(data_text)

    print("table only")
    this_evaluate_result(data_table)

    print("both")
    this_evaluate_result(data_both)


def eval_cat_const(json_in):

    with open(json_in) as f_in:
        data = json.load(f_in)

    data1 = []

    for each_data in data:
        steps = each_data["qa"]["steps"]

        program = each_data["qa"]["program"]

        has_const = 0
        for each_const in const_units:
            if each_const in program:
                has_const = 1

        if has_const == 1:
            data1.append(each_data)

    print("const")
    this_evaluate_result(data1)


def cat_distance(json_in):
    '''
    distribution of evi
    '''

    res = {}

    with open(json_in) as f_in:
        data = json.load(f_in)

    res = {1: 0, 2: 0, 3: 0}
    num_all = 0
    for each_data in data:
        num_all += 1
        gold_inds = each_data["qa"]["gold_inds"]
        if len(gold_inds) == 1:
            res[1] += 1
        elif len(gold_inds) == 2:
            res[2] += 1
        else:
            res[3] += 1

    for tmp in res:
        res[tmp] /= num_all
    print(res)


def eval_distance(json_in):
    '''
    distribution of evi
    '''

    res = {}

    with open(json_in) as f_in:
        data = json.load(f_in)

    res = {">6": 0, "1-3": 0, "4-6": 0}
    num_all = 0

    for each_data in data:
        gold_inds = each_data["qa"]["gold_inds"]
        pre_text_len = len(each_data["pre_text"])
        table_len = len(each_data["table"])
        if len(gold_inds) > 1:
            num_all += 1
            this_inds = []
            for tmp in gold_inds:
                if "text" in tmp:
                    num = int(tmp.split("_")[1])
                    if num >= pre_text_len:
                        num += table_len
                elif "table" in tmp:
                    num = int(tmp.split("_")[1])
                    num += pre_text_len
                this_inds.append(num)

            this_inds = sorted(this_inds, reverse=True)
            max_dist = this_inds[0] - this_inds[-1]
            # print(this_inds)

            if max_dist <= 3:
                res["1-3"] += 1
            elif max_dist <= 6:
                res["4-6"] += 1
            else:
                res[">6"] += 1

    for tmp in res:
        res[tmp] = res[tmp] / num_all
    print(res)


def sum_report_page(json_in):
    '''
    all report pages
    '''

    res = {}
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        this_id = each_data["filename"]

        if this_id not in res:
            res[this_id] = 0


    print(len(res))


def sum_vocab(json_in):
    '''
    all vocab, exclude numbers
    '''

    res = {}
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        table = each_data["table"]
        question = each_data["qa"]["question"]

        table_flat = []
        for row in table:
            table_flat.append(" ".join(row))
        table_flat.append(question)

        for sent in pre_text + post_text + table_flat:
            for tok in sent.split(" "):
                num = str_to_num(tok)

                if num == "n/a":
                    if tok not in res:
                        res[tok] = 0


    print(len(res))


def sum_tok(json_in):
    '''
    num of tokens in text
    '''

    res = {}
    with open(json_in) as f_in:
        data = json.load(f_in)

    num_text_tok = 0
    num_table_rows = 0
    num_text_sents = 0
    num_table_tok = 0
    num_all_tok = 0
    num_question_tok = 0

    max_all_tok = 0

    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        table = each_data["table"]
        question = each_data["qa"]["question"]

        num_table_rows += len(table)
        num_text_sents += (len(pre_text) + len(post_text))

        this_all_tok = 0

        for sent in pre_text + post_text:
            num_text_tok += len(sent.split(" "))
            this_all_tok += len(sent.split(" "))

        for row in table:
            flat_row = " ".join(row)
            num_table_tok += len(flat_row.split(" "))
            this_all_tok += len(flat_row.split(" "))

        if this_all_tok > max_all_tok:
            max_all_tok = this_all_tok

        num_question_tok += len(question.split(" "))


    avg_tok = float(num_text_tok) / len(data)
    avg_table_row = float(num_table_rows) / len(data)
    avg_text_sents = float(num_text_sents) / len(data)
    avg_table_tok = float(num_table_tok) / len(data)

    num_all_tok = num_text_tok + num_table_tok

    avg_all_tok = float(num_all_tok) / len(data)

    avg_question = float(num_question_tok) / len(data)



    print("avg text tok")
    print(avg_tok)
    print("avg table row")
    print(avg_table_row)
    print("avg text sents")
    print(avg_text_sents)
    print("avg table tok")
    print(avg_table_tok)
    print("avg all tok")
    print(avg_all_tok)
    print("avg question tok")
    print(avg_question)
    print("max all tok")
    print(max_all_tok)


def data_stats(json_in):
    '''
    data statistics
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    steps_dict = {1: 0, 2: 0, 3: 0}
    table_text_split = {"table": 0, "text": 0, "both": 0}
    this_op_map = {
        "add": 0,
        "subtract": 0,
        "multiply": 0,
        "divide": 0,
        "exp": 0,
        "greater": 0,
        "table_max": 0,
        "table_min": 0,
        "table_sum": 0,
        "table_average": 0,
        "table": 0
    }

    for each_data in data:
        this_num = len(each_data["qa"]["steps"])
        if this_num == 1:
            steps_dict[1] += 1
        elif this_num == 2:
            steps_dict[2] += 1
        else:
            steps_dict[3] += 1
        # if this_num not in steps_dict:
        #     steps_dict[this_num] = 0
        # steps_dict[this_num] += 1

        has_table = 0
        has_text = 0
        for tmp in each_data["qa"]["gold_inds"]:
            if "table" in tmp:
                has_table = 1
            if "text" in tmp:
                has_text = 1

        if has_table == 1 and has_text == 0:
            table_text_split["table"] += 1
        elif has_table == 1 and has_text == 1:
            table_text_split["both"] += 1
        elif has_table == 0 and has_text == 1:
            table_text_split["text"] += 1
        else:
            print("error")


        program = each_data["qa"]["program"]
        for step in program.split("),"): 
            for op in this_op_map:
                if op in step:
                    this_op_map[op] += 1


    sum_step = 0
    for tmp in steps_dict:
        sum_step += steps_dict[tmp]

    for tmp in steps_dict:
        steps_dict[tmp] /= sum_step


    print("steps")
    print(len(data))
    print(steps_dict)

    sum_step = 0
    for tmp in table_text_split:
        sum_step += table_text_split[tmp]

    for tmp in table_text_split:
        table_text_split[tmp] /= sum_step

    print("table text")
    print(table_text_split)

    sum_step = 0
    for tmp in this_op_map:
        sum_step += this_op_map[tmp]

    for tmp in this_op_map:
        this_op_map[tmp] /= sum_step

    this_op_map["table"] = this_op_map["table_max"] + this_op_map["table_min"] + this_op_map["table_sum"] + this_op_map["table_average"]

    print("ops")
    print(this_op_map)


if __name__ == '__main__':

    root_path = "/mnt/george_bhd/zhiyuchen/"
    outputs = root_path + "outputs/"
    dataset = root_path + "finQA/dataset/"
    all_data = dataset + "our_data_final.json"

    json_in = outputs + \
        "inference_only_20210505181301_roberta-large-7k/results/test/full_results.json"

    # eval_cat_steps(json_in)

    # eval_cat_const(json_in)

    # eval_cat_types(json_in)

    # cat_distance(all_data)

    # eval_distance(all_data)


    # dataset statistics
    # sum_report_page(all_data)

    # sum_vocab(all_data)

    sum_tok(all_data)


    # data_stats(all_data)
