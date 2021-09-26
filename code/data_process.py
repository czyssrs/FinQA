import csv
import re
import sys
import os
import json
import random
import copy
import math
csv.field_size_limit(sys.maxsize)

op_map = {
    "add": "add",
    "minus": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "exp": "exp",
    "compare_larger": "compare_larger",
    "max": "table_max",
    "min": "table_min",
    "sum": "table_sum",
    "average": "table_average"
}

const_map = {
    "1": "const_1",
    "2": "const_2",
    "3": "const_3",
    "4": "const_4",
    "5": "const_5",
    "6": "const_6",
    "7": "const_7",
    "8": "const_8",
    "9": "const_9",
    "10": "const_10",
    "100": "const_100",
    "1000": "const_1000",
    "10000": "const_10000",
    "100000": "const_100000",
    "1000000": "const_1000000",
    "10000000": "const_10000000",
    "1000000000": "const_1000000000",
    "-1": "const_m1",
    "": "none"
}

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

def cleanhtml(raw_html):
    cleanr = re.compile('<[A-Za-z]*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def string_process(text_in):
    '''
    remove white spaces in text
    remove unicode
    to lower case
    '''

    res = []
    if text_in == "":
        return ""
    text_in = (text_in.encode('unicode-escape')).decode("utf-8", "strict")
    text_in = text_in.replace("\\u", " ")
    text_in = text_in.replace("\\x", " ")
    text_in = text_in.replace("\\t", " ")
    text_in = cleanhtml(text_in)

    text_in = text_in.replace("$", "$ ")
    # text_in = text_in.replace("%", " %")
    
    # text_in = text_in.replace(". ", " . ")
    # text_in = text_in.replace(", ", " , ")
    # text_in = text_in.replace("; ", " ; ")
    # text_in = text_in.replace("? ", " ? ")
    # text_in = text_in.replace(": ", " : ")

    for token in text_in.split(" "):
        if token != "":
            token = token.lower()
            if token != "":
                if "," in token[1:-1]:
                    out = token[0] + token[1:-1].replace(",", "") + token[-1]
                    res.append(out)
                else:
                    res.append(token)
                    
    res = " ".join(res)
    res = res.replace(". ", " . ")
    res = res.replace(", ", " , ")
    res = res.replace("; ", " ; ")
    res = res.replace("? ", " ? ")
    res = res.replace(": ", " : ")
    res = res.replace("(", " ( ")
    res = res.replace(")", " ) ")
    
    if res != "" and res[-1] == ".":
        res = res[:-1] + " ."
        
    res_final = []
    for tmp in res.split(" "):
        if tmp != "":
            res_final.append(tmp)

    return " ".join(res_final)

def string_process_table(text_in):
    '''
    remove white spaces in text
    remove unicode
    to lower case
    '''

    res = []
    if text_in == "":
        return ""
    text_in = (text_in.encode('unicode-escape')).decode("utf-8", "strict")
    text_in = text_in.replace("\\u", " ")
    text_in = text_in.replace("\\x", " ")
    text_in = text_in.replace("\\t", " ")
    text_in = cleanhtml(text_in)

    text_in = text_in.replace("$", "$ ")
    # text_in = text_in.replace("%", " %")
    
    # text_in = text_in.replace(". ", " . ")
    # text_in = text_in.replace(", ", " , ")
    # text_in = text_in.replace("; ", " ; ")
    # text_in = text_in.replace("? ", " ? ")
    # text_in = text_in.replace(": ", " : ")

    for token in text_in.split(" "):
        if token != "":
            token = token.lower()
            if token != "":
                if "," in token[1:-1]:
                    out = token[0] + token[1:-1].replace(",", "") + token[-1]
                    res.append(out)
                else:
                    res.append(token)
                    
    res = " ".join(res)
    res = res.replace(". ", " . ")
    res = res.replace(", ", " , ")
    res = res.replace("; ", " ; ")
    res = res.replace("? ", " ? ")
    res = res.replace(": ", " : ")
    res = res.replace("(", " ( ")
    res = res.replace(")", " ) ")
    
    if res != "" and res[-1] == ".":
        res = res[:-1] + " ."
        
    res_final = []
    for tmp in res.split(" "):
        if tmp != "":
            res_final.append(tmp)

    return " ".join(res_final)


def string_process_percent(text_in):
    '''
    two version of percents
    '''
    
    res = []
    for token in text_in.split(" "):
        if token:
            if token[-1] == "%":
                res.append(token)
                res.append("(")
                res.append(token[:-1])
                res.append("%")
                res.append(")")
                
            else:
                res.append(token)
            
    return " ".join(res)




def check_table_row(row):
    '''
    check if this table header has repeats
    '''
    tmp = set()
    for ele in row[1:]:
        if ele not in tmp:
            tmp.add(ele)
        else:
            return False

    return True


def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            num = None
    return num

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)


    return " ".join(res)

def process_agg(folder, json_out):

    all_data = []
    res = []
    res_invalid = []

    for r, d, fs in os.walk(folder):
        for f in fs:
            with open(r+f) as f_in:
                data = json.load(f_in)
                all_data.extend(data)

    all_qs = 0
    steps_0 = 0
    final_steps_0 = 0
    
    for each_data in all_data:

        this_out = {}

        pre_text = []
        post_text = []
        all_input = []
        
        if "q1" in each_data:
            all_qs += 1
        if "q2" in each_data:
            all_qs += 1

        for each_sent in each_data["pre_text_ori"]:
            each_sent = string_process(each_sent)
            each_sent = string_process_percent(each_sent)
            pre_text.append(each_sent)
            all_input.extend(pre_text[-1].split(" "))

        for each_sent in each_data["post_text_ori"]:
            each_sent = string_process(each_sent)
            each_sent = string_process_percent(each_sent)
            post_text.append(each_sent)
            all_input.extend(post_text[-1].split(" "))

        old_pre_text = each_data["pre_text"]

        this_out["pre_text"] = pre_text
        this_out["post_text"] = post_text

        num_all_text_lines = len(
            each_data["pre_text"]) + len(each_data["post_text"])

        this_out["filename"] = each_data["filename"]

        this_table = []

        table = each_data["table"]
        table_header = []
        table_flag = 0

        this_out["table_ori"] = each_data["table_ori"]
        
        # this_out["old_pre_text"] = old_pre_text

        if check_table_row(table[0]) == False:
            
            # need first 2 rows
            new_first_row = []
            for ele1, ele2 in zip(table[0], table[1]):
                cell = string_process_table(ele1 + " " + ele2)
                # cell = ele1 + " " + ele2
                cell = string_process_percent(cell)
                new_first_row.append(cell)

            table = table[2:]
            this_table.append(new_first_row)
            table_flag = 1

            for ele in new_first_row:
                all_input.append(ele)
                all_input.extend(ele.split(" "))

        for row in table:
            this_row = []
            # table_header.append(row[0])
            for ele in row:
                ele = string_process_table(ele)
                ele = string_process_percent(ele)

                # if ele:
                #     if ele[0] == "(" and ele[-1] == ")":
                #         # negative number
                #         ele = "-" + ele[1:-1]
                #         ele += (" ( " + ele[1:-1] + " )")
                        
                #     # negative number with $ (40.9)
                #     if ele[0] == "$" and ele[2] == "(" and ele[-1] == ")" and ele.replace(".", "")[3:-1].isdigit():
                #         ele = ele[:2] + "-" + ele[3:-1]

                this_row.append(ele)
                all_input.append(ele)
                all_input.extend(ele.split(" "))

            table_header.append(this_row[0])
            this_table.append(this_row)

        this_out["table"] = this_table
        this_out["annotator"] = each_data["annotator"]

        # "question": ...
        # "answer": ...
        # "explanation": ...
        # "steps":
        # [
        # 	{"op": ..., "arg1": ..., "arg2": ..., "res": ...}
        # 	{"op": ..., "arg1": ..., "arg2": ..., "res": ...}
        # 	...
        # ]
        # "table_rows": []
        # "text_rows": []
        
        this_out_tem = copy.deepcopy(this_out)

        for qind in ["q1", "q2"]:
            if qind in each_data:

                invalid_flag = 0
                this_all_support = []
                this_q1 = each_data[qind]
                this_out[qind] = {}
                
                this_question = string_process(this_q1["question"])
                this_question = string_process_percent(this_question)
                this_out[qind]["question"] = this_question
                this_out[qind]["answer"] = string_process(this_q1["answer"])
                this_out[qind]["explanation"] = string_process(
                    this_q1["explanation"])
                
                this_all_support.extend(this_question.split(" "))
                all_input.extend(this_question.split(" "))

                ann_table_rows = []
                if "table_rows" in this_q1:
                    for tmp in this_q1["table_rows"]:
                        if tmp.isdigit():
                            if table_flag == 1:
                                if int(tmp) <= len(this_out["table"]) + 1:
                                    this_table_ind = int(tmp) - 2
                                    if this_table_ind > 0:
                                        ann_table_rows.append(this_table_ind)

                            else:
                                if int(tmp) <= len(this_out["table"]):
                                    this_table_ind = int(tmp) - 1
                                    if this_table_ind > 0:
                                        ann_table_rows.append(this_table_ind)

                    for ind in ann_table_rows:
                        for tmp in this_out["table"][ind]:
                            this_all_support.append(tmp)
                            this_all_support.extend(tmp.split(" "))

                # print(this_q1)
                # print(this_q1["table_rows"])

                ann_text_rows = []
                if "text_rows" in this_q1:
                    for tmp in this_q1["text_rows"]:
                        if tmp.isdigit() and int(tmp) <= num_all_text_lines:
                            ann_text_rows.append(
                                int(tmp) - 1 + len(pre_text) - len(old_pre_text))

                    for ind in ann_text_rows:
                        all_text = this_out["pre_text"] + this_out["post_text"]
                        this_line = all_text[ind]
                        this_all_support.extend(this_line.split(" "))

                this_out[qind]["ann_table_rows"] = ann_table_rows
                this_out[qind]["ann_text_rows"] = ann_text_rows
                
                # if "text_rows" in this_q1:
                #     this_out[qind]["ori_text_rows"] = this_q1["text_rows"]

                steps = this_q1["steps"]
                this_out[qind]["steps"] = []
                
                if len(steps) == 0:
                    invalid_flag = 1
                    steps_0 += 1

                for step in steps:
                    step["arg1"] = string_process(step["arg1"].replace("$", ""))
                    step["arg2"] = string_process(step["arg2"].replace("$", ""))
                    step["res"] = string_process(step["res"].replace("$", ""))

                    if "average" in step["op"]:
                        if step["arg1"] not in table_header:
                            if step["arg1"].replace(".", "").replace("%", "").replace("-", "").isdigit():
                                step["op"] = step["op"].replace("average", "add")
                                ori_res = step["res"]
                                try:
                                    step["res"] = str(str_to_num(step["res"]) * 2)
                                except:
                                    invalid_flag = 1
                                    print("avg invalid")
                                    break

                                new_step = {}
                                new_step["op"] = "divide0-0"
                                new_step["arg1"] = step["res"]
                                new_step["arg2"] = "const_2"
                                new_step["res"] = ori_res

                                this_out[qind]["steps"].append(step)
                                this_out[qind]["steps"].append(new_step)
                                
                        else:
                            if "(" in step["arg1"] or ")" in step["arg1"]:
                                invalid_flag = 1
                                break
                            this_out[qind]["steps"].append(step)

                    elif "sum" in step["op"]:
                        if step["arg1"] not in table_header:
                            if step["arg1"].replace(".", "").replace("%", "").replace("-", "").isdigit():
                                step["op"] = step["op"].replace("sum", "add")
                                this_out[qind]["steps"].append(step)
                            else:
                                print("sum error")
                                print(step["arg1"])
                                print(table_header)
                                
                        else:
                            if "(" in step["arg1"] or ")" in step["arg1"]:
                                invalid_flag = 1
                                break
                            this_out[qind]["steps"].append(step)

                    elif "max" == step["op"][:-3] or "min" == step["op"][:-3]:
                        if step["arg1"] not in table_header:
                            if step["arg1"].replace(".", "").replace("%", "").replace("-", "").isdigit():
                                invalid_flag = 1
                            else:
                                print("max error")
                                print(step["arg1"])
                                print(table_header)
                                
                        else:
                            if "(" in step["arg1"] or ")" in step["arg1"]:
                                invalid_flag = 1
                                break
                            this_out[qind]["steps"].append(step)

                    else:
                        this_out[qind]["steps"].append(step)

                # if len(this_out[qind]["steps"]) == 0:
                #     print("steps 0 error")
                # program
                # multiply(n0,n1)|divide(#0,const_100)|add(n0,#1)|
                # add(add(negate(26), const_1), add(add(negate(26), const_1), const_1))
                if len(this_all_support) == 0:
                	this_all_support = all_input
                # this_all_support = all_input
                res_dict = {}
                program = ""
                minus_ind = -1
                minus_arg = ""
                # print(all_input)
                # print("$$$$$$$$$$")
                for ind, step in enumerate(this_out[qind]["steps"]):
                    if_minus = 0
                    this_op = op_map[step["op"][:-3]]
                    arg1 = step["arg1"].strip()
                    if arg1 in res_dict:
                        arg1 = "#" + str(res_dict[arg1])
                        step["arg1"] = arg1
                    elif arg1 not in this_all_support:
                        if arg1 in const_map:
                            arg1 = const_map[arg1]
                            step["arg1"] = arg1
                        elif arg1 + "0" in this_all_support + all_input:
                            arg1 = arg1 + "0"
                            step["arg1"] = arg1
                        elif arg1 + ".0" in this_all_support + all_input:
                            arg1 = arg1 + ".0"
                            step["arg1"] = arg1
                        elif "0" + arg1 in this_all_support + all_input:
                            arg1 = "0" + arg1
                            step["arg1"] = arg1
                        elif arg1[0] == "-" and arg1.strip("-") in this_all_support + all_input:
                            if_minus = 1
                            minus_ind = ind
                            minus_arg = arg1
                        else:
                            if arg1 not in all_input:
                                invalid_flag = 1
                                # print("#############")
                                # print(arg1)
                                # print(this_all_support)
                                # print(all_input)
                                # print(this_out["annotator"])
                                # print(this_out["filename"])
                                
                                # for line in this_out["pre_text"] + this_out["post_text"]:
                                #     print(line)
                                # for row in this_out["table"]:
                                #     print(row)
                                # print("#########original table")
                                # for row in this_out["table_ori"]:
                                #     print(row)
                                # if "table_rows" in this_q1:
                                #     print(this_q1["table_rows"])
                                # else:
                                #     print("not annotated")
                                # if "text_rows" in this_q1:
                                #     print(this_q1["text_rows"])
                                # else:
                                #     print("not annotated")
                                    
                                # print(this_out[qind]["steps"])
                                # print(this_q1["steps"])
                                break
                    # if "table" not in this_op:
                    arg2 = step["arg2"].strip()
                    if arg2 in res_dict:
                        arg2 = "#" + str(res_dict[arg2])
                        step["arg2"] = arg2
                    elif arg2 not in this_all_support:
                        if arg2 in const_map:
                            arg2 = const_map[arg2]
                            step["arg2"] = arg2
                        elif arg2 + "0" in this_all_support + all_input:
                            arg2 = arg2 + "0"
                            step["arg2"] = arg2
                        elif arg2 + ".0" in this_all_support + all_input:
                            arg2 = arg2 + ".0"
                            step["arg2"] = arg2
                        elif "0" + arg2 in this_all_support + all_input:
                            arg2 = "0" + arg2
                            step["arg2"] = arg2
                        elif arg2 == "":
                            arg2 = "none"
                            step["arg2"] = arg2
                        # for average
                        elif arg2 in const_list:
                            step["arg2"] = arg2
                        else:
                            if arg2 not in all_input:
                                invalid_flag = 1
                                break
                            
                    if arg2 == "":
                        arg2 = "none"

                    res_dict[step["res"]] = ind
                    if if_minus:
                        print("minus step")
                        if arg1.strip("-") in res_dict and ind != 0:
                            m_arg = "#" + str(res_dict[arg1.strip("-")])
                        else:
                            m_arg = arg1.strip("-")
                            
                        add_step = "multiply" + "(" + m_arg + ", " + "const_m1), "
                        this_step_str = this_op + "(#" + str(ind) + ", " + arg2 + "), "
                    else:
                        add_step = ""
                        this_step_str = this_op + "(" + arg1 + ", " + arg2 + "), "

                    program += add_step
                    program += this_step_str

                program = program.strip(" ").strip(",")
                
                if minus_ind != -1:
                    add_step = {
                        "op": "multiply0-0",
                        "arg1": minus_arg.strip("-"),
                        "arg2": "const_m1",
                        "res": minus_arg
                    }
                    
                    this_out[qind]["steps"].insert(minus_ind, add_step)

                if invalid_flag == 1:
                    del this_out[qind]
                    continue

                this_out[qind]["program"] = program

        if "q1" not in this_out and "q2" not in this_out:
            continue

        # make separated
        if "q1" in this_out:
            this_out_1 = copy.deepcopy(this_out_tem)
            this_out_1["qa"] = this_out["q1"]
            this_out_1["id"] = this_out["filename"] + "-1"
            if len(this_out_1["qa"]["steps"]) == 0:
                final_steps_0 += 1
            else:
                res.append(this_out_1)
            
        if "q2" in this_out:
            this_out_2 = copy.deepcopy(this_out_tem)
            this_out_2["qa"] = this_out["q2"]
            this_out_2["id"] = this_out["filename"] + "-2"
            if len(this_out_2["qa"]["steps"]) == 0:
                final_steps_0 += 1
            else:
                res.append(this_out_2)
            

    random.shuffle(res)
    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    print("Original: ", all_qs)
    print("Res: ", len(res))
    print("Steps 0: ", steps_0)
    print("Final steps 0: ", final_steps_0)

def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("The " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()

def process_map(json_in, json_out, json_out_invalid):
    '''
    get the gold retrieve results
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)
    res_out = []

    all_good_evi = 0
    all_qs = 0
    res_invalid = []
    steps_0 = 0

    for each_data in data:
        all_qs += 1
        this_q = each_data["qa"]
        this_question = this_q["question"]
        question_map = this_question.split(" ")
        
        # out: { ind: text, ... }
        gold_inds = {}
        
        
        all_input_map = {}
        i = 0
        for ind, each_sent in enumerate(each_data["pre_text"]):
            for token in each_sent.split(" "):
                all_input_map[token] = "text_" + str(ind)
            i += 1
                
        for ind, each_sent in enumerate(each_data["post_text"]):
            for token in each_sent.split(" "):
                all_input_map[token] = "text_" + str(ind + i)
                
        for ind, each_row in enumerate(each_data["table"]):
            for cell in each_row:
                all_input_map[cell] = "table_" + str(ind)
                for token in cell.split(" "):
                    all_input_map[token] = "table_" + str(ind)


        given_evi = []
        
        for ind in this_q["ann_table_rows"]:
            this_table_row = each_data["table"][int(ind)]
            this_row_text = table_row_to_text(each_data["table"][0], this_table_row)
            
            this_row_text = string_process_table(this_row_text)
            gold_inds["table_" + str(ind)] = this_row_text
            
            given_evi.extend(this_row_text.split(" "))
            
            for token in this_table_row:
                given_evi.append(token)
        
        
        all_text = each_data["pre_text"] + each_data["post_text"]
        for ind in this_q["ann_text_rows"]:
            
            this_line = all_text[int(ind)]
            
            gold_inds["text_" + str(ind)] = this_line
            
            given_evi.extend(this_line.split(" "))
            
            
        steps = this_q["steps"]
        res_dict = {}
        good_evi = 1
        
        if len(steps) == 0:
            steps_0 += 1
        
        
        for step in steps:
            arg1 = step["arg1"]
            arg2 = step["arg2"]
            res = step["res"]
            
            if arg1 not in res_dict:
                if arg1 not in given_evi:
                    if arg1 in all_input_map:
                        good_evi = 0
                        # print("#############")
                        # print(arg1)
                        # this_out = each_data
                        # print(this_out["annotator"])
                        # print(this_out["filename"])
                        
                        # for line in this_out["pre_text"] + this_out["post_text"]:
                        #     print(line)
                        # for row in this_out["table"]:
                        #     print(row)
                        # print("#########original table")
                        # for row in this_out["table_ori"]:
                        #     print(row)
                        # if "ann_table_rows" in this_q:
                        #     print(this_q["ann_table_rows"])
                        # else:
                        #     print("not annotated")
                        # if "ann_text_rows" in this_q:
                        #     print(this_q["ann_text_rows"])
                        # else:
                        #     print("not annotated")
                            
                        # print(this_q["steps"])
                        # assert arg1 in all_input_map
                        
                        if "table" in all_input_map[arg1]:
                            this_ind = int(all_input_map[arg1].replace("table_", ""))
                            this_table_row = each_data["table"][this_ind]
                            this_row_text = table_row_to_text(each_data["table"][0], this_table_row)
                            this_row_text = string_process_table(this_row_text)
                            gold_inds[all_input_map[arg1]] = this_row_text
                            
                        else:
                            this_ind = int(all_input_map[arg1].replace("text_", ""))
                            gold_inds[all_input_map[arg1]] = all_text[this_ind]
                          
                    else:
                        if arg1 not in const_list and arg1 not in question_map:
                            print("arg1 not found error")
                            print("#############")
                            print(arg1)
                            this_out = each_data
                            print(this_out["annotator"])
                            print(this_out["filename"])
                            
                            for line in this_out["pre_text"] + this_out["post_text"]:
                                print(line)
                            for row in this_out["table"]:
                                print(row)
                            print("#########original table")
                            for row in this_out["table_ori"]:
                                print(row)
                            if "ann_table_rows" in this_q:
                                print(this_q["ann_table_rows"])
                            else:
                                print("not annotated")
                            if "ann_text_rows" in this_q:
                                print(this_q["ann_text_rows"])
                            else:
                                print("not annotated")
                                
                            print(this_q["steps"])
        
        
            if arg2 not in res_dict:
                if arg2 not in given_evi:
                    if arg2 in all_input_map:
                        
                        good_evi = 0
                        # print("#############")
                        # print(arg2)
                        # this_out = each_data
                        # print(this_out["annotator"])
                        # print(this_out["filename"])
                        
                        # for line in this_out["pre_text"] + this_out["post_text"]:
                        #     print(line)
                        # for row in this_out["table"]:
                        #     print(row)
                        # print("#########original table")
                        # for row in this_out["table_ori"]:
                        #     print(row)
                        # if "ann_table_rows" in this_q:
                        #     print(this_q["ann_table_rows"])
                        # else:
                        #     print("not annotated")
                        # if "ann_text_rows" in this_q:
                        #     print(this_q["ann_text_rows"])
                        # else:
                        #     print("not annotated")
                            
                        # print(this_q["steps"])
                        
                        # assert arg2 in all_input_map
                        
                        if "table" in all_input_map[arg2]:
                            this_ind = int(all_input_map[arg2].replace("table_", ""))
                            this_table_row = each_data["table"][this_ind]
                            this_row_text = table_row_to_text(each_data["table"][0], this_table_row)
                            this_row_text = string_process(this_row_text)
                            gold_inds[all_input_map[arg2]] = this_row_text
                            
                        else:
                            this_ind = int(all_input_map[arg2].replace("text_", ""))
                            gold_inds[all_input_map[arg2]] = all_text[this_ind]
                            
                    else:
                        if arg2 not in const_list and arg2 not in question_map:
                            print("arg2 not found error")
                            print("#############")
                            print(arg2)
                            this_out = each_data
                            print(this_out["annotator"])
                            print(this_out["filename"])
                            
                            for line in this_out["pre_text"] + this_out["post_text"]:
                                print(line)
                            for row in this_out["table"]:
                                print(row)
                            print("#########original table")
                            for row in this_out["table_ori"]:
                                print(row)
                            if "ann_table_rows" in this_q:
                                print(this_q["ann_table_rows"])
                            else:
                                print("not annotated")
                            if "ann_text_rows" in this_q:
                                print(this_q["ann_text_rows"])
                            else:
                                print("not annotated")
                                
                            print(this_q["steps"])
                            
            res_dict[res] = 0
            
        this_q["gold_inds"] = gold_inds
        if good_evi:
            all_good_evi += 1
            
        gold_len = []
        for tmp in gold_inds:
            gold_len.extend(gold_inds[tmp].split(" "))
            
        if len(gold_inds) != 0 and len(gold_len) < 290:
            res_out.append(each_data)
        else:
            res_invalid.append(each_data)
            

    with open(json_out, "w") as f:
        json.dump(res_out, f, indent=4)
        
    with open(json_out_invalid, "w") as f:
        json.dump(res_invalid, f, indent=4)
        
    print("Original: ", all_qs)
    print("All: ", len(res_out))
    print("Invalid: ", len(res_invalid))
    print("Good evi: ", all_good_evi)
    print("Steps 0: ", steps_0)
    


def train_test_split(json_in, train, valid, test):
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
        
    random.shuffle(data)
    filename_list = []
    
    for tmp in data:
        if tmp["filename"] not in filename_list:
            filename_list.append(tmp["filename"])
            
            
    size_test = int(len(filename_list) * 0.14)
    size_valid = int(len(filename_list) * 0.11)
    
    size_train = len(filename_list) - size_test - size_valid
    
    random.shuffle(filename_list)
    
    train_files = filename_list[:size_train]
    valid_files = filename_list[size_train: size_train + size_valid]
    test_files = filename_list[size_train + size_valid: ]
    
    data_train = []
    data_valid = []
    data_test = []
    
    for tmp in data:
        if tmp["filename"] in train_files:
            data_train.append(tmp)
        elif tmp["filename"] in valid_files:
            data_valid.append(tmp)
        else:
            data_test.append(tmp)
            
    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    with open(train, "w") as f:
        json.dump(data_train, f, indent=4)
        
    with open(valid, "w") as f:
        json.dump(data_valid, f, indent=4)

    with open(test, "w") as f:
        json.dump(data_test, f, indent=4)



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
            


def eval_program(json_in, json_out):
    '''
    calculate the numerical results of the program
    '''
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
    res = []
    for each_data in data:
        program = each_data["qa"]["program"]
        steps = program.split("),")
        table = each_data["table"]
        
        res_dict = {}
        
        # print(program)
        
        invalid_flag = 0
        print(program)
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0]
            args = step.split("(")[1]
            
            arg1 = args.split(",")[0].strip()
            arg2 = args.split(",")[1].strip(")").strip()
            
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
                    
                # if this_res != "yes" and this_res != "no":
                #     this_res = round(this_res, 5)
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
                

                
            
                
        if invalid_flag:
            print(each_data["filename"])
            print(program)
            
            continue

        if this_res != "yes" and this_res != "no":
            this_res = round(this_res, 5)
        
        each_data["qa"]["exe_ans"] = this_res
        res.append(each_data)
                    
                
    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)
        
    print("Original: ", len(data))
    print("Res: ", len(res))
            





def equal_program(program1, program2):
    '''
    symbolic program if equal
    program1: gold
    program2: pred
    '''
    from sympy import simplify
    
    sym_map = {}
    
    steps = program1.split("),")
    
    invalid_flag = 0
    sym_ind = 0
    step_dict_1 = {}
    
    # symbolic map
    for ind, step in enumerate(steps):
        step = step.strip()
        
        assert len(step.split("(")) <= 2

        op = step.split("(")[0]
        args = step.split("(")[1]
        
        arg1 = args.split(",")[0].strip()
        arg2 = args.split(",")[1].strip(")").strip()
        
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
        steps = program2.split("),")
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0]
            args = step.split("(")[1]
            
            arg1 = args.split(",")[0].strip()
            arg2 = args.split(",")[1].strip(")").strip()
            
            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False
                    
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                        
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
    except:
        return False
    
    

    def symbol_recur(step, step_dict):
        
        op = step.split("(")[0].strip()
        args = step.split("(")[1].strip()
        
        arg1 = args.split(",")[0].strip()
        arg2 = args.split(",")[1].strip(")").strip()
        
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


    # derive symbolic program 1
    steps = program1.split("),")
    # print(steps)
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    # print(sym_prog1)
    
    # derive symbolic program 2
    steps = program2.split("),")
    sym_prog2 = symbol_recur(steps[-1], step_dict_2)
    # print(sym_prog2)
    
    
    return simplify(sym_prog1 + " - " + sym_prog2) == 0


def correct_ids(json_in, json_out):
    '''
    remove duplicate ids
    '''
    
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
        
    id_list = []
    for each_data in data:
        if each_data["id"] in id_list:
            while each_data["id"] in id_list:
                ind = int(each_data["id"][-1])
                each_data["id"] = each_data["id"][:-1] + str(ind + 1)
                
            id_list.append(each_data["id"])
            
            # print(each_data["id"])
            # print(ind)
        else:
            id_list.append(each_data["id"])

    id_list = []
    for each_data in data:
        assert each_data["id"] not in id_list
        id_list.append(each_data["id"])
        
    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)


def check_same_file(json_1, json_2):
    '''
    check split have same file
    '''

    
    with open(json_1) as f_in:
        data1 = json.load(f_in)

    with open(json_2) as f_in:
        data2 = json.load(f_in)


    id_list = []

    for each_data in data1:
        filename = each_data["id"].split("-")[0]
        if filename not in id_list:
            id_list.append(filename)

    print(len(id_list))
    for each_data in data2:
        filename = each_data["id"].split("-")[0] 
        assert filename not in id_list



def test_user_op(json_in):
    users = {
        "MBarrus": 0,
        "CRussell": 0,
        "Tmugera": 0,
        "VMaia": 0,
        "qsopjani": 0,
        "reemamoussa": 0,
        "iana": 0,
        "dylan": 0
    }
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
    res = []
    for each_data in data:
        program = each_data["qa"]["program"]
        this_user = each_data["annotator"]
        
        if "compare_larger" in program:
            users[this_user] += 1
            
            
    print(users)
            

def process_compare(json_in, json_out):
    '''
    turn compare larger to greater
    remove craig
    '''
    
    with open(json_in) as f_in:
        data = json.load(f_in)
        
    res = []
    for each_data in data:
        program = each_data["qa"]["program"]
        this_user = each_data["annotator"]
        
        
        if "compare_larger" in program:
            if this_user == "CRussell":
                continue
            
        program = program.replace("compare_larger", "greater")
        each_data["qa"]["program"] = program
        res.append(each_data)
    
    
    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)
        
        
    print(len(data))
    print(len(res))



def get_tf_idf_query_similarity(allDocs, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc

    return: cosine similarity between query and all docs
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words='english')
    docs_tfidf = vectorizer.fit_transform(allDocs)
    
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    
    # print(cosineSimilarities)
    return cosineSimilarities


def get_top_tfidf(json_in, json_out, topn, max_len):
    '''
    get top tf idf except the gold inds
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    res = []
    for each_data in data:

        question = each_data["qa"]["question"]
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table = each_data["table"]

        gold_inds = each_data["qa"]["gold_inds"]
        context = ""
        for each_con in gold_inds:
            context += gold_inds[each_con]
            context += " "

        gold_text_ids = []
        gold_table_ids = []

        for tmp in gold_inds:
            if "text" in tmp:
                gold_text_ids.append(int(tmp.replace("text_", "")))
            else:
                gold_table_ids.append(int(tmp.replace("table_", "")))

        all_text = pre_text + post_text
        all_text_ids = range(len(pre_text) + len(post_text))
        all_table_ids = range(1, len(table))

        all_docs = []
        for tmp in all_text_ids:
            all_docs.append(all_text[tmp])
            
        for tmp in all_table_ids:
            all_docs.append(table_row_to_text(table[0], table[tmp]))
            
        tfidf_sim_mat = get_tf_idf_query_similarity(all_docs, question)
        
        tfidf_dict = {}
        for ind, score in enumerate(tfidf_sim_mat):
            tfidf_dict[ind] = score
            
        sorted_dict = sorted(tfidf_dict.items(), key=lambda kv: kv[1], reverse=True)

        tfidf_topn = {}
        acc_len = len(context.split(" "))
        for tmp in sorted_dict:
            if len(tfidf_topn) + len(gold_inds) >= topn:
                break
            if tmp[0] < len(all_text):
                if tmp[0] not in gold_text_ids:
                    if acc_len + len(all_docs[tmp[0]].split(" ")) < max_len:
                        assert "text_" + str(tmp[0]) not in tfidf_topn
                        tfidf_topn["text_" + str(tmp[0])] = all_docs[tmp[0]]
                        acc_len += len(all_docs[tmp[0]].split(" "))
            else:
                this_table_tmp = tmp[0] - len(all_text) + 1
                if this_table_tmp not in gold_table_ids:
                    if acc_len + len(all_docs[tmp[0]].split(" ")) < max_len:
                        assert "table_" + str(this_table_tmp) not in tfidf_topn
                        tfidf_topn["table_" + str(this_table_tmp)] = all_docs[tmp[0]]
                        acc_len += len(all_docs[tmp[0]].split(" "))


        each_data["qa"]["tfidftopn"] = tfidf_topn
        # acc input: text first
        all_text_in = {}
        all_table_in = {}
        for tmp in gold_inds:
            if "text" in tmp:
                all_text_in[tmp] = gold_inds[tmp]
            elif "table" in tmp:
                all_table_in[tmp] = gold_inds[tmp]

        for tmp in tfidf_topn:
            if "text" in tmp:
                all_text_in[tmp] = tfidf_topn[tmp]
            elif "table" in tmp:
                all_table_in[tmp] = tfidf_topn[tmp]

        this_model_input = []

        sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        this_model_input.extend(sorted_dict)

        sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        this_model_input.extend(sorted_dict)

        each_data["qa"]["model_input"] = this_model_input
        res.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)



def sliding_window_test(json_in):

    with open(json_in) as f_in:
        data = json.load(f_in)

    res_table = {}
    res_text = {}
    agg_table = {"0": 0, "<=3": 0, ">3": 0}
    agg_text = {"0": 0, "<=3": 0, ">3": 0}
    for each_data in data:
        gold_inds = each_data["qa"]["gold_inds"]
        table_inds = []
        text_inds = []
        for tmp in gold_inds:
            if "table" in tmp:
                table_inds.append(int(tmp.split("_")[1]))
            else:
                text_inds.append(int(tmp.split("_")[1]))

        if len(table_inds) > 0:
            slide_table = max(table_inds) - min(table_inds)
            if slide_table not in res_table:
                res_table[slide_table] = 0
            res_table[slide_table] += 1
            if slide_table == 0:
                agg_table["0"] += 1
            elif slide_table <= 3:
                agg_table["<=3"] += 1
            else:
                agg_table[">3"] += 1

        if len(text_inds) > 0:
            slide_text = max(text_inds) - min(text_inds)
            if slide_text not in res_text:
                res_text[slide_text] = 0
            res_text[slide_text] += 1
            if slide_text == 0:
                agg_text["0"] += 1
            elif slide_text <= 3:
                agg_text["<=3"] += 1
            else:
                agg_text[">3"] += 1


    print("table")
    print(res_table)
    print("text")
    print(res_text)

    print("table")
    print(agg_table)
    print("text")
    print(agg_text)



def add_table_des(json_in, json_out):
    '''
    add the table intro if have table evi
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)


    num_add_last = 0
    for each_data in data:
        pre_text = each_data["pre_text"]
        gold_inds = each_data["qa"]["gold_inds"]

        last_pre = "text_" + str(len(pre_text) - 1)

        if_table = 0
        if_last_pre = 0

        for tmp in gold_inds:
            if "table" in tmp:
                if_table = 1
            if tmp == last_pre:
                if_last_pre = 1


        if if_table and if_last_pre == 0:
            num_add_last += 1
            each_data["qa"]["gold_inds"][last_pre] = pre_text[-1]

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print("all: ", len(data))
    print("add table des: ", num_add_last)


def final_filter(json_in, json_out):
    '''
    final filtering!
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    res = []
    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]

        all_text = "".join(pre_text) + "".join(post_text)
        if "............." in all_text or "************" in all_text:
            continue

        res.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    print("all: ", len(data))
    print("res: ", len(res))




def all_len_test(json_in):

    with open(json_in) as f_in:
        data = json.load(f_in)

    res = []
    all_len = 0
    max_len = 0
    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]

        table = each_data["table"]

        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        all_text = " ".join(each_data["pre_text"]) + " " + " ".join(each_data["post_text"]) + " " + table_text
        all_len += len(all_text.split(" "))
        if len(all_text.split(" ")) > max_len:
            max_len = len(all_text.split(" "))

    print(float(all_len) / len(data))
    print(max_len)



def add_recursive(json_in, json_out):
    '''
    turn program into nested form
    subtract(137582, 143746), divide(#0, 143746)
    divide(72, multiply(6, 210))
    '''


    with open(json_in) as f_in:
        data = json.load(f_in)


    def recur_prog(this_step):

        arg1 = this_step.split(",")[0].split("(")[1].strip()
        arg2 = this_step.split(",")[1].strip().strip(")").strip()
        op = this_step.split(",")[0].split("(")[0].strip()
        # print(arg1)

        if "#" in arg1:
            ind1 = int(arg1.replace("#", ""))
            arg1_res = recur_prog(step_dict[ind1])
        else:
            arg1_res = arg1

        
        if "#" in arg2:
            ind2 = int(arg2.replace("#", ""))
            arg2_res = recur_prog(step_dict[ind2])
        else:
            arg2_res = arg2


        return (op + "(" + arg1_res + ", " + arg2_res + ")").strip()


    for each_data in data:
        program = each_data["qa"]["program"]

        step_dict = {}
        for ind, step in enumerate(program.split("),")):
            step = step.strip().strip(")")

            step_dict[ind] = step + ")"

        last_step = program.split("),")[-1]

        re_prog = recur_prog(last_step)

        each_data["qa"]["program_re"] = re_prog

        # print("#######")
        # print(program)
        # print(re_prog)



    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)



def generate_fewshot_train(json_in, json_out, num):
    '''
    get few shot training set
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    train = data[:num]
    print(len(train))

    with open(json_out, "w") as f:
        json.dump(train, f, indent=4)


def correct_id_last(new_data, old_all_json, json_out):
    '''
    last batch of data:
    1. correct ids
    '''

    with open(new_data) as f_in:
        data = json.load(f_in)

    print("new data: ", len(data))

    id_list = []
    with open(old_all_json) as f_in:
        data_all_old = json.load(f_in)

    print("old data: ", len(data_all_old))

    for each_data in data_all_old:
        assert each_data["id"] not in id_list
        id_list.append(each_data["id"])
    
    for each_data in data:
        if each_data["id"] in id_list:
            while each_data["id"] in id_list:
                ind = int(each_data["id"][-1])
                each_data["id"] = each_data["id"][:-1] + str(ind + 1)
                
            id_list.append(each_data["id"])
            
            # print(each_data["id"])
            # print(ind)
        else:
            id_list.append(each_data["id"])

    id_list = []
    for each_data in data:
        assert each_data["id"] not in id_list
        id_list.append(each_data["id"])


    print(len(data))
    # with open(old_train) as f_in:
    #     data_old_train = json.load(f_in)

    # data_new_train = data_old_train + data

    # print("new train: ", len(data_new_train))

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def merge_last_batch(last_batch, train, dev, test, train_new, dev_new, test_new):
    '''
    merge last batch into existing train dev test
    '''

    with open(train) as f_in:
        data_train = json.load(f_in)

    with open(dev) as f_in:
        data_dev = json.load(f_in)

    with open(test) as f_in:
        data_test = json.load(f_in)

    train_files = []
    for each_data in data_train:
        if each_data["filename"] not in train_files:
            train_files.append(each_data["filename"])


    dev_files = []
    for each_data in data_dev:
        if each_data["filename"] not in dev_files:
            dev_files.append(each_data["filename"])


    test_files = []
    for each_data in data_test:
        if each_data["filename"] not in test_files:
            test_files.append(each_data["filename"])


    with open(last_batch) as f_in:
        data_last_batch = json.load(f_in)


    for each_data in data_last_batch:
        this_filename = each_data["filename"]
        if this_filename not in test_files + dev_files:
            data_train.append(each_data)
            continue

        if this_filename not in dev_files + train_files:
            data_test.append(each_data)
            continue

        if this_filename not in test_files + train_files:
            data_dev.append(each_data)
            continue

        

    print("train: ", len(data_train))
    print("dev: ", len(data_dev))
    print("test: ", len(data_test))


    with open(train_new, "w") as f:
        json.dump(data_train, f, indent=4)

    with open(dev_new, "w") as f:
        json.dump(data_dev, f, indent=4)

    with open(test_new, "w") as f:
        json.dump(data_test, f, indent=4)


def get_final_data(train, dev, test, json_out):

    with open(train) as f_in:
        data_train = json.load(f_in)

    with open(dev) as f_in:
        data_dev = json.load(f_in)

    with open(test) as f_in:
        data_test = json.load(f_in)


    data_all = data_train + data_dev + data_test
    random.shuffle(data_all)
    print("All: ", len(data_all))

    with open(json_out, "w") as f:
        json.dump(data_all, f, indent=4)




def get_release_data(json_in, json_out):
    '''
    get the data to release,
    remove annotator names
    '''

    with open(json_in) as f_in:
        data = json.load(f_in)

    print(len(data))

    for each_data in data:
        del each_data["annotator"]


    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)






root = "/mnt/george_bhd/zhiyuchen/finQA/"
data_folder = root + "data/"
our_data = root + "dataset/"

# our_data_batch = our_data + "json_processed/"
# our_data_agg_1 = our_data + "processed_1.json"

# our_data_final = our_data + "processed_1_final.json"
# invalid_final = our_data + "processed_1_final_invalid.json"
# train = our_data + "train.json"
# valid = our_data + "dev.json"
# test = our_data + "test.json"

# train_1000 = our_data + "train_1000.json"
# train_2000 = our_data + "train_2000.json"
# train_3000 = our_data + "train_3000.json"
# train_4000 = our_data + "train_4000.json"

# test_reprog = our_data + "test_reprog.json"

# our_data_correctid_final = our_data + "processed_1_correctid_final.json"
# our_data_greater_final = our_data + "processed_1_greater_final.json"
# our_data_exe_final = our_data + "processed_1_exe_final.json"
# our_data_tfidf = our_data + "processed_1_tfidf_final.json"
# our_data_re = our_data + "processed_1_re.json"
# our_data_final_last = our_data + "processed_1_final_last.json"


### for last bacth
last_batch_folder = our_data_batch = our_data + "json_processed_last/"

our_data_agg_1 = last_batch_folder + "processed_1.json"
our_data_final = last_batch_folder + "processed_1_final.json"
invalid_final = last_batch_folder + "processed_1_final_invalid.json"
our_data_correctid_final = last_batch_folder + "processed_1_correctid_final.json"
our_data_greater_final = last_batch_folder + "processed_1_greater_final.json"
our_data_exe_final = last_batch_folder + "processed_1_exe_final.json"
our_data_tfidf = last_batch_folder + "processed_1_tfidf_final.json"
our_data_re = last_batch_folder + "processed_1_re.json"
our_data_final_last = last_batch_folder + "processed_1_last_batch.json"


# process_agg(our_data_batch, our_data_agg_1)


# process_map(our_data_agg_1, our_data_final, invalid_final)

# correct_ids(our_data_final, our_data_correctid_final)


# process_compare(our_data_correctid_final, our_data_greater_final)

# eval_program(our_data_greater_final, our_data_exe_final)


# get_top_tfidf(our_data_exe_final, our_data_tfidf, topn=3, max_len=290)

# add_recursive(our_data_tfidf, our_data_final_last)

# old_all_json = our_data + "processed_1_final_last.json"
# old_train = our_data + "train.json"
# json_out = our_data + "last_batch.json"
# correct_id_last(our_data_final_last, old_all_json, json_out)


# last_batch = our_data + "last_batch.json"
# train = our_data + "train.json"
# valid = our_data + "dev.json"
# test = our_data + "test.json"

# train_new = our_data + "train_new.json"
# valid_new = our_data + "dev_new.json"
# test_new = our_data + "test_new.json"

# merge_last_batch(last_batch, train, valid, test, train_new, valid_new, test_new)

# train = our_data + "train_new.json"
# dev = our_data + "dev_new.json"
# test = our_data + "test_new.json"
# json_out = our_data + "our_data_final.json"

# get_final_data(train, dev, test, json_out)

# train_test_split(our_data_final_last, train, valid, test)


### get release data
json_in = our_data + "test_new.json"
json_out = our_data + "test_release.json"
get_release_data(json_in, json_out)



# generate_fewshot_train(train, train_4000, num=4000)


# train_ori = our_data + "train_retrieve_top3.json"
# valid_ori = our_data + "dev_retrieve_top3.json"
# test_ori = our_data + "test_retrieve_top3.json"

# train_re = our_data + "train_retrieve_re.json"
# valid_re = our_data + "dev_retrieve_re.json"
# test_re = our_data + "test_retrieve_re.json"

# add_recursive(train_ori, train_re)
# add_recursive(valid_ori, valid_re)
# add_recursive(test_ori, test_re)

# check_same_file(valid_new, test_new)

# test = program_tokenization(
#     "multiply(add(28, const_1), add(add(28, const_1), const_1))")
# print(test)


# test_user_op(our_data_exe_final)

# program1 = "multiply(1.1, const_1000), divide(#0, 6205)"
# program2 = "multiply(const_1000, 1.1)"

# res = equal_program(program1, program2)
# print(res)


# data_stats(our_data_exe_final)


# sliding_window_test(our_data_tfidf)



# get_year(dir)

# header = [
#                 "( in millions )",
#                 "payments due by period total",
#                 "payments due by period < 1 year",
#                 "payments due by period 1-3 years",
#                 "payments due by period 4-5 years",
#                 "payments due by period > 5 years"
#             ]
# row =  [
#                 "operating leases ( 3 )",
#                 "143.2",
#                 "22.5",
#                 "41.7",
#                 "37.1",
#                 "41.9"
#             ]
# res = table_row_to_text(header, row)
# print(res)


# all_len_test(our_data_final_last)