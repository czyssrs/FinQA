import csv
import re
import sys
import os
import json
import copy
import Levenshtein
import numpy as np
csv.field_size_limit(sys.maxsize)


def read_table(table_json_file, file_out):
    '''
    filter: at most change length 1 time
    '''

    res = {}
    length_change = {0: 0, 1: 0, 2: 0, 3: 0}

    num_invalid = 0
    invalid_list = ["page", "pagenumber", "description", "exhibitnumber"]
    with open(table_json_file, 'r') as fp:

        for line in fp:
            if_valid = 1
            sample = json.loads(line)
            bbox = sample["bbox"]
            left = bbox[0]
            right = bbox[2]

            this_table = []
            this_row = []
            prev_right = 0.0

            num_len_change = 0

            has_number = 0

            for item in sample["html"]["cells"]:
                this_token = "".join(item["tokens"]).strip()

                if this_token.lower() in invalid_list or "certification of" in this_token.lower() or "consolidated statements" in this_token.lower():
                    # content page
                    if_valid = 0
                    break

                if this_token.replace(",", "").replace("$", "").replace(".", "").isdigit():
                    has_number = 1

                if "bbox" in item:
                    this_right = item["bbox"][2]
                else:
                    this_right = prev_right

                if this_right < prev_right:
                    # change row
                    # check length
                    if len(this_table) != 0 and len(this_table[-1]) != len(this_row):
                        # print("incompatible length")
                        num_len_change += 1
                    this_table.append(this_row[:])

                    # print("\t#\t".join(this_row))
                    # print("\n")

                    del this_row[:]
                this_row.append(this_token)

                prev_right = this_right

            if if_valid == 0:
                num_invalid += 1
                continue

            if has_number == 0:
                num_invalid += 1
                continue

            this_table.append(this_row[:])
            if num_len_change < 2:
                res[sample['filename']] = this_table
            if num_len_change in length_change:
                length_change[num_len_change] += 1
            # print(sample['filename'])
            # print("\n##############\n")

    print(length_change)
    with open(file_out, "w") as f:
        json.dump(res, f, indent=4)

    print("All: ", len(res))
    print("Invalid: ", num_invalid)
    return res


def test_multiple_table(json_in):

    res = {}
    with open(json_in, 'r') as fp:

        for line in fp:
            sample = json.loads(line)
            name = sample['filename']

            if name in res:
                res[name] += 1
            else:
                res[name] = 1

    num_single = 0
    for name in res:
        if res[name] == 1:
            num_single += 1

    print("All: ", len(res))
    print("Single: ", num_single)


def filter_single_table(json_in, json_out):
    '''
    filter out pages with one table
    '''
    res = {}
    with open(json_in, 'r') as fp:

        for line in fp:
            sample = json.loads(line)
            name = sample['filename']

            if name in res:
                res[name] += 1
            else:
                res[name] = 1

    # num_single = 0
    # for name in res:
    #     if res[name] == 1:
    #         num_single += 1

    # print("All: ", len(res))
    # print("Single: ", num_single)
    num_write = 0
    with open(json_out, "w") as fo:
        with open(json_in, 'r') as fp:

            for line in fp:
                sample = json.loads(line)
                name = sample['filename']

                if res[name] == 1:
                    fo.write(line)
                    num_write += 1

    print("Write: ", num_write)


def convert_num_text(text_in):
    '''
    convert the number in text, remove comma
    '''
    res = []
    for token in text_in.split(" "):
        token = token.strip()
        if token != "":
            if "," in token[1:-1]:
                out = token[0] + token[1:-1].replace(",", "") + token[-1]
                res.append(out)
            else:
                res.append(token)

    return " ".join(res)


def convert_num_table(table_in):
    '''
    convert the number in text, remove comma
    '''

    out = []
    for row in table_in:
        out_row = []
        for token in row:
            if "," in token[1:-1]:
                this_out = token[0] + token[1:-1].replace(",", "") + token[-1]
                out_row.append(this_out)
            else:
                out_row.append(token)

        out.append(out_row)

    return out


def convert2final(json_in, json_out, max_len):
    '''
    unify row length of a table
    truncate text to max 300
    max table row: 10
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = []
    num_invalid = 0
    max_len = 0
    all_len = 0

    for data in data_all:
        table = data["table"]
        pre_text = data["pre_text"]
        post_text = data["post_text"]

        if len(table) <= 10 or len(table) > 20:
            num_invalid += 1
            continue

        data["table_ori"] = copy.deepcopy(table)
        data["pre_text_ori"] = copy.deepcopy(pre_text)
        data["post_text_ori"] = copy.deepcopy(post_text)

        pre_text_str = " ".join(data["pre_text_ori"])
        post_text_str = " ".join(data["post_text_ori"])


        # if len(pre_text_str.split(" ")) + len(post_text_str.split(" ")) < max_len:
        #     data["pre_text"] = data["pre_text_ori"]
        #     data["post_text"] = data["post_text_ori"]

        # elif len(pre_text_str.split(" ")) > max_len / 2 and len(post_text_str.split(" ")) > max_len / 2:
        #     '''
        #     both truncate to 1/2
        #     '''
        #     trun_pre_text = []
        #     ind = len(data["pre_text_ori"]) - 1
        #     cur_len = 0
        #     while cur_len < max_len / 2:
        #         trun_pre_text.append(data["pre_text_ori"][ind])
        #         cur_len += len(data["pre_text_ori"][ind].split(" "))
        #         ind -= 1

        #     trun_pre_text.reverse()

        #     trun_post_text = []
        #     ind = 0
        #     cur_len = 0
        #     while cur_len < max_len / 2:
        #         trun_post_text.append(data["post_text_ori"][ind])
        #         cur_len += len(data["post_text_ori"][ind].split(" "))
        #         ind += 1

        #     data["pre_text"] = trun_pre_text
        #     data["post_text"] = trun_post_text

        # elif len(pre_text_str.split(" ")) > max_len / 2:
        #     '''
        #     pre > 1/2, post < 1/2
        #     '''

        #     max_pre = max_len - len(post_text_str.split(" "))

        #     trun_pre_text = []
        #     ind = len(data["pre_text_ori"]) - 1
        #     cur_len = 0
        #     while cur_len < max_pre:
        #         trun_pre_text.append(data["pre_text_ori"][ind])
        #         cur_len += len(data["pre_text_ori"][ind].split(" "))
        #         ind -= 1

        #     trun_pre_text.reverse()
        #     data["pre_text"] = trun_pre_text
        #     data["post_text"] = data["post_text_ori"]

        # elif len(pre_text_str.split(" ")) < max_len / 2:
        #     '''
        #     pre < 1/2, post > 1/2
        #     '''

        #     max_post = max_len - len(pre_text_str.split(" "))

        #     trun_post_text = []
        #     ind = 0
        #     cur_len = 0
        #     while cur_len < max_post:
        #         trun_post_text.append(data["post_text_ori"][ind])
        #         cur_len += len(data["post_text_ori"][ind].split(" "))
        #         ind += 1

        #     data["pre_text"] = data["pre_text_ori"]
        #     data["post_text"] = trun_post_text

        # else:
        #     print("error")
        #     num_invalid += 1
        #     continue


        data["pre_text"] = data["pre_text_ori"]
        data["post_text"] = data["post_text_ori"]


        larger_len = 0
        for row in table:
            if len(row) > larger_len:
                larger_len = len(row)

        for ind_t, row in enumerate(table):
            if len(row) < larger_len:
                # for rows with only 2 elems
                num_makeup = larger_len - len(row) + 1
                new_row = []
                start_ind = 0
                start_token = ""
                for ind, token in enumerate(row):
                    if token != "":
                        start_ind = ind
                        start_token = token
                        break

                for ind in range(start_ind):
                    new_row.append("")

                for ind in range(num_makeup):
                    new_row.append(start_token)

                for ind in range(larger_len - len(new_row)):
                    new_row.append("")

                new_row = new_row[:larger_len]
                table[ind_t] = new_row

        # check corrupted table
        all_cell = 0
        empty_cell = 0
        for row in table:
            for ele in row:
                all_cell += 1
                if ele.strip() == "":
                    empty_cell += 1

        if empty_cell / all_cell > 0.4:
            num_invalid += 1
            continue

        if empty_cell > 2:
            num_invalid += 1
            continue

        # process comma in numbers
        res_pre = []
        for each_sent in data["pre_text"]:
            res_pre.append(convert_num_text(each_sent))
        data["pre_text"] = res_pre

        res_post = []
        for each_sent in data["post_text"]:
            res_post.append(convert_num_text(each_sent))
        data["post_text"] = res_post

        data["table"] = convert_num_table(data["table"])

        this_len = len(pre_text_str.split(" ")) + len(post_text_str.split(" "))
        if this_len > max_len:
            max_len = this_len
        all_len += this_len

        res.append(data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    print("All: ", len(res))
    print("Invalid empty table: ", num_invalid)
    print("Max len: ", max_len)
    print("Avg len: ", float(all_len) / len(res))


def convert2final_old(json_in, json_out, max_len):
    '''
    unify row length of a table
    truncate text to max 300
    max table row: 10
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = []
    num_invalid = 0
    for data in data_all:
        table = data["table"]
        pre_text = data["pre_text"]
        post_text = data["post_text"]

        if len(table) > 10:
            num_invalid += 1
            continue

        data["table_ori"] = copy.deepcopy(table)
        data["pre_text_ori"] = copy.deepcopy(pre_text)
        data["post_text_ori"] = copy.deepcopy(post_text)

        pre_text_str = " ".join(data["pre_text_ori"])
        post_text_str = " ".join(data["post_text_ori"])

        if len(pre_text_str.split(" ")) + len(post_text_str.split(" ")) < max_len:
            data["pre_text"] = data["pre_text_ori"]
            data["post_text"] = data["post_text_ori"]

        elif len(pre_text_str.split(" ")) > max_len / 2 and len(post_text_str.split(" ")) > max_len / 2:
            '''
            both truncate to 1/2
            '''
            trun_pre_text = []
            ind = len(data["pre_text_ori"]) - 1
            cur_len = 0
            while cur_len < max_len / 2:
                trun_pre_text.append(data["pre_text_ori"][ind])
                cur_len += len(data["pre_text_ori"][ind].split(" "))
                ind -= 1

            trun_pre_text.reverse()

            trun_post_text = []
            ind = 0
            cur_len = 0
            while cur_len < max_len / 2:
                trun_post_text.append(data["post_text_ori"][ind])
                cur_len += len(data["post_text_ori"][ind].split(" "))
                ind += 1

            data["pre_text"] = trun_pre_text
            data["post_text"] = trun_post_text

        elif len(pre_text_str.split(" ")) > max_len / 2:
            '''
            pre > 1/2, post < 1/2
            '''

            max_pre = max_len - len(post_text_str.split(" "))

            trun_pre_text = []
            ind = len(data["pre_text_ori"]) - 1
            cur_len = 0
            while cur_len < max_pre:
                trun_pre_text.append(data["pre_text_ori"][ind])
                cur_len += len(data["pre_text_ori"][ind].split(" "))
                ind -= 1

            trun_pre_text.reverse()
            data["pre_text"] = trun_pre_text
            data["post_text"] = data["post_text_ori"]

        elif len(pre_text_str.split(" ")) < max_len / 2:
            '''
            pre < 1/2, post > 1/2
            '''

            max_post = max_len - len(pre_text_str.split(" "))

            trun_post_text = []
            ind = 0
            cur_len = 0
            while cur_len < max_post:
                trun_post_text.append(data["post_text_ori"][ind])
                cur_len += len(data["post_text_ori"][ind].split(" "))
                ind += 1

            data["pre_text"] = data["pre_text_ori"]
            data["post_text"] = trun_post_text

        else:
            print("error")
            num_invalid += 1
            continue

        larger_len = 0
        for row in table:
            if len(row) > larger_len:
                larger_len = len(row)

        for ind_t, row in enumerate(table):
            if len(row) < larger_len:
                # for rows with only 2 elems
                num_makeup = larger_len - len(row) + 1
                new_row = []
                start_ind = 0
                start_token = ""
                for ind, token in enumerate(row):
                    if token != "":
                        start_ind = ind
                        start_token = token
                        break

                for ind in range(start_ind):
                    new_row.append("")

                for ind in range(num_makeup):
                    new_row.append(start_token)

                for ind in range(larger_len - len(new_row)):
                    new_row.append("")

                new_row = new_row[:larger_len]
                table[ind_t] = new_row

        # check corrupted table
        all_cell = 0
        empty_cell = 0
        for row in table:
            for ele in row:
                all_cell += 1
                if ele.strip() == "":
                    empty_cell += 1

        if empty_cell / all_cell > 0.4:
            num_invalid += 1
            continue

        if empty_cell > 2:
            num_invalid += 1
            continue

        # process comma in numbers
        res_pre = []
        for each_sent in data["pre_text"]:
            res_pre.append(convert_num_text(each_sent))
        data["pre_text"] = res_pre

        res_post = []
        for each_sent in data["post_text"]:
            res_post.append(convert_num_text(each_sent))
        data["post_text"] = res_post

        data["table"] = convert_num_table(data["table"])

        res.append(data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    print("All: ", len(res))
    print("Invalid empty table: ", num_invalid)

def if_in_table(line, table):
    '''
    if this line is actually a table line
    '''

    for row in table:
        # this_row = "".join(row).replace(" ", "").lower()
        in_row = 0
        for token in line.strip().split(" "):
            this_token = token.replace(" ", "").lower()
            for row_ele in row:
                this_row_ele = row_ele.replace(" ", "").lower()
                if this_token in this_row_ele:
                    in_row += 1

        if in_row / len(line.strip().split(" ")) >= 0.8:
            return True

    return False


def if_in_table_editdistance(line, table):
    '''
    if this line is actually a table line
    using Levenshtein distance
    '''

    line = line.replace(".", "")
    for row in table:
        this_row = "".join(row).replace(" ", "").replace(".", "").lower()
        this_line = line.replace(" ", "").lower()
        distance = Levenshtein.distance(this_row, this_line)

        if distance < 3:
            return True

    return False


def extract_text(table_json, pdf_dir, json_out):
    '''
    extract pre and post text from pdfs


    {
        "filename": str
        "table": list of lists (list of table rows)
        "pre_text": list of texts (list of paragraphs)
        "post_text": list of texts (list of paragraphs)
    }
    '''

    from tika import parser

    with open(table_json) as f:
        data = json.load(f)
    print(len(data))

    res = []

    for filename in data:
        table = data[filename]
        pdf_in = os.path.join(pdf_dir, filename)

        try:
            raw = parser.from_file(pdf_in)
        except:
            continue

        this_pre_text = []
        this_post_text = []

        # 0: text, 1: table
        cont_type = []
        cont_list = []

        try:
            cont_list_ori = raw['content'].split("\n")
        except:
            continue
        ori_str = ""
        for line in cont_list_ori:
            # print(line)
            if line.strip() != "" and len(line.strip()) > 5:
                # ori_str += (" " + line.strip())
                cont_list.append(line.strip())

        for line in cont_list:
            # print(line)

            if if_in_table_editdistance(line, table):
                cont_type.append(1)
            else:
                cont_type.append(0)

        for ind, tmp in enumerate(cont_type):
            if tmp == 0:
                this_pre_text.append(cont_list[ind])
            else:
                break

        cont_type.reverse()
        cont_list_reverse = cont_list[:]
        cont_list_reverse.reverse()

        for ind, tmp in enumerate(cont_type):
            if tmp == 0:
                this_post_text.append(cont_list_reverse[ind])
            else:
                break

        this_post_text.reverse()

        # convert text into sentences
        all_pre = " ".join(this_pre_text)
        # if ". . . . . . . . . ." in all_pre:
        #     continue
        this_pre_text = []
        for sent in all_pre.split(". "):
            this_pre_text.append(sent.strip() + ".")

        all_post = " ".join(this_post_text)
        # if ". . . . . . . . . ." in all_post:
        #     continue
        this_post_text = []
        for sent in all_post.split(". "):
            this_post_text.append(sent.strip() + ".")

        this_data = {
            "filename": filename,
            "table": table,
            "pre_text": this_pre_text,
            "post_text": this_post_text
        }

        # print("######################")
        # print("original cont")
        # for line in cont_list:
        #     print(line)

        # print("\n")
        # print("table")
        # for line in table:
        #     print(" ".join(line))

        # print("\n")

        # print("pre text")
        # print("\n".join(this_pre_text))

        # print("post text")
        # print("\n".join(this_post_text))

        # print("\n")

        res.append(this_data)
        if len(res) % 1000 == 0:
            print(len(res))

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    print("All: ", len(res))


def expert_to_csv(json_in, csv_out, start, end):
    '''
    turn to csv on amt
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    count = 0
    with open(csv_out, "w") as f_out:
        fieldnames = ['id', 'table', 'pre_text', 'post_text']
        writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for data in data_all[start: end]:
            this_id = data["filename"]
            table = data["table"]
            pre_text = data["pre_text"]
            post_text = data["post_text"]

            amt_table = "<table class='wikitable'>"
            i = 1
            for row in table:
                amt_table += "<tr>"
                amt_table += ("<td>" + str(i) + "</td>")
                i += 1
                for ele in row:
                    amt_table += ("<td>" + ele.strip() + "</td>")
                amt_table += "</tr>"
            amt_table += "</table>"

            # amt_pre_text = " ".join(pre_text)
            # amt_post_text = " ".join(post_text)

            amt_pre_text = ""
            amt_post_text = ""

            ind = 1
            for each_sent in pre_text:
                amt_pre_text += ("<strong>[" + str(ind) +
                                 "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            for each_sent in post_text:
                amt_post_text += ("<strong>[" + str(ind) +
                                  "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            writer.writerow({"id": this_id, "table": amt_table,
                             "pre_text": amt_pre_text, "post_text": amt_post_text})
            count += 1

    print(count)


def read_steps_csv(row, i):
    '''
    read calculation steps from csv row
    '''

    # for calculation steps
    steps = []
    cal_1 = row["Answer.cal{}-1".format(i)]
    if "tmp" not in cal_1:
        steps.append({
            "op": cal_1,
            "arg1": row["Answer.a{}-1_0".format(i)],
            "arg2": row["Answer.a{}-1_1".format(i)],
            "res": row["Answer.r{}-1".format(i)]
        })

    if len(steps) == 1:
        cal_2 = row["Answer.cal{}-2".format(i)]
        if "tmp" not in cal_2:
            steps.append({
                "op": cal_2,
                "arg1": row["Answer.a{}-2_0".format(i)],
                "arg2": row["Answer.a{}-2_1".format(i)],
                "res": row["Answer.r{}-2".format(i)]
            })

    if len(steps) == 2:
        cal_3 = row["Answer.cal{}-3".format(i)]
        if "tmp" not in cal_3:
            steps.append({
                "op": cal_3,
                "arg1": row["Answer.a{}-3_0".format(i)],
                "arg2": row["Answer.a{}-3_1".format(i)],
                "res": row["Answer.r{}-3".format(i)]
            })

    if len(steps) == 3:
        cal_4 = row["Answer.cal{}-4".format(i)]
        if "tmp" not in cal_4:
            steps.append({
                "op": cal_4,
                "arg1": row["Answer.a{}-4_0".format(i)],
                "arg2": row["Answer.a{}-4_1".format(i)],
                "res": row["Answer.r{}-4".format(i)]
            })

    if len(steps) == 4:
        cal_5 = row["Answer.cal{}-5".format(i)]
        if "tmp" not in cal_5:
            steps.append({
                "op": cal_5,
                "arg1": row["Answer.a{}-5_0".format(i)],
                "arg2": row["Answer.a{}-5_1".format(i)],
                "res": row["Answer.r{}-5".format(i)]
            })

    return steps


def get_res_test(csv_in, json_in, json_out):
    '''
    parse amt csv to json
    format:
        {
            "filename":
            "table":
            "pre_text":
            "post_text":
            "q1":
            {
                "question": ...
                "answer": ...
                "explanation": ...
                "steps": 
                [
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    ...
                ]
            }
        }
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = {}
    num_hits = 0
    num_questions = 0
    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row["AssignmentStatus"] == "Approved":
                # if row["Turkle.Username"] == "MBarrus":
                num_hits += 1
                this_id = row["Input.id"]

                q1 = row["Answer.q1"]
                if q1 != "" and len(q1) > 5:
                    num_questions += 1
                    a1 = row["Answer.a1"]
                    e1 = row["Answer.e1"]

                    steps = read_steps_csv(row, 1)
                    res[this_id] = []
                    this_q = {
                        "question": q1,
                        "answer": a1,
                        "explanation": e1,
                        "steps": steps
                    }
                    res[this_id].append(this_q)

                q2 = row["Answer.q2"]
                if q2 != "" and len(q2) > 5:
                    num_questions += 1
                    a2 = row["Answer.a2"]
                    e2 = row["Answer.e2"]

                    steps = read_steps_csv(row, 2)
                    this_q = {
                        "question": q2,
                        "answer": a2,
                        "explanation": e2,
                        "steps": steps
                    }
                    res[this_id].append(this_q)

    print("All: ", num_hits)
    print("Questions: ", num_questions)

    res_data = []
    for each_data in data_all:
        if each_data["filename"] in res:
            print(each_data["filename"])
            each_data["q1"] = res[each_data["filename"]][0]
            if len(res[each_data["filename"]]) > 1:
                each_data["q2"] = res[each_data["filename"]][1]

            res_data.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res_data, f, indent=4)

    print("All: ", len(res_data))


def get_res(csv_in, json_in, json_out, ann_name):
    '''
    parse amt csv to json
    format:
        {
            "filename":
            "table":
            "pre_text":
            "post_text":
            "q1":
            {
                "question": ...
                "answer": ...
                "explanation": ...
                "steps": 
                [
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    ...
                ]
            }
        }
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = {}
    num_hits = 0
    num_questions = 0
    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # if row["AssignmentStatus"] == "Approved":
            # if row["Turkle.Username"] == "MBarrus":
            num_hits += 1
            this_id = row["Input.id"]

            q1 = row["Answer.q1"]

            if q1 != "" and len(q1) > 5:
                num_questions += 1

                a1 = row["Answer.a1"]

                if "Answer.e1" in row:
                    e1 = row["Answer.e1"]
                else:
                    e1 = ""

                row_table1 = []
                row_text1 = []

                if "Answer.tableline_1" in row:
                    for tmp in row["Answer.tableline_1"].strip().split(","):
                        if tmp != "":
                            row_table1.append(tmp.strip())

                if "Answer.textline_1" in row:
                    for tmp in row["Answer.textline_1"].strip().split(","):
                        if tmp != "":
                            row_text1.append(tmp.strip())

                steps = read_steps_csv(row, 1)
                res[this_id] = []

                this_q = {
                    "question": q1,
                    "answer": a1,
                    "explanation": e1,
                    "steps": steps,
                    "table_rows": row_table1,
                    "text_rows": row_text1
                }
                res[this_id].append(this_q)


            q2 = row["Answer.q2"]

            if q2 != "" and len(q2) > 5:
                num_questions += 1

                a2 = row["Answer.a2"]

                if "Answer.e2" in row:
                    e2 = row["Answer.e2"]
                else:
                    e2 = ""

                row_table2 = []
                row_text2 = []

                if "Answer.tableline_2" in row:
                    for tmp in row["Answer.tableline_2"].strip().split(","):
                        if tmp != "":
                            row_table2.append(tmp.strip())

                if "Answer.textline_2" in row:
                    for tmp in row["Answer.textline_2"].strip().split(","):
                        if tmp != "":
                            row_text2.append(tmp.strip())

                steps = read_steps_csv(row, 2)
                this_q = {
                    "question": q2,
                    "answer": a2,
                    "explanation": e2,
                    "steps": steps,
                    "table_rows": row_table2,
                    "text_rows": row_text2
                }
                res[this_id].append(this_q)

            if this_id in res:
                # append annotator name
                if ann_name != "":
                    res[this_id].append(ann_name)
                else:
                    res[this_id].append(row["Turkle.Username"])

    print("All: ", num_hits)
    print("Questions: ", num_questions)

    res_data = []
    for each_data in data_all:
        if each_data["filename"] in res:
            # print(each_data["filename"])
            each_data["q1"] = res[each_data["filename"]][0]
            if len(res[each_data["filename"]]) > 2:
                each_data["q2"] = res[each_data["filename"]][1]

            each_data["annotator"] = res[each_data["filename"]][-1]

            res_data.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res_data, f, indent=4)

    print("All: ", len(res_data))


def get_res_eval(csv_in, json_in, json_out, ann_name):
    '''
    parse amt csv to json
    format:
        {
            "filename":
            "table":
            "pre_text":
            "post_text":
            "q1":
            {
                "question": ...
                "answer": ...
                "explanation": ...
                "steps": 
                [
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    ...
                ]
            }
        }
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = {}
    num_hits = 0
    num_questions = 0
    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # if row["AssignmentStatus"] == "Approved":
            # if row["Turkle.Username"] == "MBarrus":
            num_hits += 1
            this_id = row["Input.id"]

            q1 = row["Input.question"]

            if q1 != "" and len(q1) > 5:
                num_questions += 1

                a1 = row["Answer.answer"]

                if "Answer.e1" in row:
                    e1 = row["Answer.e1"]
                else:
                    e1 = ""

                row_table1 = []
                row_text1 = []

                if "Answer.tableline_1" in row:
                    for tmp in row["Answer.tableline_1"].strip().split(","):
                        if tmp != "":
                            row_table1.append(tmp.strip())

                if "Answer.textline_1" in row:
                    for tmp in row["Answer.textline_1"].strip().split(","):
                        if tmp != "":
                            row_text1.append(tmp.strip())

                steps = read_steps_csv(row, 1)
                res[this_id] = []

                this_q = {
                    "question": q1,
                    "answer": a1,
                    "explanation": e1,
                    "steps": steps,
                    "table_rows": row_table1,
                    "text_rows": row_text1
                }
                res[this_id].append(this_q)



            if this_id in res:
                # append annotator name
                if ann_name != "":
                    res[this_id].append(ann_name)
                else:
                    res[this_id].append(row["Turkle.Username"])

    print("All: ", num_hits)
    print("Questions: ", num_questions)

    res_data = []
    for each_data in data_all:
        if each_data["id"] in res:
            # print(each_data["filename"])
            each_data["eval"] = res[each_data["id"]][0]
            each_data["annotator"] = res[each_data["id"]][-1]

            res_data.append(each_data)

    # for tmp in res:
    #     print(tmp)

    with open(json_out, "w") as f:
        json.dump(res_data, f, indent=4)

    print("All: ", len(res_data))





def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
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
    res = res.replace(" (", " ( ")
    res = res.replace(") ", " ) ")
    
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
    
    if res != "" and res[-1] == ".":
        res = res[:-1] + " ."
        
    res_final = []
    for tmp in res.split(" "):
        if tmp != "":
            res_final.append(tmp)

    return " ".join(res_final)


def check_valid_arg(arg_in, source):
    '''
    check if arg in source
    '''

    for token in source:
        token = token.replace("$", "")
        token_clean = token.replace("%", "")
        if arg_in == token_clean or arg_in == token:
            return True
        
        # if "." not in arg_in and "%" not in arg_in:
        #     this_float = str(float(int(arg_in)))
        #     if this_float == token_clean or this_float == token:
        #         return True
            
        if arg_in + "0" == token_clean or arg_in + "0" == token:
            return True
        if arg_in + ".0" == token_clean or arg_in + ".0" == token:
            return True

    return False


def check_each_step(each_q, all_input):
    '''
    check if the steps is valid
    '''

    # check argument validation
    valid = 1
    # print(each_q)
    if len(each_q["steps"]) > 0:
        prev_res = []
        for ind, each_step in enumerate(each_q["steps"]):
            if "tmp" in each_step["op"]:
                valid = 0
                break

            arg_1 = each_step["arg1"].strip().lower().replace(",", "").replace("$", "")
            if arg_1 == "":
                valid = 0
                break
            if check_valid_arg(arg_1, all_input) == False:
                if ind > 0:
                    if arg_1 not in prev_res:
                        valid = 0
                        break
                else:
                    valid = 0
                    break
            each_step["arg1"] = arg_1

            arg_2 = each_step["arg2"].strip().lower().replace(",", "").replace("$", "")
            if arg_2 == "":
                if not ("max" in each_step["op"] or "min" in each_step["op"] or "average" in each_step["op"] or "sum" in each_step["op"]):
                    valid = 0
                    break

            else:
                if check_valid_arg(arg_2, all_input) == False:
                    if ind > 0:
                        if arg_2 not in prev_res:
                            valid = 0
                            break
                    else:
                        valid = 0
                        break

            each_step["arg2"] = arg_2

            res = each_step["res"].strip().lower().replace(",", "").replace("$", "")
            if res == "":
                valid = 0
                break

            each_step["res"] = res

            prev_res.append(res)

    else:
        valid = 0

    return valid


def check_line_num(each_q, len_text, len_table):
    '''
    check line numbers for each q
    '''

    # if len(each_q["table_rows"]) == 0 and len(each_q["text_rows"]) == 0:
    #     return False
    # for each_row in each_q["table_rows"]:
    #     if not each_row.isdigit():
    #         return False

    #     if int(each_row) > len_table:
    #         return False

    # for each_row in each_q["text_rows"]:
    #     if not each_row.isdigit():
    #         return False

    #     if int(each_row) > len_text:
    #         return False

    return True


def filter_res(json_in, json_out, json_out_invalid):
    '''
    filter invalid examples
    process all inputs
    format:
        {
            "filename":
            "table":
            "pre_text":
            "post_text":
            "annotator":
            "q1":
            {
                "question": ...
                "answer": ...
                "explanation": ...
                "steps": 
                [
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    ...
                ]
                "table_rows": []
                "text_rows": []
            }
        }
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = []
    res_invalid = []
    const_list = ["10", "100", "1000", "1000000", "1000000000",
                  "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for each_data in data_all:
        processed_pre = []
        processed_post = []

        # for checking arguments
        all_input = const_list[:]

        for each_sent in each_data["pre_text"]:
            processed_str = string_process(each_sent)
            processed_pre.append(processed_str)
            all_input.extend(processed_str.split(" "))

        each_data["pre_text"] = processed_pre

        for each_sent in each_data["post_text"]:
            processed_str = string_process(each_sent)
            processed_post.append(processed_str)
            all_input.extend(processed_str.split(" "))

        each_data["post_text"] = processed_post

        # process table: replace () as -
        processed_table = []
        for row in each_data["table"]:
            this_row = []
            for token in row:
                token = string_process_table(token)
                if token != "":
                    # turn () to negative number
                    if token[0] == "(" and token[-1] == ")" and token.replace(".", "")[1:-1].isdigit():
                        this_num = token[1:-1]
                        token = "-" + token[1:-1]
                        token += (" ( " + this_num + " )")
                        
                    # negative number with $ (40.9)
                    if len(token) > 2 and token[0] == "$" and token[2] == "(" and token[-1] == ")" and token.replace(".", "")[3:-1].isdigit():
                        this_num = token[3:-1][:]
                        token = token[:2] + "-" + token[3:-1]
                        token += (" ( " + this_num + " )")

                this_row.append(token)
                all_input.append(token)
                all_input.extend(token.split(" "))
                
            processed_table.append(this_row)
            # all_input.extend(this_row)

        each_data["table"] = processed_table

        # questions
        # print(each_data["q1"])
        # if "q2" in each_data:
        #     print(each_data["q2"])

        # check argument validation
        if check_each_step(each_data["q1"], all_input) == 0:
            res_invalid.append(copy.deepcopy(each_data))
            del each_data["q1"]

        if "q2" in each_data and check_each_step(each_data["q2"], all_input) == 0:
            del each_data["q2"]

        # check line numbers
        len_all_text = len(each_data["pre_text"])+len(each_data["post_text"])
        if "q1" in each_data and check_line_num(each_data["q1"], len_all_text, len(each_data["table"])) == False:
            res_invalid.append(copy.deepcopy(each_data))
            del each_data["q1"]

        if "q2" in each_data and check_line_num(each_data["q2"], len_all_text, len(each_data["table"])) == False:
            del each_data["q2"]

        if "q1" in each_data or "q2" in each_data:
            if "q1" in each_data:
                for item in ["question", "answer", "explanation"]:
                    each_data["q1"][item] = string_process(
                        each_data["q1"][item])

            if "q2" in each_data:
                for item in ["question", "answer", "explanation"]:
                    each_data["q2"][item] = string_process(
                        each_data["q2"][item])

            res.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    with open(json_out_invalid, "w") as f:
        json.dump(res_invalid, f, indent=4)

    print("############")
    print(json_in)
    print("All: ", len(data_all))
    print("All valid: ", len(res))
    print("\n")

def filter_res_eval(json_in, json_out, json_out_invalid):
    '''
    filter invalid examples
    process all inputs
    format:
        {
            "filename":
            "table":
            "pre_text":
            "post_text":
            "annotator":
            "q1":
            {
                "question": ...
                "answer": ...
                "explanation": ...
                "steps": 
                [
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    {"op": ..., "arg1": ..., "arg2": ..., "res": ...}
                    ...
                ]
                "table_rows": []
                "text_rows": []
            }
        }
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = []
    res_invalid = []
    const_list = ["10", "100", "1000", "1000000", "1000000000",
                  "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for each_data in data_all:
        processed_pre = []
        processed_post = []

        # for checking arguments
        all_input = const_list[:]

        for each_sent in each_data["pre_text"]:
            processed_str = string_process(each_sent)
            processed_pre.append(processed_str)
            all_input.extend(processed_str.split(" "))

        each_data["pre_text"] = processed_pre

        for each_sent in each_data["post_text"]:
            processed_str = string_process(each_sent)
            processed_post.append(processed_str)
            all_input.extend(processed_str.split(" "))

        each_data["post_text"] = processed_post

        # process table: replace () as -
        processed_table = []
        for row in each_data["table"]:
            this_row = []
            for token in row:
                token = string_process_table(token)
                if token != "":
                    # turn () to negative number
                    if token[0] == "(" and token[-1] == ")" and token.replace(".", "")[1:-1].isdigit():
                        this_num = token[1:-1]
                        token = "-" + token[1:-1]
                        token += (" ( " + this_num + " )")
                        
                    # negative number with $ (40.9)
                    if len(token) > 2 and token[0] == "$" and token[2] == "(" and token[-1] == ")" and token.replace(".", "")[3:-1].isdigit():
                        this_num = token[3:-1][:]
                        token = token[:2] + "-" + token[3:-1]
                        token += (" ( " + this_num + " )")

                this_row.append(token)
                all_input.append(token)
                all_input.extend(token.split(" "))
                
            processed_table.append(this_row)
            # all_input.extend(this_row)

        each_data["table"] = processed_table

        # questions
        # print(each_data["q1"])
        # if "q2" in each_data:
        #     print(each_data["q2"])

        # check argument validation
        if check_each_step(each_data["eval"], all_input) == 1:
            res.append(each_data)

    with open(json_out, "w") as f:
        json.dump(res, f, indent=4)

    # with open(json_out_invalid, "w") as f:
    #     json.dump(res_invalid, f, indent=4)

    print("############")
    print(json_in)
    print("All: ", len(data_all))
    print("All valid: ", len(res))
    print("\n")


def get_year(folder):

    min_year = 3000
    max_year = 0

    for r, d, fs in os.walk(folder):
        print(d)
        for tmp in d:
            if tmp.isdigit():
                if int(tmp) < min_year:
                    min_year = int(tmp)
                if int(tmp) > max_year:
                    max_year = int(tmp)

    print(min_year)
    print(max_year)


def expert_to_csv_human_eval(json_in, csv_out, start, end):
    '''
    turn to csv on amt
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    count = 0
    with open(csv_out, "w") as f_out:
        fieldnames = ['id', 'table', 'pre_text', 'post_text', 'question']
        writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for data in data_all[start: end]:
            this_id = data["id"]
            table = data["table"]
            pre_text = data["pre_text"]
            post_text = data["post_text"]
            question = data["qa"]["question"]

            amt_table = "<table class='wikitable'>"
            i = 1
            for row in table:
                amt_table += "<tr>"
                amt_table += ("<td>" + str(i) + "</td>")
                i += 1
                for ele in row:
                    amt_table += ("<td>" + ele.strip() + "</td>")
                amt_table += "</tr>"
            amt_table += "</table>"

            # amt_pre_text = " ".join(pre_text)
            # amt_post_text = " ".join(post_text)

            amt_pre_text = ""
            amt_post_text = ""

            ind = 1
            for each_sent in pre_text:
                amt_pre_text += ("<strong>[" + str(ind) +
                                 "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            for each_sent in post_text:
                amt_post_text += ("<strong>[" + str(ind) +
                                  "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            writer.writerow({"id": this_id, "table": amt_table,
                             "pre_text": amt_pre_text, "post_text": amt_post_text, "question": question})
            count += 1

    print(count)


def check_to_csv_human_eval(json_in, csv_out):
    '''
    turn to csv on amt
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    count = 0
    with open(csv_out, "w") as f_out:
        fieldnames = ['id', 'table', 'pre_text', 'post_text', 'question', 'gold_prog', 'eval_prog', 'gold_res']
        writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for data in data_all:
            this_id = data["id"]
            table = data["table"]
            pre_text = data["pre_text"]
            post_text = data["post_text"]
            question = data["qa"]["question"]
            gold_prog = data["qa"]["program"]
            eval_prog = data["eval"]["steps"]
            gold_res = data["qa"]["exe_ans"]

            amt_table = "<table class='wikitable'>"
            i = 1
            for row in table:
                amt_table += "<tr>"
                amt_table += ("<td>" + str(i) + "</td>")
                i += 1
                for ele in row:
                    amt_table += ("<td>" + ele.strip() + "</td>")
                amt_table += "</tr>"
            amt_table += "</table>"

            # amt_pre_text = " ".join(pre_text)
            # amt_post_text = " ".join(post_text)

            amt_pre_text = ""
            amt_post_text = ""

            ind = 1
            for each_sent in pre_text:
                amt_pre_text += ("<strong>[" + str(ind) +
                                 "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            for each_sent in post_text:
                amt_post_text += ("<strong>[" + str(ind) +
                                  "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            this_eval = json.dumps(eval_prog, indent=4)

            writer.writerow({"id": this_id, "table": amt_table,
                             "pre_text": amt_pre_text, "post_text": amt_post_text, "question": question, 'gold_prog': gold_prog,
                             'eval_prog': this_eval, 'gold_res': gold_res})
            count += 1

    print(count)



def get_eval_check(csv_in):
    '''
    get eval check result
    '''

    prog_correct = 0
    exe_correct = 0
    all_data = 0

    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            this_exe = row["Answer.res-exe"]
            this_prog = row["Answer.res-prog"]

            if this_exe == "correct":
                exe_correct += 1

            if this_prog == "correct":
                prog_correct += 1

            if this_exe != "n/a" and this_prog != "n/a":
                all_data += 1

    print("All: ", all_data)
    print("Exe acc: ", float(exe_correct) / all_data)
    print("Prog acc: ", float(prog_correct) / all_data)



def human_eval_agreement(csv_1, csv_2):
    '''
    human evaluation agreement
    '''

    res_1 = {}
    res_2 = {}

    # 0: correct, 1: wrong
    fless_mat_exe = []
    fless_mat_prog = []

    with open(csv_1) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            this_exe = row["Answer.res-exe"]
            this_prog = row["Answer.res-prog"]

            this_id = row["Input.id"]

            assert this_id not in res_1
            res_1[this_id] = [this_exe, this_prog]


    with open(csv_2) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            this_exe = row["Answer.res-exe"]
            this_prog = row["Answer.res-prog"]

            this_id = row["Input.id"]

            assert this_id not in res_2
            res_2[this_id] = [this_exe, this_prog]


    num = 0
    exe_eq = 0
    prog_eq = 0
    for tmp in res_1:
        if tmp in res_2:
            if res_1[tmp][0] != "n/a" and res_2[tmp][0] != "n/a":
                num += 1

                if res_1[tmp][0] == res_2[tmp][0]:
                    exe_eq += 1

                if res_1[tmp][1] == res_2[tmp][1]:
                    prog_eq += 1


                this_exe = [0, 0]
                this_prog = [0, 0]
                if res_1[tmp][0] == "correct":
                    this_exe[0] += 1
                else:
                    this_exe[1] += 1

                if res_2[tmp][0] == "correct":
                    this_exe[0] += 1
                else:
                    this_exe[1] += 1

                # prog
                if res_1[tmp][1] == "correct":
                    this_prog[0] += 1
                else:
                    this_prog[1] += 1

                if res_2[tmp][1] == "correct":
                    this_prog[0] += 1
                else:
                    this_prog[1] += 1


                # exe
                fless_mat_exe.append(this_exe)
                fless_mat_prog.append(this_prog)



                
    kappa_exe = fleiss_kappa(fless_mat_exe)
    kappa_prog = fleiss_kappa(fless_mat_prog)

    print("exe agree: ", float(exe_eq) / num)
    print("prog agree: ", float(prog_eq) / num)
    print("Kappa exe: ", kappa_exe)
    print("Kappa prog: ", kappa_prog)



def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """

    M = np.array(M)
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / \
        (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa



def check_to_csv_laymen_eval(json_in, csv_in, csv_out):
    '''
    turn to csv on amt
    '''

    with open(json_in) as f:
        data_all = json.load(f)

    res = {}
    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            this_id = row["Input.id"]
            assert this_id not in res

            this_ans = row["Answer.answer"]
            this_step = row["Answer.step"]

            res[this_id] = [this_ans, this_step]

    count = 0
    with open(csv_out, "w") as f_out:
        fieldnames = ['id', 'table', 'pre_text', 'post_text', 'question', 'gold_prog', 'eval_prog', 'gold_res']
        writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for data in data_all:
            this_id = data["id"]
            table = data["table"]
            pre_text = data["pre_text"]
            post_text = data["post_text"]
            question = data["qa"]["question"]
            gold_prog = data["qa"]["program"]
            # eval_prog = data["eval"]["steps"]
            gold_res = data["qa"]["exe_ans"]

            eval_ans = res[this_id][0]
            eval_step = res[this_id][1]

            this_eval = eval_step + " Ans: " + eval_ans

            amt_table = "<table class='wikitable'>"
            i = 1
            for row in table:
                amt_table += "<tr>"
                amt_table += ("<td>" + str(i) + "</td>")
                i += 1
                for ele in row:
                    amt_table += ("<td>" + ele.strip() + "</td>")
                amt_table += "</tr>"
            amt_table += "</table>"

            # amt_pre_text = " ".join(pre_text)
            # amt_post_text = " ".join(post_text)

            amt_pre_text = ""
            amt_post_text = ""

            ind = 1
            for each_sent in pre_text:
                amt_pre_text += ("<strong>[" + str(ind) +
                                 "]: </strong>" + each_sent + "<br><br>")
                ind += 1

            for each_sent in post_text:
                amt_post_text += ("<strong>[" + str(ind) +
                                  "]: </strong>" + each_sent + "<br><br>")
                ind += 1


            writer.writerow({"id": this_id, "table": amt_table,
                             "pre_text": amt_pre_text, "post_text": amt_post_text, "question": question, 'gold_prog': gold_prog,
                             'eval_prog': this_eval, 'gold_res': gold_res})
            count += 1

    print(count)



def get_eval_time(csv_in):
    '''
    working time for laymen
    '''

    all_time = 0.0
    all_data = 0

    with open(csv_in) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            all_time += float(row["WorkTimeInSeconds"])
            all_data += 1


    print("All: ", all_data)
    print("Avg time: ", all_time / all_data)





root = "/mnt/george_bhd/zhiyuchen/finQA/"
data_folder = root + "data/"
our_data = root + "dataset/"

data_fintabnet = data_folder + "fintabnet/fintabnet_ori/"
data_fintabnet_processed = data_folder + "fintabnet/processed/"
pdf_dir = data_fintabnet + "pdf/"
our_data_fintabnet_amt = our_data + "amt_ori/"
our_data_fintabnet_json_ori = our_data + "json_ori/"
our_data_fintabnet_json_processed = our_data + "json_processed/"
our_data_fintabnet_amt_upload = our_data + "amt_upload/"
human_eval_folder = our_data + "human_eval/"


# train set
table_json_file_ori = data_fintabnet + "FinTabNet_1.0.0_table_train.jsonl"

table_single_perpage = data_fintabnet_processed + "train_single_table.jsonl"
table_rowlendiff_1 = data_fintabnet_processed + "train_rowlendiff_1.jsonl"
table_rowlendiff_1_makeup = data_fintabnet_processed + \
    "train_rowlendiff_1_makeup.jsonl"

processed_original = data_fintabnet_processed + "processed_original.jsonl"
processed = data_fintabnet_processed + "processed.jsonl"
processed_old = data_fintabnet_processed + "processed_old.jsonl"

processed_table10 = data_fintabnet_processed + "processed_table10.jsonl"

sample_csv = our_data_fintabnet_amt_upload + "test_csv_100.csv"


# # test set
# table_json_file_ori = data_fintabnet + "FinTabNet_1.0.0_table_test.jsonl"

# table_single_perpage = data_fintabnet_processed + "test_single_table.jsonl"
# table_rowlendiff_1 = data_fintabnet_processed + "test_rowlendiff_1.jsonl"
# table_rowlendiff_1_makeup = data_fintabnet_processed + "test_rowlendiff_1_makeup.jsonl"

# processed_original = data_fintabnet_processed + "test_processed_original.jsonl"
# processed_test = data_fintabnet_processed + "test_processed.jsonl"
# sample_csv = our_data_fintabnet_amt_upload + "test_csv_100.csv"


# test_multiple_table(table_single_perpage)

# filter_single_table(table_json_file_ori, table_single_perpage)

# read_table(table_single_perpage, table_rowlendiff_1)

# extract_text(table_rowlendiff_1, pdf_dir, processed_original)

# convert2final_old(processed_original, processed_old, 300)

# convert2final(processed_original, processed_table10, 300)

# expert_to_csv(processed, sample_csv, 0, 100)


# sample_batch_csv = our_data_fintabnet_amt + "test_batch_1.csv"
# sample_batch = our_data_fintabnet_json_ori + "test_batch.json"

# get_res(sample_batch_csv, processed, sample_batch)


# # for each batch
# batch_name = our_data_fintabnet_amt_upload + "ethan_week3.csv"
# expert_to_csv(processed, batch_name, 2440, 2500)

# batch_name = our_data_fintabnet_amt_upload + "qsopjani_week6.csv"
# expert_to_csv(processed_table10, batch_name, 260, 350)

# ["iana_5", "dylan_4", "reema_2", "mark_4", "tony_4", "qsopjani_3"]

# processed = processed_old

# for file_name in ["iana_1", "iana_2", "iana_3", "iana_4", 
#                   "dylan_1", "dylan_2", "dylan_3", 
#                   "reema_1", 
#                   "mark_1", "mark_2", "mark_3", 
#                   "tony_1", "tony_2", "tony_3", 
#                   "qsopjani_1", "qsopjani_2", 
#                   ]:
    
# for file_name in ["iana_5", "dylan_4", "reema_2", "mark_4", "tony_4", "qsopjani_3"]:
    
#     sample_batch_csv = our_data_fintabnet_amt + file_name + ".csv"
#     sample_batch = our_data_fintabnet_json_ori + file_name + ".json"

#     if "iana" in file_name:
#         get_res(sample_batch_csv, processed, sample_batch, ann_name="iana")
#     elif "dylan" in file_name:
#         get_res(sample_batch_csv, processed, sample_batch, ann_name="dylan")
#     else:
#         get_res(sample_batch_csv, processed, sample_batch, ann_name="")

#     # filter res
#     json_ori = our_data_fintabnet_json_ori + file_name + ".json"
#     json_processed = our_data_fintabnet_json_processed + file_name + ".json"
#     test_invalid = our_data_fintabnet_json_processed + file_name + "_invalid.json"
#     filter_res(json_ori, json_processed, test_invalid)



# # processed = processed_table10

# sample_batch_csv = our_data_fintabnet_amt + "victor_7.csv"
# sample_batch = our_data_fintabnet_json_ori + "victor_7.json"

# get_res(sample_batch_csv, processed, sample_batch, ann_name="")


# our_data_fintabnet_json_processed = our_data + "json_processed_last/"
# # filter res
# json_ori = our_data_fintabnet_json_ori + "victor_7.json"
# json_processed = our_data_fintabnet_json_processed + "victor_7.json"
# test_invalid = our_data_fintabnet_json_processed + "victor_7_invalid.json"
# filter_res(json_ori, json_processed, test_invalid)


# user = "craig"
# for i in range(1, 3):

#     print("Ind: " + str(i))

#     sample_batch_csv = our_data_fintabnet_amt + user + "_" + str(i) + ".csv"
#     sample_batch = our_data_fintabnet_json_ori + user + "_" + str(i) + ".json"

#     get_res(sample_batch_csv, processed, sample_batch, ann_name="")


#     # filter res
#     json_ori = our_data_fintabnet_json_ori + user + "_" + str(i) + ".json"
#     json_processed = our_data_fintabnet_json_processed + user + "_" + str(i) + ".json"
#     test_invalid = our_data_fintabnet_json_processed + user + "_" + str(i) + "_invalid.json"
#     filter_res(json_ori, json_processed, test_invalid)



# get_year(pdf_dir)

# train = our_data + "train.json"
# valid = our_data + "dev.json"
# test = our_data + "test.json"

# human_eval_file = our_data_fintabnet_amt_upload + "human_eval_200.csv"

# expert_to_csv_human_eval(test, human_eval_file, 0, 200)


#### human eval

# human_eval_folder = our_data + "human_eval/"
# human_eval_ori = our_data + "test_human_eval.json"

# sample_batch_csv = human_eval_folder + "human_eval_2.csv"
# sample_batch = human_eval_folder + "human_eval_2_tmp.json"

# get_res_eval(sample_batch_csv, human_eval_ori, sample_batch, ann_name="")


# # filter res
# json_ori = human_eval_folder + "human_eval_2_tmp.json"
# json_processed = human_eval_folder + "human_eval_2.json"
# test_invalid = human_eval_folder + "human_eval_2_invalid.json"

# filter_res_eval(json_ori, json_processed, test_invalid)


# human_eval_check = our_data_fintabnet_amt_upload = our_data + "amt_upload/human_eval_check_200_2.csv"
# check_to_csv_human_eval(json_ori, human_eval_check)

# human_eval_check_res = human_eval_folder + "human_eval_check.csv"
# get_eval_check(human_eval_check_res)

# human_eval_check_res = human_eval_folder + "human_eval_check_2.csv"
# get_eval_check(human_eval_check_res)

# csv_1 = human_eval_folder + "human_eval_laymen_check_res.csv"
# csv_2 = human_eval_folder + "human_eval_laymen_check_res_1.csv"
# human_eval_agreement(csv_1, csv_2)


csv_in = human_eval_folder + "human_eval_laymen_1.csv"
# csv_out = human_eval_folder + "human_eval_laymen_check_1.csv"
# check_to_csv_laymen_eval(json_ori, csv_in, csv_out)



# human_eval_check_res = human_eval_folder + "human_eval_laymen_check_res_1.csv"
# get_eval_check(human_eval_check_res)


get_eval_time(csv_in)