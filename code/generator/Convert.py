import argparse
import collections
import json
import os
import sys
import random


'''
convert retriever results to generator test input
'''


def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


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

# for single sent retrieve


def convert_test(json_in, json_out, topn, max_len):

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + \
            each_data["text_retrieved"]

        sorted_dict = sorted(
            all_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_row_to_text(table[0], table[this_sent_ind])
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # original_order
        sorted_dict_table = sorted(
            all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(
            all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        each_data["qa"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(len(data))


def convert_train(json_in, json_out, topn, max_len):

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        gold_inds = each_data["qa"]["gold_inds"]

        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + \
            each_data["text_retrieved"]

        false_retrieved = []
        for tmp in all_retrieved:
            if tmp["ind"] not in gold_inds:
                false_retrieved.append(tmp)

        sorted_dict = sorted(
            false_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        for tmp in gold_inds:
            if "table" in tmp:
                all_table_in[tmp] = gold_inds[tmp]
            else:
                all_text_in[tmp] = gold_inds[tmp]

        context = ""
        for tmp in gold_inds:
            context += gold_inds[tmp]

        acc_len = len(context.split(" "))

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_row_to_text(table[0], table[this_sent_ind])
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # original_order
        sorted_dict_table = sorted(
            all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(
            all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        each_data["qa"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)


def convert_test_infer(json_in, json_out, topn, mode):

    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved_all"]
        text_retrieved = each_data["text_retrieved_all"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table = each_data["table"]

        # all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]
        # sorted_dict = sorted(all_retrieved, key=lambda kv: kv["score"], reverse=True)

        sorted_dict_text = sorted(
            text_retrieved, key=lambda kv: kv["score"], reverse=True)
        sorted_dict_table = sorted(
            table_retrieved, key=lambda kv: kv["score"], reverse=True)

        # print(sorted_dict_table)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        # if mode == "table":
        for tmp in sorted_dict_table[:topn]:
            this_sent_ind = int(tmp["ind"].split("_")[1])
            this_sent = table_row_to_text(table[0], table[this_sent_ind])
            all_table_in[tmp["ind"]] = this_sent

        for tmp in sorted_dict_text[:topn]:
            this_sent_ind = int(tmp["ind"].split("_")[1])
            all_text_in[tmp["ind"]] = all_text[this_sent_ind]

        this_model_input = []

        # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        # this_model_input.extend(sorted_dict)

        # original_order
        sorted_dict_table = sorted(
            all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(
            all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        # for tmp in sorted_dict_text:
        #     if int(tmp[0].split("_")[1]) < len(pre_text):
        #         this_model_input.append(tmp)

        # for tmp in sorted_dict_table:
        #     this_model_input.append(tmp)

        # for tmp in sorted_dict_text:
        #     if int(tmp[0].split("_")[1]) >= len(pre_text):
        #         this_model_input.append(tmp)

        if mode == "table":
            for tmp in sorted_dict_table:
                this_model_input.append(tmp)
        else:
            for tmp in sorted_dict_text:
                this_model_input.append(tmp)

        each_data["qa"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(len(data))


if __name__ == '__main__':

    root = "your_root_path"

    ### convert the results from the retriever. 
    ### json_in is the inference result file generated from the retriever. Edit the paths here. 

    json_in = root + "outputs/inference_only_20210503002312_retriever-bert-base-train/results/test/predictions.json"
    json_out = root + "FinQA/dataset/train_retrieve.json"
    convert_train(json_in, json_out, topn=3, max_len=290)

    json_in = root + "outputs/inference_only_20210503002312_retriever-bert-base-dev/results/test/predictions.json"
    json_out = root + "FinQA/dataset/dev_retrieve.json"
    convert_train(json_in, json_out, topn=3, max_len=290)

    json_in = root + "outputs/inference_only_20210503012815_retriever-bert-base-test/results/test/predictions.json"
    json_out = root + "FinQA/dataset/test_retrieve.json"
    convert_test(json_in, json_out, topn=3, max_len=290)

    # json_in = root + "outputs/inference_only_20210505220955_retriever-bert-base-7k-test-new/results/test/predictions.json"
    # json_out = root + "FinQA/dataset/test_retrieve_7k_text_only.json"
    # convert_test_infer(json_in, json_out, topn=3, mode="text")
