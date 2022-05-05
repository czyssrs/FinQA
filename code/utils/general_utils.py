import argparse
import collections
import json
import os
import re
import string
import sys
import random


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
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()