import csv
import re, sys
import json
csv.field_size_limit(sys.maxsize)






def read_file_id(file_in):
	'''
	read original file
	'''

	num = 0
	header_map = {}
	res = {}

	with open(file_in, encoding = "ISO-8859-1") as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			# print(line)
			num += 1
			if num == 1:
				# read header
				for ind, name in enumerate(line):
					header_map[ind] = name
				print(header_map)

			else:
				this_res = {}
				for ind, name in enumerate(line):
					if ind in header_map:
						this_res[header_map[ind]] = name

				if line[0] not in res:
					res[line[0]] = []

				res[line[0]].append(this_res)

				# print(this_res)


	return res
	# print(num)


def read_file_tag(file_in):
	'''
	read original file
	'''

	num = 0
	header_map = {}
	res = {}

	with open(file_in, encoding = "ISO-8859-1") as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			# print(line)
			num += 1
			if num == 1:
				# read header
				for ind, name in enumerate(line):
					header_map[ind] = name
				print(header_map)

			else:
				this_res = {}
				for ind, name in enumerate(line):
					if ind in header_map:
						this_res[header_map[ind]] = name

				if line[1] not in res:
					res[line[1]] = []

				res[line[1]].append(this_res)

				# print(this_res)


	return res



def find_common(num_file, txt_file):
	'''
	find common adsh+tag
	'''
	data_num = read_file(num_file)
	data_txt = read_file(txt_file)

	for each_id in data_txt:
		this_batch_txt = data_txt[each_id]
		this_batch_num = data_num[each_id]

		for each_data in this_batch_txt:
			this_tag = each_data["tag"]
			# print(this_tag)
			# print("\n#################")
			# print(each_data)

			for each_num in this_batch_num:
				if each_num["tag"] == this_tag:
					print("===================")
					print(each_data)
					print(each_num)




# def test_amt(file_in, file_out):


def filter_num(this_text, thresh):
	'''
	whther a text contains over thresh
	'''

	# over 5 numericals
	pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"
	num_numericals = 0
	for token in this_text.strip().split(" "):
		val_pat = re.findall(pat_num, token)
		if len(val_pat) != 0:
			num_numericals += 1
			if num_numericals >= thresh:
				break


	if "follow" in this_text:
		return False

	if num_numericals >= thresh:
		return True
	else:
		return False





def filter_text(file_in, file_sub, file_out):
	'''
	filter text of txt.tsv
	'''

	pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

	res = {}
	data_txt = read_file(file_in)
	data_sub = read_file(file_sub)
	all_written = 0


	for each_id in data_txt:
		this_batch_txt = data_txt[each_id]
		for each_data in this_batch_txt:
			this_text = each_data["value"]

			# over 5 numericals
			num_numericals = 0
			for token in this_text.strip().split(" "):
				val_pat = re.findall(pat_num, token)
				if len(val_pat) != 0:
					num_numericals += 1
					if num_numericals >= 5:
						break

			if "following" in this_text:
				continue
			if "follow" in this_text:
				continue

			if num_numericals >= 5:
				if each_id not in res:
					res[each_id] = []

				try:
					this_text.encode(encoding='utf-8').decode('ascii')
				except UnicodeDecodeError:
					continue
				this_company = data_sub[each_id][0]["name"]
				each_data["value"] = this_company + " " + this_text
				each_data["id"] = all_written
				# each_data["value"] = each_data["value"].lower()
				res[each_id].append(each_data)
				all_written += 1

	print(all_written)

	with open(file_out, "w") as f:
		json.dump(res, f, indent=4)


def amt_test(file_in, file_out, count):
	'''
	test batch on amt
	'''

	with open(file_in) as f:
		data = json.load(f)

	num = 0

	with open(file_out, "w") as f_out:
		fieldnames = ['id', 'text']
		writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
		writer.writeheader()
		for each_id in data:
			this_batch_data = data[each_id]
			for each_data in this_batch_data:
				row = {"id":each_data["id"], "text":each_data["value"].encode('utf-8')}
				writer.writerow(row)
				num += 1
			if num > count:
				break


	print (num)


def read_tag_map(file_in):
	'''
	read tag maps
	'''
	num = 0
	header_map = {}
	res = {}

	with open(file_in, encoding = "ISO-8859-1") as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			# print(line)
			num += 1
			if line[0] not in res:
				# print(line[5])
				# res[line[0]] = line[7].lower().strip(" [Text Block]")
				res[line[0]] = line[7].lower()
				# print(line[0])


	print(len(res))
	return res


def is_num_related(row_tag, txt):
	'''
	if the row is related to a txt
	'''

	row_tag = row_tag.lower()
	txt = txt.lower()

	if row_tag in txt:
		return True

	row_tag_list = row_tag.split()
	if len(row_tag.split()) > 2:
		for cut in range(2, len(row_tag.split())):
			if " ".join(row_tag_list[0: cut]) in txt:
				return True
		for cut in range(2, len(row_tag.split())):
			if " ".join(row_tag_list[-cut:]) in txt:
				return True

	return False





def extract_table_text(text_json_in, num_in, tag_map, json_out):
	'''
	extract text and table
	'''

	with open(text_json_in) as f:
		data = json.load(f)

	data_num = read_file(num_in)
	res = []


	for each_adsh in data:
		this_batch = data[each_adsh]
		for each_data in this_batch:
			this_id = each_data["adsh"]
			this_text = each_data["value"]

			this_num = data_num[this_id]

			res_rows = []
			for each_num in this_num:
				if each_num["tag"] in tag_map:
					each_tag = tag_map[each_num["tag"]]
					if is_num_related(each_tag, this_text):
						res_rows.append(each_num)
						# enough samples
						if len(res_rows) > 10:
							break


			if len(res_rows) > 5:
				each_data["table"] = res_rows
				res.append(each_data)


	print(len(res))

	with open(json_out, "w") as f:
		json.dump(res, f, indent=4)




def table_txt_amt(json_in, csv_out, count):
	'''
	turn table txt samples to amt
	'''

	with open(json_in) as f:
		data = json.load(f)

	num = 0

	with open(csv_out, "w") as f_out:
		fieldnames = ['id', 'text', 'table']
		writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
		writer.writeheader()
		for each_data in data:

			table_string = ""

			for each_row in each_data["table"]:
				table_string += ("<strong>" + each_row["tag"] + "</strong>" + "&nbsp end date: " + each_row["ddate"] + "&nbsp quarters: " + each_row["qtrs"] + "&nbsp value: " + each_row["value"] + "<br>")


			row = {"id":each_data["id"], "text":each_data["value"].encode('utf-8'), "table": table_string.encode('utf-8').decode('utf-8')}
			writer.writerow(row)
			num += 1
			if num > count:
				break


	print (num)


def valid_text(text_in):
	'''
	if this text is valid
	'''
	if "follow" in text_in:
		return False
	if "font " in text_in:
		return False
	if "font-" in text_in:
		return False


	return True



def two_para(file_in, file_sub, file_out):
	'''
	associate same tag with two companies
	'''

	data_txt = read_file_tag(file_in)
	data_sub = read_file_id(file_sub)

	res = []
	inc_id = 0

	for each_tag in data_txt:
		this_batch_txt = data_txt[each_tag]

		ind = 0
		while ind < len(this_batch_txt) - 2:
		# for ind, each_data in enumerate(this_batch_txt):
			each_data = this_batch_txt[ind]
			each_data_1 = this_batch_txt[ind+1]
			this_company = each_data["adsh"]
			this_text = each_data["value"]
			company_name = data_sub[this_company][0]["name"]

			if valid_text(this_text):
				if filter_num(this_text, 5):
					# for each_data_1 in this_batch_txt:
					if each_data_1["adsh"] != this_company:
						# print(this_company)
						# print(each_data_1["adsh"])
						# print("tttttttt")
						this_text_1 = each_data_1["value"]
						company_name_1 = data_sub[each_data_1["adsh"]][0]["name"]
						if filter_num(this_text_1, 5):

							this_tag = each_data["tag"]
							text1 = "Company name: " + company_name + ". Financial Statement Text: " + this_text
							text2 = "Company name: " + company_name_1 + ". Financial Statement Text: " + this_text_1

							print("###############")
							print(each_data["tag"])
							print("######### " + text1)
							print("######### " + text2)

							print("\n")

							res.append({
								"id": inc_id,
								"tag": this_tag,
								"text1": text1,
								"text2": text2,
								})

			ind += 2

	print(len(res))
	with open(file_out, "w") as f:
		json.dump(res, f, indent=4)


def two_compare_amt(json_in, csv_out, count):
	'''
	turn table txt samples to amt
	'''

	with open(json_in) as f:
		data = json.load(f)

	num = 0

	with open(csv_out, "w") as f_out:
		fieldnames = ['id', 'text1', 'text2']
		writer = csv.DictWriter(f_out, delimiter=',', fieldnames=fieldnames)
		writer.writeheader()
		for each_data in data:

			row = {"id":each_data["id"], "text1":each_data["text1"].encode('utf-8'), "text2": each_data["text2"].encode('utf-8')}
			writer.writerow(row)
			num += 1
			if num > count:
				break


	print (num)



def test_pdf(pdf_in):
	'''
	read pdf content
	'''
	from tika import parser

	raw = parser.from_file(pdf_in)
	# print(raw['content'])

	print("start")

	for line in raw['content'].split("\n"):
		print(line)



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




def get_year(dir):

    for r, d, fs in os.walk(folder):
        for f in fs:
			print(f)




root = "/scratch/home/zhiyu/finQA/"
data_folder = root + "data/"

gov_data = data_folder + "gov_data/"
ibm_data = data_folder + "page_65.pdf"



test_data = gov_data + "2009q3_notes/"
txt_in = test_data + "txt.tsv"
num_in = test_data + "num.tsv"
sub_in = test_data + "sub.tsv"
tag_in = test_data + "tag.tsv"

txt_5_num = test_data + "txt_5_num.json"
txt_5_amt = test_data + "txt_5_num.csv"

txt_table_out = test_data + "table_txt_5_row.json"
txt_table_csv = test_data + "table_txt_5_row.csv"

two_compare_json = test_data + "txt_two_compare.json"
two_compare_csv = test_data + "txt_two_compare.csv"

# read_file(txt_in)

# find_common(num_in, txt_in)

# filter_text(txt_in, sub_in, txt_5_num)


# amt_test(txt_5_num, txt_5_amt, 200)


# tag_map = read_tag_map(tag_in)

# extract_table_text(txt_5_num, num_in, tag_map, txt_table_out)


# table_txt_amt(txt_table_out, txt_table_csv, 50)


# two_para(txt_in, sub_in, two_compare_json)

# two_compare_amt(two_compare_json, two_compare_csv, 100)


# test_pdf(ibm_data)


test = program_tokenization("multiply(add(28, const_1), add(add(28, const_1), const_1))")
print(test)





















