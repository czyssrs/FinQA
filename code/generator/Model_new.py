import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from config import parameters as conf

if conf.pretrained_model == "bert":
    from transformers import BertModel
elif conf.pretrained_model == "roberta":
    from transformers import RobertaModel
elif conf.pretrained_model == "finbert":
    from transformers import BertModel
elif conf.pretrained_model == "longformer":
    from transformers import LongformerModel


class Bert_model(nn.Module):

    def __init__(self, num_decoder_layers, hidden_size, dropout_rate, input_length,
                 program_length, op_list, const_list):

        super(Bert_model, self).__init__()

        self.op_list_size = len(op_list)
        self.const_list_size = len(const_list)
        self.reserved_token_size = self.op_list_size + self.const_list_size
        self.program_length = program_length
        self.hidden_size = hidden_size
        self.const_list = const_list
        self.op_list = op_list
        self.input_length = input_length

        self.reserved_ind = nn.Parameter(torch.arange(
            0, self.reserved_token_size), requires_grad=False)
        self.reserved_go = nn.Parameter(torch.arange(op_list.index(
            'GO'), op_list.index('GO') + 1), requires_grad=False)

        self.reserved_para = nn.Parameter(torch.arange(op_list.index(
            ')'), op_list.index(')') + 1), requires_grad=False)

        # masking for decoidng for test time
        op_ones = nn.Parameter(torch.ones(
            self.op_list_size), requires_grad=False)
        op_zeros = nn.Parameter(torch.zeros(
            self.op_list_size), requires_grad=False)
        other_ones = nn.Parameter(torch.ones(
            input_length + self.const_list_size), requires_grad=False)
        other_zeros = nn.Parameter(torch.zeros(
            input_length + self.const_list_size), requires_grad=False)
        self.op_only_mask = nn.Parameter(
            torch.cat((op_ones, other_zeros), 0), requires_grad=False)
        self.seq_only_mask = nn.Parameter(
            torch.cat((op_zeros, other_ones), 0), requires_grad=False)

        # for ")"
        para_before_ones = nn.Parameter(torch.ones(
            op_list.index(')')), requires_grad=False)
        para_after_ones = nn.Parameter(torch.ones(
            input_length + self.reserved_token_size - op_list.index(')') - 1), requires_grad=False)
        para_zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.para_mask = nn.Parameter(torch.cat(
            (para_before_ones, para_zero, para_after_ones), 0), requires_grad=False)

        # for step embedding
        # self.step_masks = []
        all_tmp_list = self.op_list + self.const_list
        self.step_masks = nn.Parameter(torch.zeros(
            conf.max_step_ind, input_length + self.reserved_token_size), requires_grad=False)
        for i in range(conf.max_step_ind):
            this_step_mask_ind = all_tmp_list.index("#" + str(i))
            self.step_masks[i, this_step_mask_ind] = 1.0

        # self.step_mask_eye = torch.eye(conf.max_step_ind)

        if conf.pretrained_model == "bert":
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)
        elif conf.pretrained_model == "roberta":
            self.bert = RobertaModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)
        elif conf.pretrained_model == "finbert":
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)
        elif conf.pretrained_model == "longformer":
            self.bert = LongformerModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)

        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)

        self.seq_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.seq_dropout = nn.Dropout(dropout_rate)

        self.reserved_token_embedding = nn.Embedding(
            self.reserved_token_size, hidden_size)

        # attentions
        self.decoder_history_attn_prj = nn.Linear(
            hidden_size, hidden_size, bias=True)
        self.decoder_history_attn_dropout = nn.Dropout(dropout_rate)

        self.question_attn_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.question_attn_dropout = nn.Dropout(dropout_rate)

        self.question_summary_attn_prj = nn.Linear(
            hidden_size, hidden_size, bias=True)
        self.question_summary_attn_dropout = nn.Dropout(dropout_rate)

        if conf.sep_attention:
            self.input_embeddings_prj = nn.Linear(
                hidden_size*3, hidden_size, bias=True)
        else:
            self.input_embeddings_prj = nn.Linear(
                hidden_size*2, hidden_size, bias=True)
        self.input_embeddings_layernorm = nn.LayerNorm([1, hidden_size])

        self.option_embeddings_prj = nn.Linear(
            hidden_size*2, hidden_size, bias=True)

        # decoder lstm
        self.rnn = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                 num_layers=conf.num_decoder_layers, batch_first=True)

        # step vector
        self.decoder_step_proj = nn.Linear(
            3*hidden_size, hidden_size, bias=True)
        self.decoder_step_proj_dropout = nn.Dropout(dropout_rate)

        self.step_mix_proj = nn.Linear(
            hidden_size*2, hidden_size, bias=True)

    def forward(self, is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, device):

        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        bert_sequence_output = bert_outputs.last_hidden_state
        bert_pooled_output = bert_sequence_output[:, 0, :]
        batch_size, seq_length, bert_dim = list(bert_sequence_output.size())

        split_program_ids = torch.split(program_ids, 1, dim=1)
        # print(self.program_length)
        # print(program_ids.size())
        # print(split_program_ids[0].size())

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        option_size = self.reserved_token_size + seq_length

        sequence_output = self.seq_prj(bert_sequence_output)
        sequence_output = self.seq_dropout(sequence_output)

        op_embeddings = self.reserved_token_embedding(self.reserved_ind)
        op_embeddings = op_embeddings.repeat(batch_size, 1, 1)

        logits = []

        init_decoder_output = self.reserved_token_embedding(self.reserved_go)
        decoder_output = init_decoder_output.repeat(batch_size, 1, 1)

        # [batch, op + seq len, hidden]
        initial_option_embeddings = torch.cat(
            [op_embeddings, sequence_output], dim=1)

        if conf.sep_attention:
            decoder_history = decoder_output
        else:
            decoder_history = torch.unsqueeze(pooled_output, dim=-1)

        decoder_state_h = torch.zeros(
            1, batch_size, self.hidden_size, device=device)
        decoder_state_c = torch.zeros(
            1, batch_size, self.hidden_size, device=device)

        float_input_mask = input_mask.float()
        float_input_mask = torch.unsqueeze(float_input_mask, dim=-1)

        this_step_new_op_emb = initial_option_embeddings

        for cur_step in range(self.program_length):

            # decoder history att
            decoder_history_attn_vec = self.decoder_history_attn_prj(
                decoder_output)
            decoder_history_attn_vec = self.decoder_history_attn_dropout(
                decoder_history_attn_vec)

            decoder_history_attn_w = torch.matmul(
                decoder_history, torch.transpose(decoder_history_attn_vec, 1, 2))
            decoder_history_attn_w = F.softmax(decoder_history_attn_w, dim=1)

            decoder_history_ctx_embeddings = torch.matmul(
                torch.transpose(decoder_history_attn_w, 1, 2), decoder_history)

            if conf.sep_attention:
                # input seq att
                question_attn_vec = self.question_attn_prj(decoder_output)
                question_attn_vec = self.question_attn_dropout(
                    question_attn_vec)

                question_attn_w = torch.matmul(
                    sequence_output, torch.transpose(question_attn_vec, 1, 2))
                question_attn_w -= 1e6 * (1 - float_input_mask)
                question_attn_w = F.softmax(question_attn_w, dim=1)

                question_ctx_embeddings = torch.matmul(
                    torch.transpose(question_attn_w, 1, 2), sequence_output)

            # another input seq att
            question_summary_vec = self.question_summary_attn_prj(
                decoder_output)
            question_summary_vec = self.question_summary_attn_dropout(
                question_summary_vec)

            question_summary_w = torch.matmul(
                sequence_output, torch.transpose(question_summary_vec, 1, 2))
            question_summary_w -= 1e6 * (1 - float_input_mask)
            question_summary_w = F.softmax(question_summary_w, dim=1)

            question_summary_embeddings = torch.matmul(
                torch.transpose(question_summary_w, 1, 2), sequence_output)

            if conf.sep_attention:
                concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                     question_ctx_embeddings,
                                                     decoder_output], dim=-1)
            else:
                concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                     decoder_output], dim=-1)

            input_embeddings = self.input_embeddings_prj(
                concat_input_embeddings)

            if conf.layer_norm:
                input_embeddings = self.input_embeddings_layernorm(
                    input_embeddings)

            question_option_vec = this_step_new_op_emb * question_summary_embeddings
            option_embeddings = torch.cat(
                [this_step_new_op_emb, question_option_vec], dim=-1)

            option_embeddings = self.option_embeddings_prj(option_embeddings)
            option_logits = torch.matmul(
                option_embeddings, torch.transpose(input_embeddings, 1, 2))
            option_logits = torch.squeeze(
                option_logits, dim=2)  # [batch, op + seq_len]
            option_logits -= 1e6 * (1 - option_mask)
            logits.append(option_logits)

            if is_training:
                program_index = torch.unsqueeze(
                    split_program_ids[cur_step], dim=1)
            else:
                # constrain decoding
                if cur_step % 4 == 0 or (cur_step + 1) % 4 == 0:
                    # op round
                    option_logits -= 1e6 * self.seq_only_mask
                else:
                    # number round
                    option_logits -= 1e6 * self.op_only_mask

                if (cur_step + 1) % 4 == 0:
                    # ")" round
                    option_logits -= 1e6 * self.para_mask
                    # print(program_index)

                program_index = torch.argmax(
                    option_logits, axis=-1, keepdim=True)

                program_index = torch.unsqueeze(
                    program_index, dim=1
                )

            if (cur_step + 1) % 4 == 0:

                # update op embeddings
                this_step_index = cur_step // 4
                this_step_list_index = (
                    self.op_list + self.const_list).index("#" + str(this_step_index))
                this_step_mask = self.step_masks[this_step_index, :]

                decoder_step_vec = self.decoder_step_proj(
                    concat_input_embeddings)
                decoder_step_vec = self.decoder_step_proj_dropout(
                    decoder_step_vec)
                decoder_step_vec = torch.squeeze(decoder_step_vec)

                this_step_new_emb = decoder_step_vec  # [batch, hidden]

                this_step_new_emb = torch.unsqueeze(this_step_new_emb, 1)
                this_step_new_emb = this_step_new_emb.repeat(
                    1, self.reserved_token_size+self.input_length, 1)  # [batch, op seq, hidden]

                this_step_mask = torch.unsqueeze(
                    this_step_mask, 0)  # [1, op seq]
                # print(this_step_mask)

                this_step_mask = torch.unsqueeze(
                    this_step_mask, 2)  # [1, op seq, 1]
                this_step_mask = this_step_mask.repeat(
                    batch_size, 1, self.hidden_size)  # [batch, op seq, hidden]

                this_step_new_op_emb = torch.where(
                    this_step_mask > 0, this_step_new_emb, initial_option_embeddings)

            # print(program_index.size())
            program_index = torch.repeat_interleave(
                program_index, self.hidden_size, dim=2)  # [batch, 1, hidden]

            input_program_embeddings = torch.gather(
                option_embeddings, dim=1, index=program_index)

            decoder_output, (decoder_state_h, decoder_state_c) = self.rnn(
                input_program_embeddings, (decoder_state_h, decoder_state_c))
            decoder_history = torch.cat(
                [decoder_history, input_program_embeddings], dim=1)

        logits = torch.stack(logits, dim=1)
        return logits
