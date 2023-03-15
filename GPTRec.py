import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from copy import copy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from env_config import *
from metrics import *


def mask_all_except(seq2mask: list, exception: list, mask_by: int = -100):
    e_l = len(exception)
    start_idx = 0
    while start_idx < len(seq2mask):
        is_overlap = start_idx + e_l <= len(seq2mask) \
                     and all([a == b for a, b in zip(seq2mask[start_idx:start_idx + e_l], exception)])
        if is_overlap:
            start_idx += e_l
        else:
            seq2mask[start_idx] = mask_by
            start_idx += 1
    return seq2mask

def pad_label(batch, pad_id=-100):
    max_length = max(*[len(each) for each in batch])
    paded = []
    for each in batch:
        paded.append(each + [pad_id,]*(max_length - len(each)))
    return paded

def custom_gpt2_pad(batch:list):
    max_length = max(*[len(each) for each in batch])
    paded = []
    mask = []
    for each in batch:
        paded.append(each + [each[-1],]*(max_length - len(each)))
        mask.append([1,]*len(each) + [0,]*(max_length - len(each)))
    return {
       'input_ids': torch.LongTensor(paded),
       'attention_mask': torch.FloatTensor(mask),
    }

def predict_a_sample(token_list: list,
                     label:list,
                     model: GPT2LMHeadModel, 
                     tokenizer: GPT2Tokenizer,
                     batch_size:int):

    with torch.no_grad():
        ## cross entropy is equivalent with negative log likelihood , in which lower means better
        ## that is why I sort without chaning the increasing direction
        loss_fct = CrossEntropyLoss(reduction='none')
        log_p = []
        for i in range(0, len(token_list), batch_size):
            batched_inputs = custom_gpt2_pad(token_list[i:i + batch_size])
            # batched_inputs = tokenizer.pad({'input_ids': token_list[i:i + batch_size]}, 
            #                                return_attention_mask=True, 
            #                                return_tensors='pt')
            batched_inputs = {k:v.to(device) for k, v in batched_inputs.items()}
            labels = label[i:i + batch_size]
            labels = torch.LongTensor(pad_label(labels)).to(device)
            assert labels.size(-1)== batched_inputs['input_ids'].size(-1), 'incomparatible label size'
            lm_logits = model(**batched_inputs).logits

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.reshape(labels.shape[0], -1).sum(dim=-1)
            log_p.extend(loss.reshape(-1).cpu().detach())
        log_p = np.array(log_p)

    return log_p

def rank_with_gpt(model:GPT2LMHeadModel, tokenizer:GPT2Tokenizer,
                                contexts:list, available_selections:list,
                                term_to_estimate: str = TERM_TO_ESTIMATE,
                                batch_size:int=DEEP_MODEL_BATCH_SIZE,
                                disable_paralell:bool=False,
                                verbose=None,):
    """
    Return the right ranked index of available_selections w.r.t corresponding context
    i.e:
    [
        [5, 2, 4, 3, 1, 0],
        [3, 2, 4, 5, 1, 0]
    ]
     means in the first context, the highest prob item is the 5-th, then the 2-nd, so on..
    :param model: a GPT2 model (with LM head to estimate token prob)
    :param tokenizer: a tokenizer
    :param contexts: context list for giving decision
    :param available_selections:
    :param term_to_estimate: the term that we want to estimate in the selections.
        if is None, the model will estimate the whole selection.
    :return: a numpy array of ranked index in shape of [num_contexts , num_selections]
    """
    batch_size = min(batch_size, len(contexts), len(available_selections))
    if isinstance(verbose, str) and verbose == 'detail':
        print('tokenizing inputs')
    context_ids = tokenizer(contexts, return_tensors=None, verbose=True)['input_ids']
    selection_ids = tokenizer(available_selections, return_tensors=None, verbose=True)['input_ids']
    labels = selection_ids
    if term_to_estimate is not None:
        labels = []
        term_to_estimate = tokenizer(term_to_estimate, add_special_tokens=False)['input_ids']
        for ids in selection_ids:
            labels.append(mask_all_except(copy(ids), term_to_estimate, -100))
    max_label_length = max(*[len(s) for s in labels])
    assert max_label_length < tokenizer.model_max_length, "target length is too large!"
    predictions = []
    all_selections = np.arange(len(available_selections))
    if isinstance(verbose, str) and  verbose == 'detail':
        print('prepared inputs')
    if model_paralell and not disable_paralell:
        model = torch.nn.DataParallel(model)
    if isinstance(verbose, str) and  verbose == 'detail':
        process = tqdm(context_ids)
    else:
        process = context_ids
    for context in process:
        truncated_ids = context[:tokenizer.model_max_length - max_label_length - 1]
        context_w_selections_ids = [truncated_ids +  selection for selection in selection_ids]
        label = [[-100,]*len(truncated_ids) +  selection for selection in labels]
        log_p = predict_a_sample(token_list=context_w_selections_ids,
                                    label=label,
                                    model=model,
                                    tokenizer=tokenizer,
                                    batch_size=batch_size)
        right_position_in_rank = np.argsort(log_p)
        predictions.append([all_selections[right_position_in_rank]])
        if isinstance(verbose, tqdm) and verbose is not None:
            verbose.update(1)
    predictions = np.concatenate(predictions, axis=0)
    return predictions


# import torch
# from torch.nn import CrossEntropyLoss
# from tqdm import tqdm
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# from env_config import *
# from metrics import *


# def mask_all_except(seq2mask: list, exception: list, mask_by: int = -100):
#     e_l = len(exception)
#     start_idx = 0
#     while start_idx < len(seq2mask):
#         is_overlap = start_idx + e_l <= len(seq2mask) \
#                      and all([a == b for a, b in zip(seq2mask[start_idx:start_idx + e_l], exception)])
#         if is_overlap:
#             start_idx += e_l
#         else:
#             seq2mask[start_idx] = mask_by
#             start_idx += 1
#     return seq2mask

# def pad_label(batch, pad_id=-100):
#     max_length = max(*[len(each) for each in batch])
#     paded = []
#     for each in batch:
#         paded.append(each + [pad_id,]*(max_length - len(each)))
#     return paded

# def pad_embedding(batch):
#     max_length = max(*[each.shape[0] for each in batch])
#     attention_mask = []
#     pad_batch = []
#     pad_vec = torch.FloatTensor([1,]*batch[0].shape[-1]).to(batch[0].device) # a vector of embedding_dim
#     pad_vec = pad_vec[None, ...] # a matrix of 1 x embedding_dim
#     for each in batch:
#         attention_mask.append([1,]*each.shape[0] + [0,]*(max_length-each.shape[0]))
#         paded = torch.cat([each, pad_vec.repeat(max_length-each.shape[0], 1)], dim=0)
#         pad_batch.append(paded)
#     attention_mask = torch.FloatTensor(attention_mask)
#     inputs_embeds = torch.cat([each[None, ...] for each in pad_batch], dim=0)
#     return {
#         'inputs_embeds': inputs_embeds,
#         'attention_mask': attention_mask
#     }

# def predict_a_sample(embeddings: list,
#                      label:list,
#                      model: GPT2LMHeadModel, 
#                      tokenizer: GPT2Tokenizer,
#                      batch_size:int):

#     with torch.no_grad():
#         ## cross entropy is equivalent with negative log likelihood , in which lower means better
#         ## that is why I sort without chaning the increasing direction
#         loss_fct = CrossEntropyLoss(reduction='none')
#         log_p = []
#         for i in tqdm(range(0, len(embeddings), batch_size)):
# #             batched_inputs = tokenizer.pad({'input_ids': token_list[i:i + batch_size]}, 
# #                                            return_attention_mask=True, 
# #                                            return_tensors='pt')
#             batched_inputs = pad_embedding(embeddings[i:i + batch_size])
#             batched_inputs = {k:v.to(device) for k, v in batched_inputs.items()}
#             labels = label[i:i + batch_size]
#             labels = torch.LongTensor(pad_label(labels)).to(device)
# #             assert labels.size(-1)== batched_inputs['input_ids'].size(-1), 'incomparatible label size'
#             lm_logits = model(**batched_inputs).logits

#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             loss = loss.reshape(labels.shape[0], -1).sum(dim=-1)
#             log_p.extend(loss.reshape(-1).cpu().detach())
#         log_p = np.array(log_p)

#     return log_p

# def rank_with_gpt(model:GPT2LMHeadModel, tokenizer:GPT2Tokenizer,
#                                 contexts:list, available_selections:list,
#                                 term_to_estimate: str = TERM_TO_ESTIMATE,
#                                 batch_size:int=DEEP_MODEL_BATCH_SIZE):
#     """
#     Return the right ranked index of available_selections w.r.t corresponding context
#     i.e:
#     [
#         [5, 2, 4, 3, 1, 0],
#         [3, 2, 4, 5, 1, 0]
#     ]
#      means in the first context, the highest prob item is the 5-th, then the 2-nd, so on..
#     :param model: a GPT2 model (with LM head to estimate token prob)
#     :param tokenizer: a tokenizer
#     :param contexts: context list for giving decision
#     :param available_selections:
#     :param term_to_estimate: the term that we want to estimate in the selections.
#         if is None, the model will estimate the whole selection.
#     :return: a numpy array of ranked index in shape of [num_contexts , num_selections]
#     """
#     print('tokenizing inputs')
#     context_ids = tokenizer(contexts, return_tensors=None, verbose=True)['input_ids']
#     selection_ids = tokenizer(available_selections, return_tensors=None, verbose=True)['input_ids']
#     labels = selection_ids
#     if term_to_estimate is not None:
#         labels = []
#         term_to_estimate = tokenizer(term_to_estimate, add_special_tokens=False)['input_ids']
#         for ids in selection_ids:
#             labels.append(mask_all_except(ids, term_to_estimate, -100))
#     max_label_length = max(*[len(s) for s in labels])
#     assert max_label_length < tokenizer.model_max_length, "target length is too large!"
#     predictions = []
#     all_selections = np.arange(len(available_selections))
#     print('prepare embeddings')
#     context_embeddings = []
#     selection_embedding = []
#     token_encode = model.transformer.wte
# #     if model_paralell:
# #         token_encode = torch.nn.DataParallel(token_encode)
#     for i in tqdm(range(0, len(context_ids), batch_size), "Embedding contexts"):
#         batch_contexts = context_ids[i:i+batch_size]
#         batch_context_ids = tokenizer.pad({'input_ids': batch_contexts}, return_tensors='pt')
#         embeddings = token_encode(batch_context_ids['input_ids'].to(device))
#         for embedding, mask in zip(embeddings, batch_context_ids['attention_mask']):
#             context_embeddings.append(embedding[mask>0, ...])
#     for i in tqdm(range(0, len(selection_ids), batch_size), "Embedding selections"):
#         batch_contexts = selection_ids[i:i+batch_size]
#         batch_context_ids = tokenizer.pad({'input_ids': batch_contexts}, return_tensors='pt')
#         embeddings = token_encode(batch_context_ids['input_ids'].to(device))
#         for embedding, mask in zip(embeddings, batch_context_ids['attention_mask']):
#             selection_embedding.append(embedding[mask>0, ...])
#     print('prepared inputs')
#     if model_paralell:
#         model = torch.nn.DataParallel(model)
#     for context in tqdm(context_embeddings):
#         truncated_embedding = context[:tokenizer.model_max_length - max_label_length - 1]
#         ctx_w_selection_embeddings = [torch.cat([truncated_embedding, selection], dim=0) for selection in selection_embedding]
# #         context_w_selections_ids = [truncated_ids.shap +  selection for selection in selection_ids]
#         label = [[-100,]*truncated_embedding.shape[0] +  selection for selection in labels]
#         log_p = predict_a_sample(embeddings=ctx_w_selection_embeddings,
#                                     label=label,
#                                     model=model,
#                                     tokenizer=tokenizer,
#                                     batch_size=batch_size)
#         right_position_in_rank = np.argsort(log_p)
#         predictions.append([all_selections[right_position_in_rank]])
#     predictions = np.concatenate(predictions, axis=0)
#     return predictions