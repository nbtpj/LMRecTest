import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
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



def predict_a_sample(context: str, available_selections: list,
                     model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                     term_to_estimate: str = TERM_TO_ESTIMATE):
    context_w_selection = [context + selection for selection in available_selections]
    inputs = tokenizer(context_w_selection, truncation=True, padding=True, return_tensors='pt')
    label = []
    if term_to_estimate is not None:
        term_to_estimate = tokenizer(term_to_estimate, add_special_tokens=False)['input_ids']
        for ids in inputs['input_ids']:
            label.append(mask_all_except(list(ids), term_to_estimate, -100))
    else:
        ## mask the whole context (estimate all selections)
        terms_to_estimate = tokenizer(available_selections, add_special_tokens=False)['input_ids']
        for ids, term_to_estimate in zip(inputs['input_ids'], terms_to_estimate):
            label.append(mask_all_except(list(ids), term_to_estimate, -100))

    label = torch.LongTensor(label)
    label = torch.where(label == tokenizer.pad_token_id, -100, label)
    with torch.no_grad():
        ## cross entropy is equivalent with negative log likelihood , in which lower means better
        ## that is why I sort without chaning the increasing direction
        loss_fct = CrossEntropyLoss(reduction='none')
        log_p = []
        for i in range(0, len(available_selections), DEEP_MODEL_BATCH_SIZE):
            batched_input_ids = inputs['input_ids'][i:i + DEEP_MODEL_BATCH_SIZE, ...].to(model.device)
            batched_attention_mask = inputs['attention_mask'][i:i + DEEP_MODEL_BATCH_SIZE, ...].to(model.device)
            labels = label[i:i + DEEP_MODEL_BATCH_SIZE, ...].to(model.device)
            lm_logits = model(input_ids=batched_input_ids, attention_mask=batched_attention_mask).logits

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            log_p.extend(loss.reshape(-1).cpu().detach())
        log_p = np.array(log_p)

    return log_p


def rank_with_gpt(model:GPT2LMHeadModel, tokenizer:GPT2Tokenizer,
                                contexts:list, available_selections:list,
                                term_to_estimate: str = TERM_TO_ESTIMATE):
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
    predictions = []
    all_selections = np.arange(len(available_selections))
    for context in tqdm(contexts):
        log_p = predict_a_sample(context, available_selections, model, tokenizer, term_to_estimate,)
        right_position_in_rank = np.argsort(log_p)
        predictions.append([all_selections[right_position_in_rank]])
    predictions = np.concatenate(predictions, axis=0)
    return predictions