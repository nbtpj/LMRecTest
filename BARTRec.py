import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

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


def pre_embed_for_decoder(list_txt: list, model: BartForConditionalGeneration,
                          tokenizer: BartTokenizer, 
                          disable_paralell:bool,
                          term_to_estimate: str = TERM_TO_ESTIMATE,
                          batch_size: int = DEEP_MODEL_BATCH_SIZE):
    token_encode = model.get_input_embeddings()
    if model_paralell and not disable_paralell:
        token_encode = torch.nn.DataParallel(token_encode)
    labels = tokenizer(list_txt, truncation=True, padding=True, return_tensors='pt')
    outputs = []
    for i in range(0, len(list_txt), batch_size):
        batched_inputs = model.prepare_decoder_input_ids_from_labels(
            labels['input_ids'][i:i + batch_size, ...])
        with torch.no_grad():
            outputs.append(token_encode(batched_inputs.to(model.device)))
    label = []
    if term_to_estimate is not None:
        term_to_estimate = tokenizer(term_to_estimate, add_special_tokens=False)['input_ids']
        for ids in labels['input_ids']:
            label.append(mask_all_except(list(ids), term_to_estimate, -100))
    else:
        for ids in labels['input_ids']:
            label.append(list(ids))
    label = torch.LongTensor(label)
    label = torch.where(label == tokenizer.pad_token_id, -100, label)
    return {
        'decoder_input_ids': labels['input_ids'],
        'decoder_inputs_embeds': torch.concat(outputs, dim=0),
        'decoder_attention_mask': labels['attention_mask'],
        'labels': label
    }


def encode(list_txt: list, model: BartForConditionalGeneration, 
           tokenizer: BartTokenizer, batch_size:int,
           disable_paralell:bool):
    encoder = model.get_encoder()
    if model_paralell and not disable_paralell:
        encoder = torch.nn.DataParallel(encoder)
    for i in range(0, len(list_txt), batch_size):
        inputs = tokenizer(list_txt[i:i + batch_size], truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = encoder(**inputs).last_hidden_state
            for x, mask in zip(outputs, inputs['attention_mask'] > 0):
                yield x[mask > 0, ...]


def predict_log_prob(context, decoder_inputs_embeds, 
                     decoder_attention_mask, labels, model, batch_size:int, disable_paralell:bool):
    with torch.no_grad():
        ## cross entropy is equivalent with negative log likelihood , in which lower means better
        ## that is why I sort without chaning the increasing direction
        loss_fct = CrossEntropyLoss(reduction='none')
        log_p = []
        decoder = model.get_decoder()
        if model_paralell and not disable_paralell:
            decoder = torch.nn.DataParallel(decoder)
        for i in range(0, decoder_inputs_embeds.shape[0], batch_size):
            batched_labels = labels[i:i + batch_size, ...]
            batched_decoder_inputs_embeds = decoder_inputs_embeds[i:i + batch_size, ...]
            batched_decoder_attention_mask = decoder_attention_mask[i:i + batch_size, ...]
            encoder_last_hidden = context.expand((batched_decoder_inputs_embeds.shape[0], -1, -1)).to(device)
            decoder_output = decoder(
                attention_mask=batched_decoder_attention_mask,
                encoder_hidden_states=encoder_last_hidden,
                inputs_embeds=batched_decoder_inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True, ).last_hidden_state

            lm_logits = model.lm_head(decoder_output)
            lm_logits = lm_logits + model.final_logits_bias.to(lm_logits.device)

            masked_lm_loss = loss_fct(lm_logits.view(-1, model.config.vocab_size), batched_labels.view(-1))
            masked_lm_loss = masked_lm_loss.reshape(batched_labels.shape[0], -1).sum(dim=-1)
            log_p.extend(masked_lm_loss.reshape(-1).cpu().detach())
        log_p = np.array(log_p)

    return log_p


def rank_with_bart(model: BartForConditionalGeneration, tokenizer: BartTokenizer,
                   contexts: list, available_selections: list,
                   decoder_embeddings: dict = None,
                   context_embedding: list = None,
                   term_to_estimate: str = TERM_TO_ESTIMATE,
                   batch_size:int=DEEP_MODEL_BATCH_SIZE,
                   disable_paralell:bool=False,
                   verbose:tqdm=None):
    """
    Return the right ranked index of available_selections w.r.t corresponding context
    i.e:
    [
        [5, 2, 4, 3, 1, 0],
        [3, 2, 4, 5, 1, 0]
    ]
     means in the first context, the highest prob item is the 5-th, then the 2-nd, so on..
    :param model: a BART model (with LM head to estimate token prob)
    :param tokenizer: a tokenizer
    :param contexts: context list for giving decision
    :param available_selections:
    :param decoder_embeddings: cached values
    :param context_embedding: cached values
    :param term_to_estimate: the term that we want to estimate in the selections.
        if is None, the model will estimate the whole selection.
    :return: a numpy array of ranked index in shape of [num_contexts , num_selections]
    """
    all_selections = np.arange(len(available_selections))
    if decoder_embeddings is None:
        decoder_embeddings = pre_embed_for_decoder(available_selections, model, tokenizer,
                                                   term_to_estimate=term_to_estimate, 
                                                   batch_size=batch_size,
                                                   disable_paralell=disable_paralell)
        decoder_embeddings = {k: v.to(model.device) for k, v in decoder_embeddings.items()}
        if 'decoder_input_ids' in decoder_embeddings:
            del decoder_embeddings['decoder_input_ids']
    predictions = []
    if context_embedding is None:
        context_embedding = encode(contexts, model, tokenizer, 
                                   batch_size=batch_size, 
                                   disable_paralell=disable_paralell)
    if isinstance(verbose, str) and  verbose == 'detail':
        process = tqdm(context_embedding)
    else:
        process = context_embedding
    for ctx_embedding in process:
        log_p = predict_log_prob(ctx_embedding, **decoder_embeddings, 
                                 model=model, batch_size=batch_size,
                                 disable_paralell=disable_paralell)
        right_position_in_rank = np.argsort(log_p)
        predictions.append([all_selections[right_position_in_rank]])
        if isinstance(verbose, tqdm) and verbose is not None:
            verbose.update(1)
    predictions = np.concatenate(predictions, axis=0)
    return predictions
