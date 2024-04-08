import numpy as np
import random
import re
import torch
from torch_scatter import scatter_mean

import ipdb
st = ipdb.set_trace


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def sanity_check_target_after_processing(target):
    assert(len(target.bbox) == len(target.extra_fields["boxes"]))


def create_positive_map_for_od_labels(tokenized, label_to_positions, max_query_len):
    """construct a map such that positive_map[i] = j, where j is the object detection label of the token i"""
    """
    {3: [1: 5)}
    256 : -1 3 3 3 3 -1 .. 8 8 ..
    the woman in the garden
    -1 -1 -1 -1 -1
    """
    positive_map = torch.ones(max_query_len, dtype=torch.float) * -1  # -1 means no match
    keys = list(label_to_positions.keys())
    for j, key in enumerate(keys):
        tok_list = label_to_positions[key]
        # one label only mapps to one location
        beg, end = tok_list
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        assert beg_pos is not None and end_pos is not None
        positive_map[beg_pos: end_pos + 1].fill_(key)
    return positive_map


def create_positive_map(tokenized, tokens_positive, max_query_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_query_len), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    assert positive_map[:, -1].sum() == 0, "the last token should not be used, NoOBJ token"
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def convert_od_to_grounding_simple(
    labels, 
    ind_to_class, 
    disable_shuffle=True, 
    add_detection_prompt=False, 
    separation_tokens=" ",
    caption_prompt=None, 
    tokenizer=None, 
    max_query_len=256):
    """
    Convert object detection data into grounding data format, on the fly.
    ind_to_class: {0: "__background__", 1 : "person" ...}, contiguous id
    """

    def generate_sentence_from_labels(positive_label_list, negative_label_list, disable_shuffle=True):
        label_to_positions = {}
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle:
            random.shuffle(label_list)
            assert (caption_prompt is None), "Should not specify caption_prompt when shuffle is enabled!!"  # avoid potential bug

        if add_detection_prompt:
            pheso_caption = "object detection : "
        else:
            pheso_caption = ""
        

        for index, label in enumerate(label_list):
            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['prefix']

            start_index = len(pheso_caption)
            if caption_prompt is not None:
                pheso_caption += clean_name(caption_prompt[index]['name'])
            else:
                pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['suffix']

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens

        return label_to_positions, pheso_caption
    label_list = list(sorted(ind_to_class.keys()))  # do not include the background
    label_to_positions, pheso_caption = generate_sentence_from_labels(
        positive_label_list=label_list,
        negative_label_list=[],
        disable_shuffle=disable_shuffle
    )
    tokens_positive = [[label_to_positions[label]] for label in labels]

    # -1 is to preserve a token for the NoOBJ token
    tokenized = tokenizer(pheso_caption, return_tensors="pt",
                max_length=max_query_len,
                truncation=True)

    positive_map_od = create_positive_map_for_od_labels(tokenized, label_to_positions, max_query_len)
    positive_map = create_positive_map(tokenized, tokens_positive, max_query_len=max_query_len)
    
    assert torch.allclose(positive_map.sum(-1), torch.ones_like(positive_map.sum(-1))), "some positive maps are empty, possibly due to sequence length larger than max_query_len"

    return positive_map, positive_map_od, tokens_positive, pheso_caption



def convert_grounding_to_od_logits(logits, num_class, positive_map_od):
    assert NotImplementedError, "Need to verify the correctness of this function"
    assert logits.max() <= 1.0 and logits.min() >= 0.0, "logits should be in [0, 1]"
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_class).to(logits.device)
    for label_j in range(num_class):        
        locations_label_j = (positive_map_od == label_j).nonzero(as_tuple=True)[0].tolist()
        if len(locations_label_j) == 0:
            continue
        scores[:, :, label_j] = logits[:, :, torch.LongTensor(locations_label_j)].mean(-1)
    return scores


def convert_grounding_to_od_logits_batched(logits, num_class, positive_map_od):
    """
    Here the logits are raw outputs not the softmax output!
    logits: (batch_size, q, seq_len)
    num_class: N
    positive_map_od: (batch_size, seq_len)
    """
    scores = torch.ones(logits.shape[0], logits.shape[1], num_class).to(logits) * -100.0

    positive_map_od[positive_map_od == -1] = num_class
    
    scores_ = scatter_mean(
        logits, positive_map_od[:, None, :].expand(-1, logits.shape[1], -1), dim=2
    )
    mask = torch.ones_like(scores_).bool()
    mask.scatter_(2, positive_map_od[:, None, :].expand(-1, logits.shape[1], -1), False)
    scores_[mask] = -100.0
    
    # remove invalid scores
    scores_ = scores_[..., :-1]

    scores[:, :, :scores_.shape[-1]] = scores_
    
    return scores
    

def get_positive_tokens(caption, token_lists):
    tokens_positive = []
    for token in token_lists:
        start_index = caption.find(token)
        if start_index == -1:
            raise ValueError(f"token {token} not found in caption {caption}")
        end_index = start_index + len(token)
        tokens_positive.append([start_index, end_index])
    return tokens_positive


if __name__ == "__main__":
    pass