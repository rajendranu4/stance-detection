import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def _pairwise_distances(embeddings, squared=False):

    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances


def _get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss


def batch_hard_easy_triplet_loss(labels, embeddings, margin, squared=False):

    max_number = torch.tensor(10000).type(torch.FloatTensor).to('cuda:0')
    min_number = torch.tensor(0).type(torch.FloatTensor).to('cuda:0')

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # identify easy positive
    anchor_positive_dist_wo_zero = torch.where(anchor_positive_dist>0, anchor_positive_dist, max_number)
    easiest_positive_dist, _ = anchor_positive_dist_wo_zero.min(1, keepdim=True)

    # concatenate hard and easy positives
    hardest_easiest_positive_dist = torch.cat((hardest_positive_dist, easiest_positive_dist), 1)

    # if all the entries in the row is 0 and was replaced with max_number previously, then it needs to be
    # updated to 0 again (the min number)
    hardest_easiest_positive_dist[hardest_easiest_positive_dist == max_number] = min_number

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # identify easy negative
    anchor_negative_dist_for_easy = mask_anchor_negative * pairwise_dist
    easiest_negative_dist, _ = anchor_negative_dist_for_easy.max(1, keepdim=True)

    # concatenate hard and easy negatives
    hardest_easiest_negative_dist = torch.cat((hardest_negative_dist, easiest_negative_dist), 1)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_easiest_positive_dist - hardest_easiest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss


def _get_anchor_positive_easy_matching_tokens_mask(tokenizer, text, mask_anchor_positive):
    exclusion_list = set(['<s>', '</s>', '<pad>', '?', '.', ',', '!', '"', '-', 'a', 'an', 'the', 'be', 'are', 'were'])
    print("******************************************************")

    tokens_matching_matrix = torch.eye(text.shape[0])
    for row in range(tokens_matching_matrix.shape[0]):
        for col in range(tokens_matching_matrix.shape[1]):
            tokens_1 = tokenizer.convert_ids_to_tokens(text[row])
            tokens_2 = tokenizer.convert_ids_to_tokens(text[col])
            cleaned_tokens_1 = set(tokens_1) - exclusion_list
            cleaned_tokens_2 = set(tokens_2) - exclusion_list
            print(cleaned_tokens_1)
            print(cleaned_tokens_2)
            matching_tokens = cleaned_tokens_1.intersection(cleaned_tokens_2)
            print(matching_tokens)
            print("\n")
        break


def batch_hard_easy_tokens_triplet_loss(tokenizer, text, labels, embeddings, margin, squared=False):

    max_number = torch.tensor(10000).type(torch.FloatTensor).to('cuda:0')
    min_number = torch.tensor(0).type(torch.FloatTensor).to('cuda:0')

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # identify easy positive with less number of matching tokens
    mask_anchor_positive_easy_matching_tokens = _get_anchor_positive_easy_matching_tokens_mask(tokenizer, text, mask_anchor_positive)
    anchor_positive_dist_wo_zero = torch.where(anchor_positive_dist>0, anchor_positive_dist, max_number)
    easiest_positive_dist, _ = anchor_positive_dist_wo_zero.min(1, keepdim=True)

    # concatenate hard and easy positives
    hardest_easiest_positive_dist = torch.cat((hardest_positive_dist, easiest_positive_dist), 1)

    # if all the entries in the row is 0 and was replaced with max_number previously, then it needs to be
    # updated to 0 again (the min number)
    hardest_easiest_positive_dist[hardest_easiest_positive_dist == max_number] = min_number

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # identify easy negative
    anchor_negative_dist_for_easy = mask_anchor_negative * pairwise_dist
    easiest_negative_dist, _ = anchor_negative_dist_for_easy.max(1, keepdim=True)

    # concatenate hard and easy negatives
    hardest_easiest_negative_dist = torch.cat((hardest_negative_dist, easiest_negative_dist), 1)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_easiest_positive_dist - hardest_easiest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss


def batch_multiple_hard_triplet_loss(labels, embeddings, margin, squared=False):

    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    #print("Pairwise Distance")
    #print(pairwise_dist)

    mask_anchor_positive = sent_sim(embeddings, labels, 'positive').float()

    #print("\nSentence Anchor Positive Mask")
    #print(mask_anchor_positive)

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    #print("\nAnchor Positive Dist")
    #print(anchor_positive_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    #print("\nHard Positive Dist")
    #print(hardest_positive_dist)

    mask_anchor_negative = sent_sim(embeddings, labels, 'negative').float()

    #print("\nSentence Anchor Negative Mask")
    #print(mask_anchor_negative)

    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    #print("\nAnchor Negative Distance")
    #print(anchor_negative_dist)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    #print("\nHardest negative distance")
    #print(hardest_negative_dist)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    #print("\ntl")
    #print(tl)
    tl = F.relu(tl)
    triplet_loss = tl.mean()
    #print("\nTriplet Loss")
    #print(triplet_loss)

    return triplet_loss


def sent_sim(embeddings, labels, pos_or_neg):
    if pos_or_neg == 'positive':
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
        #print("\nMask Anchor Positive")
        #print(mask_anchor_positive)
        '''
        sent_emb = sentence_sim_model.encode(text)

        sent_sim_matrix = torch.eye(text.shape(0))
        for row in range(sent_sim_matrix.shape[0]):
            for col in range(sent_sim_matrix.shape[1]):
                sent_sim_matrix[row][col] = util.pytorch_cos_sim(sent_emb[row], sent_emb[col])'''

        sent_sim_matrix = cosine_similarity(embeddings.detach().cpu())
        sent_sim_matrix = torch.from_numpy(sent_sim_matrix).type(torch.LongTensor).to("cuda:0")

        sent_sim_matrix = torch.where(sent_sim_matrix > 0.3, sent_sim_matrix, 0)
        sent_sim_matrix = torch.where(sent_sim_matrix < 0.6, sent_sim_matrix, 0)
        sent_sim_matrix = sent_sim_matrix.fill_diagonal_(0)
        sent_sim_matrix = torch.where(sent_sim_matrix > 0, 1, 0)

        mask_anchor_positive_similarity = mask_anchor_positive * sent_sim_matrix.float()

        return mask_anchor_positive_similarity

    else:
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
        #print("\nMask Anchor Negative")
        #print(mask_anchor_negative)

        '''sent_emb = sentence_sim_model.encode(text)

        sent_sim_matrix = torch.eye(len(text))
        for row in range(sent_sim_matrix.shape[0]):
            for col in range(sent_sim_matrix.shape[1]):
                sent_sim_matrix[row][col] = util.pytorch_cos_sim(sent_emb[row], sent_emb[col])'''

        sent_sim_matrix = cosine_similarity(embeddings.detach().cpu())
        sent_sim_matrix = torch.from_numpy(sent_sim_matrix).type(torch.LongTensor).to("cuda:0")

        sent_sim_matrix = torch.where(sent_sim_matrix < 0.5, sent_sim_matrix, 0)
        sent_sim_matrix = torch.where(sent_sim_matrix > 0, 1, 0)

        mask_anchor_negative_similarity = mask_anchor_negative * sent_sim_matrix.float()

        return mask_anchor_negative_similarity