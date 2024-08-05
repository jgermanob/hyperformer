from seqeval.metrics import f1_score

def normalize_sequences(batch_predictions, batch_gold):
    normalized_sequences = []
    for pred, gold in zip(batch_predictions, batch_gold):
        size = len(gold)
        norm_seq = ['O'] * size
        for i, tag in enumerate(pred):
            if i < size:
                norm_seq[i] = tag
        normalized_sequences.append(norm_seq)
    return normalized_sequences

def sBIO_to_BIO(tag_seq, size=None):
    token_taq_seq = tag_seq.strip().split()
    bio_seq = ['O'] * len(token_taq_seq)
    parent_index = 0
    
    for i, tag in enumerate(token_taq_seq):
        if tag != 'O' and tag !='I':
            parent_index = i + 1
            bio_seq[i] = 'B-{}'.format(tag)
            for j in range(parent_index, len(token_taq_seq)):
                if token_taq_seq[j] == 'I':
                    bio_seq[j] = 'I-{}'.format(tag)
                else:
                    break
    return bio_seq

def get_BIO_predictions(decoded_predictions):
    bio_predictions = []
    for prediction in decoded_predictions:
        bio_predictions.append(sBIO_to_BIO(prediction))
    return bio_predictions

def get_micro_f1_score(predictions, targets):
    bio_predictions = get_BIO_predictions(predictions)
    bio_labels = get_BIO_predictions(targets)
    normalized_preds = normalize_sequences(bio_predictions,bio_labels)
    f1 = f1_score(bio_labels, normalized_preds, average='micro')
    return f1