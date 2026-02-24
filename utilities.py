import yaml
import nltk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
nltk.download('punkt_tab')


def load_config_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def compute_iterater_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    au_prc = average_precision_score(labels, pred.predictions, average='macro')

    return {
        'Accuracy': acc,
        'AU_PRC': au_prc,
        'P_Avg': avg_precision,
        'R_Avg': avg_recall,
        'F1_Avg': avg_f1,
        'P_Clarity': precision[0],
        'R_Clarity': recall[0],
        'F1_Clarity': f1[0],
        'P_Fluency': precision[1],
        'R_Fluency': recall[1],
        'F1_Fluency': f1[1],
        'P_Coherence': precision[2],
        'R_Coherence': recall[2],
        'F1_Coherence': f1[2],
        'P_Style': precision[3],
        'R_Style': recall[3],
        'F1_Style': f1[3],
        'P_Meaning-Changed': precision[4],
        'R_Meaning-Changed': recall[4],
        'F1_Meaning-Changed': f1[4]
    }


def compute_erevise_purpose_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    au_prc = average_precision_score(labels, pred.predictions, average='macro')

    return {
        'Accuracy': acc,
        'AU_PRC': au_prc,
        'P_Avg': avg_precision,
        'R_Avg': avg_recall,
        'F1_Avg': avg_f1,
        'P_Relevant': precision[0],
        'R_Relevent': recall[0],
        'F1_Relevant': f1[0],
        'P_Irrelevant': precision[1],
        'R_Irrelevant': recall[1],
        'F1_Irrelevant': f1[1],
        'P_AlreadyExist': precision[2],
        'R_AlreadyExist': recall[2],
        'F1_AlreadyExist': f1[2],
        'P_LCE': precision[3],
        'R_LCE': recall[3],
        'F1_LCE': f1[3],
        'P_no_LCE': precision[4],
        'R_no_LCE': recall[4],
        'F1_no_LCE': f1[4],
        'P_no_Commentary': precision[5],
        'R_no_Commentary': recall[5],
        'F1_no_Commentary': f1[5]
    }


def get_iterater_data(data, upsample_values):
    before_sents = []
    after_sents = []
    labels = []
    data_count = [0, 0, 0, 0, 0]
    ctrl_tokens_dict = {"clarity": 0, "fluency": 1, "coherence": 2, "style": 3, "meaning-changed": 4}
    for line in data:
        before_sent = line["before_sent"]
        after_sent = line["after_sent"]
        label_str = line["labels"]
        if label_str == "others": continue
        label = ctrl_tokens_dict[label_str]

        if upsample_values:
            upsample_value = upsample_values[label]
        else:
            upsample_value = 1
        for _ in range(upsample_value):
            before_sents.append(before_sent)
            after_sents.append(after_sent)
            labels.append(label)
            data_count[label] += 1
    print("data count", data_count)
    return before_sents, after_sents, labels


def get_erevise_purpose_data(data, upsample_values):
    before_sents = []
    after_sents = []
    labels = []
    data_count = [0, 0, 0, 0, 0, 0]
    ctrl_tokens_dict = {"relevant": 0, "irrelevant": 1, "already exists": 2, "LCE": 3, "not LCE": 4, "commentary": 5}
    for line in data.to_dict(orient="records"):
        before_sent = line["old_sentences"]
        after_sent = line["new_sentences"]
        label_str = line["purpose_labels"]
        if str(before_sent) == "nan": before_sent = ""
        if str(after_sent) == "nan": after_sent = ""
        if label_str == "claim": continue
        label = ctrl_tokens_dict[label_str]

        if upsample_values:
            upsample_value = upsample_values[label]
        else:
            upsample_value = 1
        for _ in range(upsample_value):
            before_sents.append(before_sent)
            after_sents.append(after_sent)
            labels.append(label)
            data_count[label] += 1
    print("data count", data_count)
    return before_sents, after_sents, labels


def generate_prompt_erevise_purpose(first_sent, second_sent, context=None):
    return f"""
        ### Instruction: Identify the intention of the revision between original text and revised text. 
        The possible intentions include: relevant, irrelevant, already exists, linked claim evidence, not linked claim evidence, commentary

        ### Original Text: {first_sent}
        ### Revised Text: {second_sent}
        """


def generate_prompt_iterater(first_sent, second_sent, context=None):
    return f"""
        ### Instruction: Identify the intention of the revision between original sentence and revised sentence. 
        The possible intentions include: Clarity, Style, Fluency, Coherence, and Meaning-change.

        ### Original Sentence: {first_sent}
        ### Revised Sentence: {second_sent}
        """
