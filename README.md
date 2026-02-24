# [Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction](https://aclanthology.org/2025.findings-emnlp.829.pdf)


Abstract: Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora.

## Usage
Please install Anaconda 24.5.0 with Python 3.9 and create a new virtual environment named `revision` using the yaml file and command `conda env create -f environment.yaml -n revision`. Please configure huggingface key and wandb key in `config.yaml` if needed and then run the following command to start fine-tuning. Note that you can change the hyperparameters in the command and code as needed.
```aiignore
python main.py \
  --model-name llama3.1-8b \
  --dataset-name iterater-human \
  --adapter-name lora \
  --importance-metric-name gradient \
  --tuning-method ir \
  --output-dir experiments/debug \
  --use-instruction
```

## Citation
```angular2html
@inproceedings{
    liu-litman-2025-efficient,
    title = "Efficient Layer-wise {LLM} Fine-tuning for Revision Intention Prediction",
    author = "Liu, Zhexiong and Litman, Diane",
    editor = "Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = Nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.829/",
    doi = "10.18653/v1/2025.findings-emnlp.829",
    pages = "15319--15334",
    ISBN = "979-8-89176-335-7"}
```

## Acknowledgements
This project is built on prior [IST](https://github.com/antgroup/importance-aware-sparse-tuning-IST-paper?tab=readme-ov-file) codebase.