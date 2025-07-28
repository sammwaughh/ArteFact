# ArtContext

---

**Submitted by:** Samuel Waugh  
**Supervised by:** Dr Stuart James
**Affiliation:** Department of Computer Science, Durham University  
**Degree:** Bachelor of Science (BSc) in Computer Science  
**Submission date:** 8 May 2025  

---


## Quick-start (Conda)

```bash
# 1) Install Miniconda / Anaconda if you don’t already have it
#    https://docs.conda.io/en/latest/miniconda.html

# 2) Create the environment from this repo’s file
conda env create -f environment.yml        # → installs as “painting_clip_env”

# 3) Activate and go
conda activate painting_clip_env
```

## Project structure

```text
.
├── README.md
├── environment_lock.yml
│
├── clip_finetuned_lora_best
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
│
├── Excel Files
│   ├── big_results.xlsx
│   ├── clip_inference.xlsx
│   ├── fine_tune_dataset.xlsx
│   ├── openalex_topics_formatted.xlsx
│   ├── painters.xlsx
│   ├── painting_clip_inference.xlsx
│   ├── paintings_metadata.xlsx
│   └── paintings_with_labels.xlsx
│
├── Graphs
│   ├── avg_precision_recall.png
│   ├── linkcount_histogram.png
│   ├── pdf_count_histogram.png
│   ├── pdf_count_top_bottom.png
│   ├── train_loss_over_epochs.png
│   └── val_nce_over_epochs.png
│
└── Scripts
    ├── Evaluation
    │   ├── big_compare_zero_shot.py
    │   ├── distribution_of_link_counts.py
    │   ├── may_attention_map.py
    │   ├── open_access_count.py
    │   ├── precision_recall_curves.py
    │   └── top_ten_sentences.py
    │
    └── Pipeline
        ├── do_fine_tune.py
        ├── find_needle_in_haystack.py
        ├── generate_labels.py
        ├── harvest_open_alex.py
        ├── pdfs_to_markdown.py
        ├── process_wikidata.py
        └── save_bert_embeddings.py

```

## Important Notes
This code submission is structured to be easily readable and such that an examiner can verify that the work has been undertaken.
To actually run the code files with ease, they will need to be in the same directory as the Excel Files they depend upon.
Furthermore, the pipeline relies on several largedataset like the PDFs, Markdown Works, The Markdown Metadata, and the Images. Taken together these constitute over 30GB of data and it was not feasible to submit any of them with this code base. There is also a lot of small intermediate scripts which do things like formatting, renaming, and the like which have not been included.

## Declration of use of generative AI
In accordance with the department's guidelines on using generative AI I declare here how I used it for this project: helped me write LaTeX code for tables, graphs and mathematical notation, translating from plain english descriptions of what I wanted; helped me to clean and comment my code scripts, rendering them more readable; helped me write a legally sound paragraph about copyright for the project paper and found me the sources I needed for it; helped me to understand older code which I had written previously when revisiting the project; helped me to make this README.md file prettier.