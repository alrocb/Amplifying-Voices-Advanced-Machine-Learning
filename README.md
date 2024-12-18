<div align="center">

# **Amplifying Voices: Self-Supervised Learning for Low-Resource and Minority Languages**

</div>

![image](https://github.com/user-attachments/assets/20be09ae-d310-49d6-a029-917a013e3cd2)


This repository contains the implementation, datasets, and experiments for our project exploring self-supervised learning (SSL) for Automatic Speech Recognition (ASR) in low-resource languages, with a focus on Catalan. We also extend the model's capabilities to tasks like **accent recognition** and **gender detection**.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
  - [Dependencies](#main-dependencies)
  - [Installation](#installation)
- [Data](#data)
- [Training Phases](#training-phases)
  - [Pretraining](#pretraining)
  - [Fine-Tuning](#fine-tuning)
  - [Extended Tasks](#extended-tasks)
- [Results](#results)
  - [Quantitative Results](#quantitative-results)
  - [Qualitative Results](#qualitative-results)
  - [Grad-CAM Visualizations](#grad-cam-visualizations)
- [Citation](#citation)

---

## **Introduction**
Many languages around the world lack sufficient data and resources for building robust ASR systems. Using **self-supervised learning** models like Wav2Vec2, we aim to:
- Demonstrate the potential of SSL for underrepresented languages.
- Test whether linguistic similarity (e.g., Spanish-to-Catalan) improves model adaptation.
- Explore related speaker tasks like accent and gender classification.

We leverage **Mozilla Common Voice**, a community-driven dataset, to train and evaluate our models.

---

## **Features**
- **Self-Supervised Pretraining and Finetuning:** Train Wav2Vec2 from scratch on Catalan using unlabeled data and finetuning for ASR task using small sources of labeled data (transcriptions).
- **Fine-Tuning:** Fine-tune pretrained models (e.g., Spanish, English, multilingual) for Catalan ASR.
- **Extended Applications:** Apply learned embeddings to accent recognition and gender classification.
- **Model Interpretability:** Use Grad-CAM to visualize the model’s focus during transcription.

---

## **Setup**

### **Main Dependencies**
- Python >= 3.8
- Transformers (`pip install transformers`)
- Datasets (`pip install datasets`)
- PyTorch with CUDA support
- Additional libraries: pandas, scikit-learn, wandb
### **Installation**

1.Clone the repository:
  ```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
   ```
2.Install the required dependencies:
  ```bash
pip install -r requirements.txt
   ```
---

## **Data**
We use **Mozilla Common Voice**, particularly versions 18.0 and 19.0 for Catalan labeled source. You can download the dataset from [Common Voice](https://commonvoice.mozilla.org/). 

- **Data Preparation:**
   - Audio is resampled to 16kHz.
   - Transcriptions are tokenized at the character level for Connectionist Temporal Classification (CTC).
   - Additional metadata (e.g., accents, gender) is extracted for extended tasks.

---

## **Training Phases**

### **Pretraining**
- Pretrain Wav2Vec2 from scratch using 30+ hours of unlabeled Catalan speech.
- Leverage contrastive and diversity loss to learn robust speech representations.

### **Fine-Tuning**
- Fine-tune models pretrained on Spanish, English, or multilingual corpora using labeled Catalan data.
- Train with CTC loss for speech-to-text alignment.

### **Extended Tasks**
- Fine-tune embeddings for:
  - **Accent Recognition:** Classify Catalan accents into five categories.
  - **Gender Recognition:** Classify speech samples as male or female.

---

## **Results**

### **Quantitative Results**
| Model                  | WER (Normalized) | Accent Recognition F1 | Gender Recognition F1 |
|------------------------|------------------|-----------------------|------------------------|
| Scratch Pretrained     | 0.70            | -                     | -                      |
| Spanish-Pretrained     | 0.40            | -                     | -                      |
| English-Pretrained     | 0.96            | -                     | -                      |
| Multilingual-Pretrained| 0.10            | 0.9768                | 0.9725                 |

### **Qualitative Results**
See example transcriptions in the inference notebook.

### **Grad-CAM Visualizations**
Grad-CAM heatmaps illustrate how the model focuses on specific audio segments during transcription. View examples in the explainability notebook.

### Visualization on Model Overview Pretraining

![image](https://github.com/user-attachments/assets/96c4ff33-21a3-4419-8363-213d7fefe316)

---

## **Citation**
If you use this work, please consider citing:
```plaintext
@article{
  title={Amplifying Voices: Self-Supervised Learning for Low-Resource and Minority Languages},
  author={Alex Roldan},
  year={2024}
}
```

👤 **Alex Roldan**

  - GitHub: [GitHub Profile](https://github.com/alrocb)
  - LinkedIn: [LinkedIn](https://www.linkedin.com/in/alex-roldan-55488a215/)

If you need the weights for the models, contact me.
