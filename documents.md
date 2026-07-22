Maganda. Sa tingin ko **ito ang tamang sequence**.

❌ Huwag muna Android.
❌ Huwag muna Java.
❌ Huwag muna deployment.

**Model muna.**

Actually, ganito rin ang ginagawa ng OpenAI, Meta (Llama), Google (Gemma), at Alibaba (Qwen). **Una nilang binubuo ang model**, saka lang nila iniisip ang deployment.

---

# TMC-LM

# Development Documentation

## Phase 1 – Model Development

### Version 1.0

---

# Table of Contents

```text
Chapter 1
Project Overview

Chapter 2
Development Environment

Chapter 3
Model Architecture

Chapter 4
Knowledge Pipeline

Chapter 5
Dataset Pipeline

Chapter 6
Tokenizer

Chapter 7
Base Model

Chapter 8
Fine-Tuning

Chapter 9
Evaluation

Chapter 10
Knowledge Expansion

Chapter 11
Model Versioning

Chapter 12
Future Development
```

---

# CHAPTER 1

# Project Overview

## 1.1 Project Title

**TMC-LM: A Domain-Specific Large Language Model for Trinidad Municipal College**

---

## 1.2 Description

The TMC Language Model (TMC-LM) is a domain-specific Large Language Model designed to understand, learn, and answer questions exclusively related to Trinidad Municipal College (TMC).

Unlike general-purpose language models, TMC-LM focuses only on official institutional knowledge. Its knowledge base is built from authorized documents such as student handbooks, faculty manuals, academic policies, office procedures, memorandums, and other institutional records.

The primary objective is to develop an offline AI model that can continuously learn from newly added documents through incremental fine-tuning while maintaining high accuracy within the TMC domain.

---

# CHAPTER 2

# Development Environment

## Objective

Prepare a complete AI development environment for training, fine-tuning, and evaluating the TMC Language Model.

---

## Software Requirements

| Software                  | Purpose                           |
| ------------------------- | --------------------------------- |
| Windows 11                | Development Operating System      |
| Python 3.11               | Main Programming Language         |
| Visual Studio Code        | Code Editor                       |
| Git                       | Version Control                   |
| Ollama                    | Local model testing and inference |
| PyTorch                   | Deep Learning Framework           |
| Hugging Face Transformers | Base model loading and training   |
| PEFT                      | LoRA Fine-Tuning                  |
| Accelerate                | Distributed Training              |
| SentencePiece             | Tokenizer                         |
| Datasets                  | Dataset Management                |
| PyMuPDF                   | PDF Reader                        |
| python-docx               | Word Reader                       |
| openpyxl                  | Excel Reader                      |
| pandas                    | Spreadsheet Processing            |
| Tesseract OCR             | OCR Engine                        |

---

# CHAPTER 3

# Model Architecture

## Overview

The proposed AI system is based on a lightweight Transformer architecture.

Instead of creating a language model from scratch, an existing pretrained model will be adapted to institutional knowledge using parameter-efficient fine-tuning.

---

## Proposed Base Model

```
TinyLlama 1.1B
```

Alternative models

```
Gemma

Qwen

Phi
```

---

## Why TinyLlama?

Advantages

✔ Small

✔ Fast

✔ Open Source

✔ Excellent English Understanding

✔ Mobile Friendly

✔ Easy to Fine-Tune

✔ Large Community Support

---

# CHAPTER 4

# Knowledge Acquisition Pipeline

The AI acquires institutional knowledge from official documents.

Supported document types

```
PDF (.pdf)

↓

PDF Loader
```

```
Microsoft Word (.docx)

↓

Word Loader
```

```
Microsoft Excel (.xlsx)

↓

Excel Loader
```

```
Plain Text (.txt)

↓

TXT Loader
```

```
CSV (.csv)

↓

CSV Loader
```

```
JSON (.json)

↓

JSON Loader
```

---

# Unified Knowledge Pipeline

```
Documents

↓

File Type Detector

↓

Document Loader

↓

Text Extraction

↓

OCR (Only if scanned)

↓

Text Cleaning

↓

Knowledge Formatting

↓

Knowledge Validation

↓

Knowledge Repository
```

---

# CHAPTER 5

# Dataset Pipeline

After processing documents, the system prepares the training dataset.

```
Knowledge Repository

↓

Dataset Builder

↓

Training Dataset

↓

Validation Dataset

↓

Testing Dataset
```

Recommended formats

```
train.txt

validation.txt

test.txt
```

or

```
dataset.jsonl
```

---

# CHAPTER 6

# Tokenizer

## Purpose

Convert words into numerical tokens understood by the Transformer model.

Example

```
Student

↓

Token 521
```

```
Registrar

↓

Token 894
```

```
Scholarship

↓

Token 1092
```

Recommended tokenizer

```
SentencePiece
```

---

# CHAPTER 7

# Base Model

```
TinyLlama 1.1B
```

↓

Already understands

✔ English

✔ Grammar

✔ Context

✔ Question Answering

↓

Needs only TMC knowledge.

---

# CHAPTER 8

# Fine-Tuning Pipeline

```
TinyLlama

↓

LoRA

↓

Fine-Tuning

↓

Merge

↓

TMC-LM
```

Knowledge Sources

```
Student Handbook

Faculty Manual

Registrar Manual

Accounting Manual

HR Manual

Policies

Vision

Mission

Academic Programs

Research Manual

Extension Manual
```

---

# CHAPTER 9

# Evaluation

Testing Questions

```
What is the Vision of TMC?

What programs are offered?

What are the admission requirements?

Who manages the Registrar Office?

What is the philosophy of TMC?
```

Evaluation Metrics

```
Accuracy

Validation Loss

Hallucination Rate

Response Quality

Context Understanding

Answer Consistency
```

---

# CHAPTER 10

# Continuous Knowledge Expansion

One of the key features of TMC-LM is continuous knowledge acquisition.

Whenever a new institutional document becomes available, the system updates the model through incremental fine-tuning.

Example

```
Student Handbook 2026

↓

Knowledge Processing

↓

Fine-Tune

↓

TMC-LM v1.1
```

---

# CHAPTER 11

# Model Versioning

Every successful training process creates a new model version.

```
TMC-LM v1.0

↓

Student Handbook
```

```
TMC-LM v1.1

↓

+ Faculty Manual
```

```
TMC-LM v1.2

↓

+ Registrar Manual
```

```
TMC-LM v2.0

↓

+ Academic Year 2027 Documents
```

This approach ensures that institutional knowledge evolves while maintaining traceability between model versions.

---

# CHAPTER 12

# Future Development

Planned enhancements include

* Multi-language support (English, Filipino, Cebuano)
* Voice interaction
* Image-based document understanding
* Institutional search integration
* Incremental learning
* Automatic document validation
* Knowledge version management

---

# Overall System Workflow

```text
Official Documents
(PDF, DOCX, XLSX, TXT, CSV, JSON)
            │
            ▼
     File Type Detector
            │
            ▼
     Document Loader
            │
            ▼
     Text Extraction
            │
            ▼
 OCR (Only if Scanned)
            │
            ▼
      Text Cleaning
            │
            ▼
   Knowledge Formatter
            │
            ▼
   Knowledge Validator
            │
            ▼
  Knowledge Repository
            │
            ▼
     Dataset Builder
            │
            ▼
      SentencePiece
      (Tokenizer)
            │
            ▼
     TinyLlama 1.1B
     (Base Model)
            │
            ▼
    LoRA Fine-Tuning
            │
            ▼
      Merge Weights
            │
            ▼
         TMC-LM
            │
            ▼
        Evaluation
            │
            ▼
     Versioned Model
      (TMC-LM v1.0)
```
