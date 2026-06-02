# Tài Liệu Tham Khảo

Thư mục này ghi lại các tài liệu tham khảo chính cho nghiên cứu **Multilingual Aspect-Based Sentiment Analysis (M-ABSA)**. Không lưu bản PDF hoặc tài liệu có bản quyền tại đây.

## Dataset Gốc

**M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis**

- **Hội nghị**: EMNLP 2025.
- **Trang**: 2530-2557.
- **Địa điểm**: Suzhou, China.
- **Nhà xuất bản**: Association for Computational Linguistics.
- **Link**: https://acl.ldc.upenn.edu/2025.emnlp-main.128/

```bibtex
@inproceedings{mabsa2025,
  title = {M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages = {2530--2557},
  year = {2025},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics}
}
```

## Dataset Và Task

- M-ABSA gồm 7 domain: coursera, food, hotel, laptop, phone, restaurant, sight.
- Các thí nghiệm sử dụng ba ngôn ngữ `en`, `vi`, `zh`.
- Đầu ra của tác vụ là danh sách bộ ba `(entity, category, sentiment)`.

## Model Nền

- mT5 Small: https://huggingface.co/google/mt5-small
- Gemma 2 9B IT: https://huggingface.co/google/gemma-2-9b-it
- Qwen 2.5 7B Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Llama 3.1 8B Instruct: https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct

## Phương Pháp

- LoRA (Low-Rank Adaptation): https://arxiv.org/abs/2106.09685
- QLoRA (Quantized LoRA): https://arxiv.org/abs/2305.14314
- SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- BitsAndBytes 4-bit quantization: https://huggingface.co/docs/bitsandbytes
- Hugging Face PEFT: https://huggingface.co/docs/peft
- Hugging Face Transformers: https://huggingface.co/docs/transformers

## Paper Liên Quan

- **Evaluating Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models** (2024): https://arxiv.org/abs/2412.12564
- **LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation** (ACL 2025): https://aclanthology.org/2025.acl-long.41/
- **Cross-lingual Aspect-Based Sentiment Analysis: A Survey on Tasks, Approaches, and Challenges** (2025): https://arxiv.org/abs/2508.09516