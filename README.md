# Multilingual Aspect-Based Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/Hugging%20Face-PEFT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/peft)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Tóm Tắt

Nghiên cứu này khảo sát bài toán **Multilingual Aspect-Based Sentiment Analysis (M-ABSA)** trên ba ngôn ngữ English (`en`), Vietnamese (`vi`) và Chinese (`zh`). Dựa trên hướng tiếp cận của bộ dữ liệu M-ABSA, nhiệm vụ được mô hình hóa như một bài toán sinh văn bản có cấu trúc: từ một câu đánh giá, hệ thống cần trích xuất các bộ ba gồm thực thể, khía cạnh và sắc thái cảm xúc.

```text
(entity, category, sentiment)
```

Ví dụ:

```text
The screen is beautiful but the battery is poor.
=> (screen, LAPTOP#DISPLAY, positive), (battery, LAPTOP#BATTERY, negative)
```

Trọng tâm của nghiên cứu là xây dựng baseline bằng mô hình encoder-decoder đa ngôn ngữ `google/mt5-small`, sau đó so sánh với các mô hình ngôn ngữ lớn dạng instruction-tuned trong hai thiết lập: **zero-shot prompting** và **LoRA fine-tuning**.

## Bối Cảnh Nghiên Cứu

Aspect-Based Sentiment Analysis (ABSA) không chỉ xác định cảm xúc chung của văn bản, mà còn yêu cầu mô hình nhận diện cảm xúc theo từng khía cạnh cụ thể. Trong bối cảnh đa ngôn ngữ, bài toán trở nên khó hơn do khác biệt về hình thái, cú pháp, cách biểu đạt cảm xúc và phân bố nhãn giữa các ngôn ngữ.

Bộ dữ liệu M-ABSA cung cấp một benchmark đa ngôn ngữ, đa miền cho bài toán này. Nghiên cứu sử dụng dữ liệu từ ba ngôn ngữ `en`, `vi`, `zh` nhằm đánh giá khả năng tổng quát hóa của các mô hình nền đa ngôn ngữ và các LLM hiện đại trong thiết lập ít phụ thuộc vào đặc trưng thủ công.

## Mục Tiêu Và Câu Hỏi Nghiên Cứu

Nghiên cứu tập trung vào ba mục tiêu chính:

1. Xây dựng baseline có khả năng tái lập cho M-ABSA bằng mô hình `google/mt5-small` theo hướng text-to-text.
2. Đánh giá năng lực zero-shot của các LLM instruction-tuned đối với tác vụ trích xuất bộ ba ABSA đa ngôn ngữ.
3. Phân tích tác động của LoRA fine-tuning trong việc cải thiện độ chính xác của LLM trên cùng tác vụ.

Các câu hỏi nghiên cứu tương ứng:

- Mô hình encoder-decoder đa ngôn ngữ cỡ nhỏ có thể đóng vai trò baseline ổn định cho M-ABSA hay không?
- Các LLM instruction-tuned thể hiện như thế nào khi chỉ dùng prompt, không cập nhật tham số?
- LoRA fine-tuning cải thiện hiệu năng đến mức nào so với zero-shot prompting?
- Hiệu năng giữa English, Vietnamese và Chinese có khác biệt đáng kể hay không?

## Dữ Liệu

- **Nguồn**: M-ABSA Dataset trên Kaggle.
- **Tổng số mẫu sử dụng**: 26,854 câu.
- **Ngôn ngữ**: English, Vietnamese, Chinese.
- **Domain**: coursera, food, hotel, laptop, phone, restaurant, sight.
- **Số category**: 241.
- **Định dạng dữ liệu**:

```text
text####[(entity, category, sentiment), ...]
```

## Phương Pháp

### Baseline mT5

Baseline sử dụng `google/mt5-small`, một mô hình encoder-decoder đa ngôn ngữ, để chuyển bài toán M-ABSA thành tác vụ sinh chuỗi đầu ra có cấu trúc. Mỗi câu đầu vào được ánh xạ sang danh sách các bộ ba `(entity, category, sentiment)`.

Thiết lập này đóng vai trò mốc so sánh chính vì có chi phí huấn luyện thấp hơn LLM và phù hợp với bài toán sinh văn bản có cấu trúc.

### Zero-Shot Prompting

Ba LLM instruction-tuned được đánh giá trong thiết lập zero-shot. Ở thiết lập này, mô hình chỉ nhận prompt mô tả nhiệm vụ và câu đầu vào, không được fine-tune trên dữ liệu M-ABSA. Mục tiêu là đo khả năng suy luận theo chỉ dẫn và khả năng xử lý đa ngôn ngữ sẵn có của mô hình.

### LoRA Fine-Tuning

Để giảm chi phí huấn luyện LLM, nghiên cứu sử dụng LoRA/QLoRA. Phương pháp này chỉ cập nhật một số ma trận hạng thấp thay vì toàn bộ tham số mô hình, giúp fine-tune các mô hình 7B-9B trong môi trường GPU giới hạn như Kaggle T4.

## Mô Hình Thí Nghiệm

| Vai trò | Model | Nguồn | Thiết lập |
|---------|-------|-------|-----------|
| Baseline | mT5 Small | [`google/mt5-small`](https://huggingface.co/google/mt5-small) | Seq2Seq fine-tuning |
| LLM | Gemma 2 9B Instruct | [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | Zero-shot, LoRA |
| LLM | Qwen 2.5 7B Instruct | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Zero-shot, LoRA |
| LLM | Llama 3.1 8B Instruct | [`NousResearch/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct) | Zero-shot, LoRA |

Checkpoint và adapter LoRA được lưu ngoài mã nguồn, thường trong `/kaggle/working/` hoặc dưới dạng Kaggle Dataset. 

## Thiết Lập Thí Nghiệm

### mT5 Baseline

| Tham số | Giá trị |
|---------|---------|
| Base model | `google/mt5-small` |
| Objective | Text-to-text generation |
| Split | 80% train, 10% validation, 10% test |
| Epochs | 5 |
| Learning rate | `3e-4` |
| Max input length | 128 |
| Max target length | 64 |

### LLM LoRA / QLoRA

| Tham số | Giá trị |
|---------|---------|
| Quantization | 4-bit NF4 |
| LoRA rank | `r=8` |
| LoRA alpha | `16` |
| Optimizer | `paged_adamw_8bit` |
| Scheduler | cosine |
| Effective batch size | 16 |
| Max sequence length | 1536 tokens |

## Kết Quả Ghi Nhận

### Baseline mT5 Trên Clean-All Split

Notebook [mt5_baseline.ipynb](notebooks/mt5_baseline.ipynb) fine-tune `google/mt5-small` riêng theo từng ngôn ngữ trên toàn bộ `clean_all.txt`, sau đó chia lại dữ liệu theo tỉ lệ 80/10/10.

| Language | Precision | Recall | F1 |
|----------|----------:|-------:|---:|
| EN | 0.3806 | 0.3389 | 0.3585 |
| VI | 0.3242 | 0.2670 | 0.2928 |
| ZH | 0.4113 | 0.3463 | 0.3760 |
| **Macro Avg** | **0.3720** | **0.3174** | **0.3424** |

### So Sánh LLM

Bảng dưới đây tổng hợp các kết quả LLM đã ghi nhận trong quá trình thí nghiệm. Do baseline mT5 và các LLM có thể dùng split hoặc quy mô tập đánh giá khác nhau, kết quả nên được diễn giải theo từng nhóm thí nghiệm thay vì xem là so sánh tuyệt đối trực tiếp.

| Model | Setting | EN F1 | VI F1 | ZH F1 | Avg F1 |
|-------|---------|------:|------:|------:|-------:|
| Gemma 2 9B Instruct | Zero-shot | 0.230 | 0.210 | 0.210 | 0.217 |
| Qwen 2.5 7B Instruct | Zero-shot | 0.160 | 0.120 | 0.150 | 0.143 |
| Llama 3.1 8B Instruct | Zero-shot | 0.080 | 0.100 | 0.030 | 0.070 |
| Gemma 2 9B Instruct | LoRA fine-tuning | 0.487 | 0.482 | 0.379 | 0.449 |
| Qwen 2.5 7B Instruct | LoRA fine-tuning | **0.576** | **0.536** | **0.414** | **0.509** |
| Llama 3.1 8B Instruct | LoRA fine-tuning | 0.446 | 0.417 | 0.372 | 0.412 |

Kết quả cho thấy zero-shot prompting vẫn còn hạn chế với tác vụ yêu cầu đầu ra có cấu trúc chặt chẽ. LoRA fine-tuning cải thiện rõ rệt F1-score, đặc biệt với Qwen 2.5 7B Instruct. Tuy nhiên, hiệu năng trên Chinese vẫn thấp hơn so với English và Vietnamese trong nhiều thiết lập, cho thấy sự khác biệt ngôn ngữ vẫn là một thách thức đáng kể.

## Tái Lập Thí Nghiệm

### Yêu Cầu Môi Trường

- Python 3.10+.
- Kaggle Notebook hoặc Google Colab.
- GPU NVIDIA, khuyến nghị T4 16GB VRAM trở lên.
- Hugging Face token nếu model yêu cầu quyền truy cập.

### Cài Đặt Phụ Thuộc

```bash
pip install -r requirements.txt
```

### Thứ Tự Chạy Notebook

```text
1. notebooks/mt5_baseline.ipynb
2. notebooks/gemma_zero_shot.ipynb
3. notebooks/qwen_zero_shot.ipynb
4. notebooks/llama_zero_shot.ipynb
5. notebooks/gemma_fine_tune.ipynb
6. notebooks/qwen_fine_tune.ipynb
7. notebooks/llama_fine_tune.ipynb
```

Các notebook được thiết kế để chạy tuần tự từ trên xuống sau khi đã gắn dataset Kaggle `tuongmacvan/m-absb-for-ppnckh` và cấu hình `HF_TOKEN` nếu cần.

## Cấu Trúc Dự Án

```text
multilingual-absa/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── build_clean_all.py
│   └── generate_analysis.py
├── notebooks/
│   ├── mt5_baseline.ipynb
│   ├── gemma_zero_shot.ipynb
│   ├── gemma_fine_tune.ipynb
│   ├── qwen_zero_shot.ipynb
│   ├── qwen_fine_tune.ipynb
│   ├── llama_zero_shot.ipynb
│   └── llama_fine_tune.ipynb
├── results/
│   └── README.md
└── paper/
    └── README.md
```

## Tài Liệu Tham Khảo

Tài liệu tham khảo chính được tổng hợp tại [paper/README.md](paper/README.md). Nếu sử dụng nghiên cứu này, vui lòng trích dẫn dataset gốc:

```bibtex
@inproceedings{mabsa2025,
  title = {M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of EMNLP 2025},
  pages = {2530--2557},
  year = {2025},
  publisher = {Association for Computational Linguistics}
}
```

## Giấy Phép

Mã nguồn, notebook và tài liệu trong dự án được phát hành theo giấy phép MIT. Xem [LICENSE](LICENSE) để biết thêm chi tiết.