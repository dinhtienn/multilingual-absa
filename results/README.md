# Kết Quả Thí Nghiệm

Thư mục này mô tả quy ước lưu và tổng hợp kết quả thí nghiệm. Hiện chưa có file CSV kết quả được đưa lên GitHub; các bảng dưới đây ghi lại những kết quả đã được sử dụng trong tài liệu và notebook.

## Quy Ước File Kết Quả

Khi xuất CSV từ notebook, nên đặt tên theo notebook nguồn và thiết lập chạy:

| File đề xuất | Notebook nguồn | Mô tả |
|--------------|----------------|-------|
| `mt5_baseline_en.csv` | `mt5_baseline.ipynb` | Kết quả mT5 trên English. |
| `mt5_baseline_vi.csv` | `mt5_baseline.ipynb` | Kết quả mT5 trên Vietnamese. |
| `mt5_baseline_zh.csv` | `mt5_baseline.ipynb` | Kết quả mT5 trên Chinese. |
| `gemma_zero_shot.csv` | `gemma_zero_shot.ipynb` | Kết quả Gemma 2 zero-shot. |
| `qwen_zero_shot.csv` | `qwen_zero_shot.ipynb` | Kết quả Qwen 2.5 zero-shot. |
| `llama_zero_shot.csv` | `llama_zero_shot.ipynb` | Kết quả Llama 3.1 zero-shot. |
| `gemma_fine_tune.csv` | `gemma_fine_tune.ipynb` | Kết quả Gemma 2 sau LoRA fine-tuning. |
| `qwen_fine_tune.csv` | `qwen_fine_tune.ipynb` | Kết quả Qwen 2.5 sau LoRA fine-tuning. |
| `llama_fine_tune.csv` | `llama_fine_tune.ipynb` | Kết quả Llama 3.1 sau LoRA fine-tuning. |

## Format CSV Khuyến Nghị

| Cột | Mô tả |
|-----|-------|
| `language` | Ngôn ngữ: `en`, `vi`, `zh`. |
| `domain` | Domain của mẫu nếu có. |
| `text` | Câu đánh giá gốc. |
| `true_labels` | Nhãn ground truth, nên lưu dạng JSON. |
| `predicted_labels` | Nhãn mô hình dự đoán, nên lưu dạng JSON. |
| `model` | Tên model hoặc checkpoint. |
| `setting` | `zero-shot`, `lora`, `seq2seq`, ... |

## Bảng Kết Quả Ghi Nhận

### mT5 Baseline Trên Clean-All Split

| Language | Precision | Recall | F1 |
|----------|----------:|-------:|---:|
| EN | 0.3806 | 0.3389 | 0.3585 |
| VI | 0.3242 | 0.2670 | 0.2928 |
| ZH | 0.4113 | 0.3463 | 0.3760 |
| **Macro Avg** | **0.3720** | **0.3174** | **0.3424** |

### LLM Comparison

Bảng này giữ các kết quả LLM đã ghi nhận để phục vụ phân tích. Khi chạy lại thí nghiệm, cần cập nhật kết quả theo output mới nhất và ghi rõ split dữ liệu.

| Model | Setting | EN F1 | VI F1 | ZH F1 | Avg F1 |
|-------|---------|------:|------:|------:|-------:|
| Gemma 2 9B Instruct | Zero-shot | 0.230 | 0.210 | 0.210 | 0.217 |
| Qwen 2.5 7B Instruct | Zero-shot | 0.160 | 0.120 | 0.150 | 0.143 |
| Llama 3.1 8B Instruct | Zero-shot | 0.080 | 0.100 | 0.030 | 0.070 |
| Gemma 2 9B Instruct | LoRA fine-tuning | 0.487 | 0.482 | 0.379 | 0.449 |
| Qwen 2.5 7B Instruct | LoRA fine-tuning | **0.576** | **0.536** | **0.414** | **0.509** |
| Llama 3.1 8B Instruct | LoRA fine-tuning | 0.446 | 0.417 | 0.372 | 0.412 |

## Diễn Giải

- Zero-shot prompting cho thấy năng lực suy luận theo chỉ dẫn của LLM, nhưng chưa ổn định với đầu ra có cấu trúc chặt chẽ.
- LoRA fine-tuning cải thiện đáng kể F1-score so với zero-shot, đặc biệt với Qwen 2.5 7B Instruct.