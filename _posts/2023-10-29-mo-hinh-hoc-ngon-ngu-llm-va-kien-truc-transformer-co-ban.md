---
title: '[Note] Mô hình học ngôn ngữ (LLM) và kiến trúc transformer cơ bản'
date: 2023-10-29
permalink: /posts/2023/10/29/mo-hinh-hoc-ngon-ngu-llm-va-kien-truc-transformer-co-ban/
tags:
  - research
  - writing
  - science
  - LLM
---

Mô hình học ngôn ngữ (LLM) và kiến trúc transformer cơ bản

Khái niệm cơ bản về Language Learning Model (LLM)
======

## Language Learning Model (LLM) 

Nó là một loại mô hình học máy được thiết kế đặc biệt để hiểu và tạo ra ngôn ngữ tự nhiên. Đây là một phân loại lớn bao gồm nhiều mô hình khác nhau, và một ví dụ tiêu biểu là GPT-3 (Generative Pre-trained Transformer 3) của OpenAI.

### Hoạt động của Language Learning Model:

1. **Kiến trúc dựa trên Transformer:**
   - LLM thường sử dụng kiến trúc Transformer. Đây là một loại mô hình học sâu đặc biệt được thiết kế cho xử lý dữ liệu ngôn ngữ tự nhiên.

2. **Quá trình huấn luyện:**
   - LLM được huấn luyện trên một lượng lớn dữ liệu văn bản từ internet. Quá trình này bao gồm hai giai đoạn chính:
     - Pre-training: Mô hình học cách dự đoán từ tiếp theo trong các chuỗi văn bản. Điều này giúp nó học cách sắp xếp và tạo ra văn bản tự nhiên.
     - Fine-tuning: Mô hình có thể được đào tạo thêm trên một tập dữ liệu nhỏ hơn, có thể được tùy chỉnh để thực hiện các tác vụ cụ thể như dịch ngôn ngữ, trả lời câu hỏi, và nhiều tác vụ khác.

3. **Tokenization:**
   - Văn bản được chia thành các đơn vị nhỏ gọi là "token". Mỗi token có thể là một từ hoặc một phần của từ. Điều này giúp mô hình phân tích và hiểu về các thành phần nhỏ hơn của văn bản.

4. **Cơ chế tự chú ý (Self-Attention Mechanism):**
   - Cơ chế tự chú ý cho phép mô hình tập trung vào các phần quan trọng của văn bản. Nó học cách xác định mối quan hệ giữa các phần tử trong chuỗi.

### Một số mô hình LLM tiêu biểu

1. **GPT-3 (Generative Pre-trained Transformer 3)**:
   - Tên tổ chức phát triển: OpenAI.
   - GPT-3 là một trong những mô hình LLM nổi tiếng nhất, với 175 tỷ tham số. Nó có khả năng tạo ra văn bản tự nhiên, trả lời câu hỏi, dịch ngôn ngữ, và nhiều tác vụ ngôn ngữ khác.

2. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - Tên tổ chức phát triển: Google.
   - BERT tập trung vào việc hiểu ngữ cảnh cả hai chiều của câu. Điều này có ích trong nhiều tác vụ ngôn ngữ, bao gồm dịch ngôn ngữ, phân loại văn bản, và nhiều tác vụ khác.

3. **T5 (Text-To-Text Transfer Transformer)**:
   - Tên tổ chức phát triển: Google.
   - T5 đưa vào một phương pháp tiếp cận độc đáo, trong đó mọi tác vụ ngôn ngữ được biểu diễn dưới dạng tác vụ "text-to-text". Điều này bao gồm các tác vụ như dịch ngôn ngữ, phân loại văn bản, và nhiều tác vụ khác.

4. **XLNet**:
   - Tên tổ chức phát triển: Google và Carnegie Mellon University.
   - XLNet cũng là một mô hình rất mạnh sử dụng kiến trúc Transformer và phương pháp học sâu tự chú ý.

5. **RoBERTa (A Robustly Optimized BERT Pretraining Approach)**:
   - Tên tổ chức phát triển: Facebook AI Research (FAIR).
   - RoBERTa là một biến thể của BERT được tối ưu hóa để đạt hiệu suất tốt hơn trên một loạt các tác vụ ngôn ngữ.


## Kiến Trúc Transformer: 

1. **Kiến Trúc Tổng Quan**:
   - Kiến trúc Transformer được thiết kế để xử lý và hiểu các chuỗi dữ liệu như văn bản.
   - Nó bao gồm nhiều tầng, mỗi tầng được thiết kế để học các đặc trưng từ dữ liệu đầu vào.

2. **Tầng Tự Chú Ý (Self-Attention Layer)**:

   - **Tự Chú Ý Đa Đầu (Multi-Head Attention)**:
     - Mỗi tầng tự chú ý chia thành nhiều đầu. Mỗi đầu tự chú ý tập trung vào một phần khác nhau của dữ liệu đầu vào.
     - Mỗi đầu này học cách định trọng số cho các phần khác nhau của chuỗi.

   - **Biểu Diễn Vector Kích Hoạt (Query, Key, Value)**:
     - Mỗi thành phần của chuỗi (từ) được biểu diễn bởi ba vector: Query, Key và Value. Đây là các biểu diễn học được từ dữ liệu huấn luyện.

   - **Tính Toán Sự Tương Tác (Attention Score)**:
     - Sự tương tác (attention score) giữa các từ trong chuỗi được tính bằng cách nhân vô hướng giữa vector Query của một từ và vector Key của các từ khác.
     - Kết quả này thể hiện mức độ quan trọng của từ đó đối với từ đang xem xét.

   - **Tính Tổng Cộng Cuối Cùng (Final Aggregation)**:
     - Các attention score sau đó được sử dụng để kết hợp thông tin từ các từ khác nhau trong chuỗi.

3. **Mạng Nơ-ron Tiếp Thị (Feedforward Neural Network)**:

   - Mỗi đầu ra từ tầng tự chú ý sau đó được đưa qua một mạng nơ-ron tiếp thị. Đây là nơ-ron tiếp thị cơ bản với các kết nối đầy đủ và các hàm kích hoạt như ReLU.

   - Mạng này giúp biến đổi và kết hợp thông tin từ các đầu vào.

4. **Chuẩn Hóa và Kết Nối Dư (Normalization and Residual Connections)**:

   - Mỗi tầng (bao gồm cả tầng tự chú ý và tầng nơ-ron tiếp thị) có một bước chuẩn hóa và một kết nối dư (residual connection). Điều này giúp đối mặt với vấn đề biến mất gradient trong quá trình huấn luyện.

5. **Xếp Các Tầng (Stacking of Layers)**:

   - Các tầng này được xếp chồng lên nhau, cho phép mô hình học các biểu diễn ngữ cảnh phức tạp từ các tầng trước đó.

6. **Phép Mã Hóa Vị Trí (Positional Encoding)**:

   - Vì Transformer không nhận biết thứ tự tự nhiên của các từ, một phép mã hóa vị trí (positional encoding) được thêm vào để mô hình biết vị trí của các từ trong chuỗi.

## Ví dụ về sử dụng mô hình LLM (pretrained)

Để sử dụng một mô hình ngôn ngữ học sâu (LLM) dựa trên kiến trúc Transformer đã được huấn luyện trước bằng PyTorch, bạn có thể sử dụng thư viện `transformers` của Hugging Face. Đây là một ví dụ về cách sử dụng một mô hình GPT-2 (một biến thể của Transformer) đã được huấn luyện trước:

```python
# Bước 1: Cài đặt thư viện transformers
!pip install transformers

# Bước 2: Import các thư viện cần thiết
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Bước 3: Khởi tạo mô hình và tokenizer
model_name = 'gpt2'  # Chọn tên mô hình
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Bước 4: Sử dụng mô hình để sinh văn bản
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Bước 5: Tạo đầu ra từ mô hình
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_return_sequences=5)

# Bước 6: Giải mã đầu ra về văn bản
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
```

Ở đây, chúng ta sử dụng mô hình GPT-2 đã được huấn luyện trước. Chúng ta nhập và cài đặt thư viện `transformers`, sau đó tải và sử dụng mô hình và tokenizer. 

Cuối cùng, chúng ta sử dụng mô hình để sinh văn bản dựa trên đầu vào.

Lưu ý rằng, bạn có thể thay đổi mô hình và tokenizer bằng các biến thể khác của Transformer trong `transformers` library tùy thuộc vào mục tiêu cụ thể của bạn.


## Sử dụng PyTorch để triển khai một mô hình phân loại văn bản sử dụng kiến trúc Transformer. 

Trong ví dụ này, chúng ta sẽ sử dụng mô hình `BertForSequenceClassification` từ thư viện Transformers của Hugging Face.

1. **Bước 1: Cài đặt thư viện**

   Đầu tiên, bạn cần cài đặt thư viện `transformers`:

   ```python
   !pip install transformers
   ```

2. **Bước 2: Import các thư viện cần thiết**

   ```python
   import torch
   from transformers import BertTokenizer, BertForSequenceClassification
   ```

3. **Bước 3: Tải mô hình và tokenizer**

   ```python
   model_name = 'bert-base-uncased'
   tokenizer = BertTokenizer.from_pretrained(model_name)
   model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Số nhóm/phân lớp
   ```

4. **Bước 4: Tiền xử lý dữ liệu**

   Đầu vào của mô hình phân loại là chuỗi văn bản, bạn cần mã hóa chuỗi thành dạng phù hợp với mô hình.

   ```python
   input_text = "This is an example sentence."
   input_ids = tokenizer.encode(input_text, add_special_tokens=True)
   input_ids = torch.tensor(input_ids).unsqueeze(0)  # Thêm chiều batch
   ```

5. **Bước 5: Tạo đầu ra từ mô hình**

   ```python
   with torch.no_grad():
       outputs = model(input_ids)
   logits = outputs.logits
   ```

   Ở đây, `logits` chứa các dự đoán trước khi áp dụng hàm softmax.

6. **Bước 6: Đánh giá kết quả**

   Để đánh giá kết quả, bạn có thể sử dụng hàm softmax để chuyển đổi các dự đoán thành xác suất và chọn lớp có xác suất cao nhất.

   ```python
   import torch.nn.functional as F

   probs = F.softmax(logits, dim=-1)
   predicted_label = torch.argmax(probs, dim=-1).item()
   ```


Hết.
