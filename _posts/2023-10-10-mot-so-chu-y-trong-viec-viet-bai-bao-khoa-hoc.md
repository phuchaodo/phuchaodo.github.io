---
title: '[Note] Một số chú ý trong việc viết bài báo khoa học'
date: 2023-10-10
permalink: /posts/2023/10/10/mot-so-chu-y-trong-viec-viet-bai-bao-khoa-hoc/
tags:
  - paper
  - writing
  - science
  - howto write
---

Một số kinh nghiệm và chú ý trong quá trình viết bài báo khoa học

Làm sao để viết một bài thuyết trình tốt
======

* Dùng phông chữ Sans-Serif: Arial, Calibri, Helvetica, …
* Dùng cỡ chữ thật to: tầm 20-36 points.
* Màu: hoặc là chữ màu đen trên nền sáng, hoặc là chữ sáng trên nền đen (Jobs thích cái này).
* Tiêu đề slide: một câu quan trọng trả lời cho câu hỏi “slide này để làm gì?”
* Mỗi slide chỉ trình bày 1 ý (hay vấn đề). Có thể 2 ý nếu liên quan. Không nhiều hơn.
* Chỉ trình bày những nội dung sẽ được đề cập. (Nếu không định trình bày nó thì xóa đi).
* Logo trường/công ty, ngày trình bày: ở slide đầu tiên thôi. Không nên để ở tất cả slides.
* Rộng rãi với không gian trống. Một ý tưởng sai lầm rất thông dụng là chèn chữ/dữ liệu/phương trình/hình ảnh bao trùm hết slide. Đó là sai lầm. Có slide chỉ cần 1 cái hình, hoặc dăm ba chữ là đủ. Cái quan trọng là bạn chém gió kìa.
* Tránh việc dùng các hiệu ứng transition. Thường thì chúng không giúp ích gì cả. Microsoft làm ra PowerPoint rồi thêm mấy cái màu mè này để bán thôi. Chúng ta không phải dùng chúng nếu không vì một mục đích nào đó.
* Hạn chế tối đa bullet points. Đặc biệt là chỉ 1 bullet point!
* Nếu phải dùng danh sách (list) thì: không nên quá nhiều item, và tiết lộ danh sách của bạn từng cái một. Chỉ khi nào nói xong về item nào đó thì tiết lộ (dùng animation của software) item tiếp theo.
* Hình ảnh phải đẹp, rõ ràng, chữ trên hình phải to rõ. Không đánh số hình, bảng, phương trình; đây không phải là bài báo!
* Hạn chế viết nguyên câu. Dùng cái gọi là headlines thôi.
* KHÔNG VIẾT HOA TẤT CẢ CÁC KÝ TỰ! (như câu này) vì khó đọc, và mất lịch sự (kiểu như người nói muốn hét vào người nghe).
* Không nên tin vào nguyên tắc “một bài thuyết trình 15 phút thì khoảng 15 slides”. Slides là do bạn làm, bạn có thể có 50 slides, miễn sao bạn có thể nói trong khoảng thời gian cho phép. Có vài slides chỉ có một câu hay một cái hình thôi.
* Nếu bạn muốn người nghe chú ý lắng nghe bạn nói thì có khi chỉ cần 1 slide trống. Không có gì để cho khán giả đọc, nhìn, họ phải nghe bạn thôi!
* Nhiều người viết rất nhiều thông tin vào slides vì lí do đơn giản: không nhớ nổi. Lúc xưa mình cũng làm slides như vậy. Sau này nghĩ lại thì thấy, bài thuyết trình là về 1 chủ đề người trình bày rành. Vậy thì mọi thứ (về căn bản) là đã nằm trong đầu họ rồi chứ. Cần chi slides để nhắc?
* Không nên bỏ vào slides quá nhiều kiến thức. Bạn không cần phải chứng tỏ là mình là người hiểu biết rộng. Bạn chỉ có khoảng 20 phút thôi, nên tập trung vào ý chính.
* Cũng vì lí do thời gian hạn hẹp, bạn không nên bỏ quá nhiều chi tiết kỹ thuật (như phương trình hay công thức). Sau khi nghe xong một buổi thuyết trình nếu người nghe có thể hiểu được cái hồn của câu chuyện (vấn đề, cách giải quyết) thì đó đã là một thành công lớn. Chi tiết không thể nào truyền đạt trong vòng vài chục phút.

Một ví dụ về việc viết summary và slide cho paper
======


### Link bài báo

* Liu, Zhiguo, et al. "Research on Satellite Network Traffic Prediction Based on Improved GRU Neural Network." Sensors 22.22 (2022): 8678.

### Động lực viết bài báo này

Thứ nhất, động lực (motivation) gì để bạn viết bài báo này??
* Xu hướng liên quan đến tên bài báo
* Một số hạn chế
* Nên có động lực để viết bài báo này.

### Đóng góp chính của bài báo

Thứ hai, các đóp góp chính của bài báo là gì???

* Đóng góp 1: the correlation characteristics of satellite network traffic are fully considered, and the nonlinear time dynamic correlation is obtained by using a gating unit to avoid gradient disappearance or gradient explosion during training.
* Đóng góp 2: In the coding and decoding stages of the GRU network, an attention mechanism is introduced, and multiple intermediate vectors are added to uniformly process the time series and input information of the intermediate vectors at the current movement.
* Đóng góp 3: Particle swarm optimization algorithm is used to adjust the hyperparameters of the neural network.

### Nội dung chính của bài báo

Thứ ba, các nội dung về nội dung chính: 

+ Nội dung 1: Definition and Model of the Satellite Traffic Forecast Problem

+ Nội dung 2: Traffic Prediction Method of the AT-GRU Satellite Network

    * Nội dung 2.1: Design of Coding Unit Based on Attention Mechanism
    * Nội dung 2.2: Design of Decoding Unit Based on Attention Mechanism
    * Nội dung 2.3: PSO Algorithm for GRU Hyperparameter Selection Problem

### Phần thực nghiệm và kết quả

Thứ tư, về phần thực nghiệm và kết quả:

* Giới thiệu về dataset
* Môi trường thực nghiệm
* Evaluation Index and Parameter Setting of Simulation
* Comparison and Analysis of Simulation Results of Different Algorithms
* Convergence Analysis (Phân tích tính hội tụ của mô hình)
* Model Complexity Analysis (phân tích độ phức tạp của mô hình)

### Trình bày kết luận 

* Cuối cùng, kết luận của bài báo là gì??


Phương pháp đọc và tiếp cận việc viết báo
======

### Xác định loại làm nghiên cứu

* Nghiên cứu cái đã có
* Đề xuất giải pháp mới
* Kết hợp nhiều phương pháp với nhau

### Câu hỏi nghiên cứu

* Xác định vấn đề nghiên cứu
* tìm và đọc các bài báo liên quan đến câu hỏi nghiên cứu

### Cách đọc một paper

* survey và overview (đọc từ mới đến cũ)
* research cụ thể một vấn đề (giải pháp cụ thể) thì nên đọc (từ cũ đến mới)

### Khi dọc bài báo thì làm gì?

* Note các đoạn văn, tham khảo để viết lại mới.
* Đọc và hiểu abstract cho lượt đầu, nhưng absstract thì viết cuối cùng.
* Đọc background cơ bản
* Cuối phần introduction sẽ có phần nói tác giả sẽ làm gì trong bài báo và đóng góp gì? hiểu hơn về câu hỏi nghiên cứu của họ?
* Related work: chỉ ra nghiên cứu nào liên quan đến bài báo hiện tại –> liệt kê các vấn đề trước đó –> đọc phần related word để hiểu được câu hỏi “research question”

### Một ví dụ về việc viết báo

* Cách viết introduction: liệt kê vấn đề xung quanh keywords, giải thích các vấn đề, mỗi vấn đề sẽ có những đóng góp liên quan
* Viết related work: tham khảo phần cuối introduction của các bài báo khác? note lại cách viết của các bài báo họ đã làm? kết quả của họ là gì? thiếu sót cái gì?
* Phần methods: dataset gì? cách xử lý dữ liệu thế nào? họ dùng phương pháp nào cho thực nghiệm? luồng thực hiện ra sao?
* Phần thực nghiệm: thời gian, độ chính xác, mô tả kết quả, phân tích kết quả và cuối cùng là giải thích kết quả. Cách họ improve model hoặc fine tuning model ra sao?
* Phần kết luận: đóng góp chính bài báo là gì? Công việc tương lai sẽ làm gì?
* Viết abstract: Nó được viết cuối cùng? giới thiệu phương pháp và kết quả thực hiện.

Một ví dụ về cách viết literature review cho paper
======

### Tên bài báo
* Comparative Performance of ESRGAN, LDM, and SwinIR: Super-Resolution for Unmanned FPV Video Compression

### Viết nội dung thể hiện Literature review

Từ tiêu đề, xác định các keywords để viết content tương ứng.

* content 1: Video compression techniques
* content 2: Super-resolution techniques for robotics applications

Từ đó mình sẽ liệt kê các gaps (dựa vào content đã liệt kê.

Gaps in literature

* Super-resolution techniques have not been optimized for robotics FPV video, which differs from conventional datasets.
* The comparative analysis focused on robotics video compression is limited despite distinct challenges.
* Joint compression and super-resolution methods tailored for robotics remain relatively unexplored.
* Trade-offs between model accuracy, speed, and compression ratio are not well-studied for robotics.

Sau đó, đưa ra nhận định: 

* This research conducts a comprehensive comparative study of state-of-the-art super-resolution techniques for robotics FPV video compression using real-world unmanned system data. Both model improvement and evaluation are geared towards the characteristics and constraints of practical robotics applications.

Một phần khác
======