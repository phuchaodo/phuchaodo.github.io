---
title: '[Note] Step by step for writing paper'
date: 2024-01-27
permalink: /posts/2024/01/27/step-by-step-for-writing-paper/
tags:
  - paper
  - writing
  - science
  - howto write
---

Các bước tiếp cận để viết một bài báo khoa học

Bước 01 (Xác định tên của paper)
======

* Tên một paper có cấu trúc sau: Phương pháp/Contribution+Mục đích+Lĩnh vực

* Xác định "lĩnh vực" liên quan (thường có 3 keyword liên quan)
* search các paper liên quan đến nó (khoảng 40 bài đó)
* sàng lọc, chọn những bài thật sự liên quan (khoảng còn 15 bài)

Các outline phổ biến:

* Introduction
* Related work
* Methods
* Experiment
* Result
* Conclusion
* Abstract

Bước 02 (Introduction)
======

* Giới thiệu "lĩnh vực" đang cần viết (resource allocation, AI, satellite network)
* Mỗi "lĩnh vực" viết một đoạn (phải có bài báo liên quan đến mục đó)
* Nêu động lực viết paper đó (động lực, contribution của mình là gì?)

Bước 03 (Related work)
======

* Giới thiệu "lĩnh vực" đang cần viết (resource allocation, AI, satellite network)
* Mỗi "lĩnh vực" nên có một số ( > 3) bài báo liên quan. Mỗi bài báo viết xem là bài đó làm được gì? Ưu điểm và nhược điểm?
* Ví dụ: Bài báo A, họ dùng làm gì? làm phương pháp nào? dùng tập dữ liệu nào? đạt kết quả ra sao? ưu điểm, nhược điểm là gì?
* Tóm tắt thêm lần nữa là sẽ làm gì trong bài báo.

Bước 04 (Methods)
======

* quá trình thu thập dữ liệu hoặc sử dụng public data set
* Phải vẽ luồng (figure) cho chi tiết và giải thích các thành phần và tham số.
* luồng chính
* các algorithm liên quan hoặc framework sẽ dùng
* chi tiết p2 thực hiện.


Bước 05 (Experiment)
======

* gồm bao nhiêu thực nghiệm, mỗi thực nghiệm gồm những gì?
* cách thức thực hiện cho từng thực nghiệm.
* kết quả đạt được ra sao? (sơ lược)


Bước 06 (Result)
======

* kết quả đạt được sau mỗi exp thế nào?
* mô tả và diễn giải kết quả đạt được
* tóm gọn kết quả tốt thế nào?
* so sánh kết quả đạt được?

Bước 07 (Conclusion)
======

* trình bày lại vấn đề?
* bài báo thực hiện những gì?
* so sánh đánh giá kết quả
* công việc tương lai sẽ làm gì?

Bước 08 (Abstract)
======

* vấn đề của bài toán
* dùng phương pháp gì? giải quyết được vấn đề gì?
* kết quả đạt được là gì?

Các keywords (5) để làm nổi bật cho paper.


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

### Khi đọc bài báo thì làm gì?

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



Tư duy để liệt kê danh sách các bài toán nghiên cứu cụ thể
======

### Về nguyên tắc để liệt kê các bài báo cần viết

* Trước hết, cần phải có một luồng làm việc phù hợp với chủ đề của mình.
* Tiếp theo, phát thảo chi tiết của luồng đã có. Ví dụ như input là gì? Process gồm những phần nào? Và Output gồm những phần nào?
* Sau đó, với mỗi phần mình sẽ định hình những kỹ thuật nào sẽ dùng và dùng thế nào?
* Nên: Tìm các phương pháp cho từng process nhỏ (process1 thì dùng kỹ thuật 1, process2 thì dùng kỹ thuật 2,…)
* Tiếp đến, tìm cách để optimize có thể là fine tuning hoặc các thuật toán tối ưu hoặc heuristics.
* Cái quan trọng: Vẽ được luồng tổng quan hoạt động để làm việc.
* Cuối cùng sẽ tách nhỏ ra: Bài toán 1 thì làm ở chỗ process1, Bài toán 2 thì nên làm ở process2.
* Tùy vào phạm vi của bài toán mà có thể xác định submit ở conf hoặc journal hoặc ranking phù hợp.

### Về nguyên tắc viết báo

* Sử dụng công cụ hỗ trợ
* Xác định từng outline tương ứng cho từng bài toán cụ thể
* Ở mỗi outline, xác định từng phần nhỏ nhỏ, để viết làm sao người đọc vào có thể hiểu ngay được.
* Chia nhỏ từng phần cho dễ đọc và dễ focus hơn. Lúc này bài trình bày sẽ rõ ràng hơn rất nhiều.


Việc hướng dẫn sinh viên để tham gia nckh
======

Cần xác định một số nội dung sau: 

* tên đề tài
* tài liệu tham khảo
* viết review (abstract hay liệt kê các method)
* cài đặt, so sánh
* các contribution (có thể fe, preprocessing, …, chia nhỏ, pretrained)
* viết nháp theo format (latex) và trình bày slide

Viết báo thì cần lưu ý và nhớ rằng:

* Cần phải có nội dung, sau đó mới có bảng hoặc hình ảnh xuất hiện ở phía dưới. (rat quan trong)
* Dùng pdf lưu ảnh để load file dữ liệu
* Thảo luận về kết quả chủ yếu mô tả và không phân tích sâu hoặc giải thích kết quả. Hơn nữa, thiếu hỗ trợ trực quan để minh họa kết quả. —> giải thích sâu hơn về kết quả, giải thích tại sao chúng xảy ra và ý nghĩa của chúng trong ngữ cảnh của mục tiêu nghiên cứu.


Cách viết các bài analysis và survey "nhẹ nhàng" thì làm thế nào? 
======

* Ví dụ tên bài báo: An Analysis of AI Techniques to Improve Quality of Service in Low Earth Orbit Satellite Networks

### Goals

* To understand the role of AI in enhancing QoS in LEO Satellite Networks.
* To evaluate the effectiveness of different AI techniques in improving specific QoS parameters.
* To provide a comparative analysis of recent advancements in AI for satellite networks.
* Research questions → 2 research questions are fine

### Introduction

* Background on Low Earth Orbit (LEO) Satellite Networks
* Importance of Quality of Service (QoS) in LEO Networks
* Role of AI in enhancing QoS

### Methodology
* AI techniques to improve QoS in LEO satellite
* Criteria for evaluating QoS improvements 
* Criteria for evaluating AI techniques (accuracy, complex model, computional time)
* Comparative Analysis: Lập bảng so sánh (study, ảnh hưởng qos gì (từ khóa chính), liên quan đến AI nào?, các tiêu chí về đánh giá mô hình AI)

### Conclusion

### Note

* Bất cứ đoạn nào viết dài thì phải có ref (vì mình không tự viết được).
* Cần tối thiểu 10 ref
* Bảng so sánh cần thể hiện (paper nào, ảnh hưởng đến qos - nội dung chính, liên quan AI nào (kỹ thuật chính), các tiêu chí đánh giá mô hình AI)


Chi tiết bài các loại review đi
======
Chưa viết gì cả.



Hết.
