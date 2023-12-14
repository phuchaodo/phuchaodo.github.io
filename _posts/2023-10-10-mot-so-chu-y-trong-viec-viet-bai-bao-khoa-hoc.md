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

Một ví dụ về cách viết literature review cho paper
======

### Tên bài báo
* Comparative Performance of ESRGAN, LDM, and SwinIR: Super-Resolution for Unmanned FPV Video Compression

### Viết nội dung thể hiện Literature review

Từ tiêu đề, xác định các keywords để viết content tương ứng.

* Content 1: Video compression techniques
* Content 2: Super-resolution techniques for robotics applications

Từ đó mình sẽ liệt kê các gaps (dựa vào content đã liệt kê.

Gaps in literature:

* Super-resolution techniques have not been optimized for robotics FPV video, which differs from conventional datasets.
* The comparative analysis focused on robotics video compression is limited despite distinct challenges.
* Joint compression and super-resolution methods tailored for robotics remain relatively unexplored.
* Trade-offs between model accuracy, speed, and compression ratio are not well-studied for robotics.

Sau đó, đưa ra nhận định: 

* This research conducts a comprehensive comparative study of state-of-the-art super-resolution techniques for robotics FPV video compression using real-world unmanned system data. Both model improvement and evaluation are geared towards the characteristics and constraints of practical robotics applications.

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

Structure of a scientific article
======

### IMRAD (Introduction, Methods, Results and Discussion)

#### Introduction

* The reseach question
* The importance of the study
* Begin with a topic sentece (inverted pyramid)
* Brief summary of the issue or public health problem
* Concise review of pertinent literature
* Study approach (one sentece)
* What will your article add?
* Keep this brief with front-loading


#### Methods

* How you address your study question
* How, what, when, and where
* Recipe that others can repeat
* Type of study design; describe the intervention
* Data source
* Outcomes to be measured
* Describle analysis
* Statistical tests
* Ethical aproval

#### Results

* Detail individual included and excluded
* Demographic charateristics of the study groups
* Result of analyses
* Statistical significance, point estimates, and variability (confidence intervals)
* Table and figures
* Consider supplemental digital content for online posting
* Report, but do not interpret the results or editorialize

#### Discussion

* The point or “so what” of the study
* Sumary
* Compare finding with previous literature
* Implication
* Limitations: possible problems with the methods used
* Recommendataion for action
* Recommendations for further study
* Conclusion


### Research Gap types and explain

Types of Research Gaps

* (a) Evidence Gap; (Contradictory Evidence Gap)
* (b) Knowledge Gap; (Knowledge Void Gap)
* (c) Practical-Knowledge Conflict Gap; (ACtion-Knowledge Conflict Gap)
* (d) Methodological Gap; (Methods and Research desgin Gap)
* (e) Empirical Gap; and (Evaluation Void Gap)
* (f) Theoretical Gap; (Theory Application Void Gap)
* (g) Population Gap. (Population Gap)


Làm sao để định nghĩa một vấn đề toán học
======

### Một số prompt để hỗ trợ việc định nghĩa vấn đề toán học

* Prompt1:  Hãy biểu diễn toán học theo yêu cầu sau: Cho X là một dataset gồm tập hợp các ảnh X_i và nhãn C_i. Đối với mỗi ảnh X_i có kích thước là m*n Dùng dataset X làm đầu vào cho thuật toán CNN và hãy biểu diễn từng bước từng bước phần toán đó. Hãy viết bằng latex

* Prompt2: Hãy biểu diễn toán học theo yêu cầu sau: Cho X là một dataset gồm có m dòng dữ liệu, và n thuộc tính. trong thực nghiệm này thì m = 1000 và n = 15. Trong đó thuộc tính thứ i gọi là t_i Tiến hành quá trình tiền xử lý dữ liệu bằng theo dạng sau: Hãy biểu diễn one-hot encoder. sử dụng z-function. cuối cùng cho vào CNN để phân loại các dữ liệu. Hãy viết bằng latex

* Prompt3: Cho X một dataset gồm m dòng dữ liệu, mỗi dòng dữ liệu chứa thông tin traffic được truyền đi, và thông tin mỗi dòng gồm n thuộc tính. trong trường hợp thực nghiệm gồm n = 15 thuộc tính. biểu diễn một graph chứa nhiều nút, mỗi nút chứa thông tin của một dòng dữ liệu của dataset X và kết nối giữa các nút bằng cách nối giữa dòng thứ i và dòng thứ i+1 trong dataset. Hãy phát biểu bài toán theo dạng toán học Hãy viết bằng latex code


Kinh nghiệm vẽ hình ảnh bằng drawio và excel
======

### Dùng draw io để vẽ

* File –> export as –> pdf –> crop (rồi sau đó convert qua image cũng được)
* Hoặc advanced –> DPI custom (600)

### Đối với viết bằng word

Lưu ý: chỗ file –> option –> advanced –> ở tab image size and quality

* Step 1: In Word, go to File, click on Options > Advanced.
* Step 2: Under Image Size and Quality, select High fidelity in the Default resolution list.
* Step 3: Select the Do not compress images in file check box.

### Để vẽ chart gồm 2 trục y (tricks)

Trong trường hợp mình vẽ gồm có trục x là A, B, C và trục giá trị có 2 cột.

* Mình nên thêm 2 cột giả lập bên cạnh và để tạm giá trị bất kỳ (sau này sẽ đẩy giá trị của nó về 0)
* Sau đó, vẽ chart bất kỳ bằng cách insert -> vẽ biểu đồ cột -> chọn tab design để chuyển hàng thành cột bằng cách chọn design rồi chọn Switch Row Column
* Tiếp đến chọn cột muốn chuyển qua trục y2 -> format data series -> Second axis
* Tiếp đến, chỉnh gap width về khoảng 400% cho bằng với cột đầu tiên (cái này tương đối)
* Sau đó reset các giá trị của cột giả lập về 0 -> xóa các legend và các cột đó tương ứng.

### Kinh nghiệm để chuyển log history khi train model và vẽ bằng excel

Ví dụ dữ liệu mẫu thế này:

* Epoch 27/150 (A4)
* 53ms/step - loss: 0.4148 - accuracy: 0.8429 (giả sử dòng A5)

Mình sẽ thực hiện như sau:

* Bước 1: chuyển dòng A5 về dạng này: loss: 0.9211 - accuracy: 0.6500
* Viết lệnh sau: `=MID(A5, FIND("loss: ", A5), LEN(A5) - FIND("loss: ", A5) + 1)`
* Bước 2: Lấy loss value
* Thực hiện lệnh: `=MID(B5, FIND("loss: ", B5) + 6, FIND(" -", B5) - (FIND("loss: ", B5) + 6))`
* Lấy acc: `=MID(B5, FIND("accuracy: ", B5) + 10, LEN(B5) - FIND("accuracy: ", B5) - 10)`

Ghi chú:

* Lúc này dòng A4 sẽ không có kết quả, mình sẽ bỏ qua.
* Lúc này mình sẽ đánh thêm cột index, sau đó dùng hàm iseven() để check index chẵn hay không?
* Lúc đó, mình lọc các cột cần lấy loss và acc để copy ra --> kết quả


### Vẽ hình dùng excel và cách export pdf, image cho mượt

Thực hiện các bước sau để vẽ hình bằng excel

* Nhập giá trị bảng cần vẽ –> chọn bảng cần vẽ –> insert –> recommended chart
* Right click vào chart –> Move chart –> new sheet
* Thay đổi text (title, tên trục)
* Click vào từng series chart để đổi màu sắc
* Chọn shape -> Line (No line)
* Zoom size trục x, trục y
* Set min and max value của 2 trục (click vào trục giá trị –> Axis Options)
* Sau đó, chọn file -> export –> Chỗ dưới Optimize thì có thể chọn options
* Đánh dấu tick vào ô ISO 19005
* Có thể dùng PDF-Xchange Editor cho view pdf
* Organize –> Crop to White Margins (để xóa vùng trắng không cần thiết)
* Convert –> Export to Images (ảnh sẽ rất mượt)
* Dùng pdf để load lên latex là được.


Một số prompt hỗ trợ việc viết bài báo
======


### Prompt 0

* There are occasional grammatical errors and awkward phrases that need to be smoothed out. I need your help check and repair correctly. Maintain the academic tone: Insert the passage from your academic paper here

### Prompt 1

* Now I will give you some text. Your task is to rewrite it with an academic tone and in a creative way to get rid of plagiarism. However, you must keep the main idea as in the original text and do not use complex vocabulary: Insert the passage from your academic paper here

### Prompt 2

* As ChatGPT, an AI language model with expertise in condensing text while preserving the main idea and maintaining an academic tone, I need your help shortening a passage from my academic paper without losing the main idea. Can you condense the following passage from my academic paper while preserving the main idea and keeping an academic tone? Please focus on reducing the word count, retaining the main idea, and ensuring a coherent and concise result. Maintain the academic tone: Insert the passage from your academic paper here]

### Prompt 3

* I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. My first sentence is ‘istanbulu cok seviyom burada olmak cok guzel’

### Prompt 4

Evaluate and analysis result with binary classification in IoT-botnet problem with this result. Table 1: Some metrics of binary classification Evaluation
* write conclusion with the name of paper: IoT Botnet Detection and Classification using Machine Learning Algorithms
* with result: compare between some ML algorithm: DT, kNN, RF, XGB and some result

### Examples of ChatGPT Prompts for Writing Research Papers

How can I structure my introduction for a research paper on [topic]?

What are some key arguments and evidence to support my thesis statement about [topic]?

Can you suggest some reliable sources or studies related to [topic] that I can include in my literature review?

What are the main limitations of the existing research on [topic], and how can I address them in my paper?

Could you provide me with a logical flow for organizing my research findings and analysis in the results section?

Writing Assistance: “Can you help me rephrase this sentence/paragraph in a more academic tone?”

Literature Review Assistance: “Can you summarize the key findings and arguments of [insert academic paper title]?”

Brainstorming Ideas: “I’m working on a research project about [insert topic]. Can you suggest some potential research questions or hypotheses?”

Methodology Suggestions: “What are some common research methods used in [insert field of study]?”

Statistical Analysis: “Can you explain the difference between a t-test and an ANOVA?”

Writing Assistance: “Can you help me rephrase this sentence/paragraph in a more academic tone?”

Citation Guidance: “What is the correct way to cite a book in APA style?”

Historical Context: “Can you provide a brief overview of the history of [insert topic]?”

Concept Explanation: “Can you explain the concept of [insert complex concept] in simple terms?”

Paper Structuring: “What is the typical structure of a research paper in [insert field of study]?”

Data Interpretation: “What conclusions might be drawn from a dataset where [insert observed trend]?”

Ethical Considerations: “What are some ethical considerations when conducting research on [insert topic]?”

Theory Application: “How might [insert theory] apply to a situation where [insert scenario]?”

Research Proposal Feedback: “Here’s a brief outline of my research proposal on [insert topic]. Can you provide some feedback?”

Language Translation: “Can you translate this [insert language] phrase to English?”

Proofreading: “Can you check this paragraph for grammatical errors?”

### To get the most out of ChatGPT when using it for writing research papers, consider the following tips:

Be specific: Provide as much detail as possible when asking questions or providing prompts. The more specific you are, the better ChatGPT can understand your needs and provide relevant information.

Use clear language: Avoid ambiguous or vague language in your prompts. Use precise terminology and clearly state what you are looking for to receive accurate and helpful responses.

Experiment with different prompts: If you’re not satisfied with the initial response, try rephrasing or asking the question from a different angle. ChatGPT’s responses can vary based on the way you frame your prompts.

Review and refine: While ChatGPT can provide valuable insights, it’s essential to critically evaluate the information it generates. Use ChatGPT’s suggestions as a starting point and conduct further research to validate and refine the content.

Practice and iterate: The more you use ChatGPT, the better you’ll become at generating effective prompts. Practice using different prompts and experiment with various approaches to refine your research paper writing process.

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

Phân biệt giữa systematic review và literature review
======

Systematic review là phương pháp review có hệ thống, đáp ứng đủ 8 đặc tính của 1 bài nghiên cứu khoa học là tính mục tiêu, tinh khách quan, tính lặp lại, tính nghiêm ngặt, tính xác thực, tính khái quát hoá, tính chính xác, và tính tin cậy.

1. Về tính mục tiêu, syst review luôn luôn yêu cầu reviewer phải xác định đối tượng và protocol nghiên cứu trước, có 2 thứ này thì ta mới formulate được hypothesis để dùng syst review mà chứng minh. Đồng thời, hiện nay các journal thường yêu cầu reviewer phải update liên tục quá trình review thông qua các trang đăng kí như PROSPERO, từ đó mà người khác có thể check được ta đã thực hiện reviewer đúng mục tiêu chưa, có sửa đổi gì không

2. Về tính khách quan, bài syst review nào cũng bắt buộc có ít nhất 2 author đọc song song để chọn paper, đánh giá paper, và chiết xuất thông tin. Vì vậy, nếu thực hiện đúng protocol đề xuất cho syst review thì sẽ hạn chế được bias và tăng tính khách quan của bài review

3. Về tính lặp lại, bài syst review luôn có mô tả rõ ràng phương pháp thực hiện review, và luôn report quá trình lựa chọn paper dưới dạng workflow, quá trình thực hiện review qua prospero, quá trình chiết thông tin và đánh giá bias thông qua các bảng data. Qua đó, nếu thực hiện đúng theo các khuyến nghị review thì ta thường sẽ lặp lại được data của syst review với độ lệch thấp. Đặc biệt, bài syst review rất dễ để người đọc phát hiện gian dối. Không như đa phần các nghiên cứu thực nghiệm cực kì khó để người ngoài check tính trung thực và chính xác của nghiên cứu.

4. Về tính nghiêm ngặt, như đã đề cập, syst review luôn tuân theo quy trình được chuẩn hoá, và phải report hầu như mọi quá trình thực hiện. Vì vậy nên phương pháp thực hiện syst review khá nghiêm ngặt và đôi khi khá cứng nhắc.

5. Về tính xác thực, do bài syst review là bằng chứng thực chứng, nghĩa là chứng minh giả thiết dựa trên 1 loạt bằng chứng nên rõ ràng phương pháp luận của syst review là phương pháp luận có tính xác thực cao. Khác với nhiều nghiên cứu kém chất lượng đưa ra giả thiết ko thể chứng minh ở thực tế

6. Về tính khái quát hoá, do syst review tổng hợp dữ liệu từ các nghiên cứu đơn lẻ nên hiển nhiên syst review có tính khái quát kiến thức cao hơn các nghiên cứu đơn lẻ được dùng trong bài syst review.

7. Về tính chính xác và tính tin cậy, bài syst review do tổng hợp và làm rõ các conflict của các nghiên cứu đơn lẻ, vì vậy nó hạn chế và thường loại bỏ được confounding factor xuất hiện trong mô hình nghiên cứu. Do đó, các kết luận từ syst review thường chính xác hơn bài nghiên cứu đơn lẻ

Literature review (đánh giá về văn bản khoa học) là một phần quan trọng trong quá trình nghiên cứu, trong đó tác giả tập trung trình bày và tổng hợp thông tin từ các nguồn văn bản, bài báo khoa học, sách và các tài liệu khác liên quan đến chủ đề của họ. Mục đích của literature review là cung cấp một bức tranh tổng quan về tình hình nghiên cứu hiện tại, các kiến thức đã biết, và các hạn chế trong lĩnh vực nghiên cứu cụ thể. Trong khi literature review tập trung vào việc tóm tắt, tổng hợp, và đánh giá các nghiên cứu đã được công bố trước đó, systematic review là một phương pháp nghiên cứu cụ thể để tổng hợp thông tin từ nhiều nghiên cứu khác nhau với một cách hệ thống và có phương pháp học thuật đặc biệt. Mục tiêu của systematic review là cung cấp một phân tích cơ bản và chất lượng cao về một chủ đề cụ thể, thường bằng cách sử dụng các phương pháp thống kê để kết hợp kết quả của nhiều nghiên cứu và đưa ra một kết luận toàn diện.

1. **Literature Review:**
   Trong một literature review về việc giảm độ trễ trong satellite network, tác giả có thể tập trung vào việc tổng hợp các phương pháp và kỹ thuật đã được đề xuất để cải thiện hiệu suất của mạng vệ tinh và giảm độ trễ. Nghiên cứu này có thể bao gồm các khía cạnh như:

   - **Kỹ thuật truyền dẫn:** Tổng hợp các nghiên cứu về kỹ thuật truyền dẫn dữ liệu qua satellite như MIMO (Multiple Input Multiple Output) để tăng cường băng thông và giảm độ trễ.

   - **Giao thức điều khiển truyền thông:** Đánh giá các giao thức điều khiển truyền thông mới như DVB-S2X (Digital Video Broadcasting - Satellite - Second Generation Extended) và SCTP (Stream Control Transmission Protocol) để tối ưu hóa việc truyền thông và giảm độ trễ.

   - **Quản lý tài nguyên:** Xem xét cách các mô hình quản lý tài nguyên có thể được triển khai để đảm bảo sự hiệu quả cao và giảm độ trễ.

   - **Thực nghiệm và kết quả:** Tổng hợp các nghiên cứu thực nghiệm trên các hệ thống thực tế hoặc mô phỏng để đánh giá hiệu suất và độ trễ trong các kịch bản thực tế.

2. **Systematic Review:**
   Trong một systematic review, tác giả sẽ thực hiện một quy trình hệ thống để chọn lọc, đánh giá chất lượng và tổng hợp kết quả từ nhiều nghiên cứu khác nhau liên quan đến giảm độ trễ trong satellite network. Đây có thể bao gồm:

   - **Lựa chọn tiêu chí và nguồn nghiên cứu:** Xác định các tiêu chí bao gồm loại dữ liệu, kỹ thuật truyền dẫn, và phương pháp đánh giá độ trễ. Chọn các nguồn nghiên cứu thích hợp từ cơ sở dữ liệu uy tín.

   - **Phương pháp hệ thống:** Áp dụng một phương pháp hệ thống để đánh giá chất lượng nghiên cứu, bao gồm đánh giá rủi ro nghiên cứu và chi tiết phương pháp nghiên cứu.

   - **Tổng hợp kết quả:** Sử dụng phương pháp thống kê để tổng hợp kết quả từ các nghiên cứu, đưa ra một hiểu biết toàn diện về các biện pháp giảm độ trễ hiệu quả nhất.

   - **Phân tích phụ thuộc vào ngữ cảnh:** Đánh giá cách mỗi nghiên cứu có thể áp dụng trong ngữ cảnh cụ thể, như loại hệ thống vệ tinh, mục tiêu truyền thông, và đặc điểm địa lý.

Cả hai loại đánh giá đều cung cấp cái nhìn toàn diện về tình trạng hiện tại và tiến triển trong lĩnh vực giảm độ trễ trong satellite network, nhưng systematic review có nhiệm vụ chủ đạo hóa hơn và tuân thủ một quy trình hệ thống để đảm bảo tính minh bạch và khách quan.

[Link về cách sử dụng AI để làm review paper](https://dauedu-my.sharepoint.com/:u:/g/personal/haodp_mi_dau_edu_vn/Ec67H5k92gpChXA4i-fiyk0Bd0fH1bCMAkmL-eFjf6DDJg?e=Ma7Pue)

Hết.
