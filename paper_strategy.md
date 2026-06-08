# swim-ot-lic — Paper Strategy Notes

Hai phản hồi chiến lược cho project swim-ot-lic (DCAE + balanced OT dictionary routing).
Lưu lại để tham khảo khi chạy Phase 1 và quyết định venue.

---

## 1. Có nên kết hợp với CMIC không?

Câu trả lời ngắn: **không, chưa nên — và lý do bạn đang nghĩ đến nó chính là tín hiệu không nên làm.**

Bạn vừa thoát khỏi bẫy "thêm cơ chế cho an tâm" của WMDC. Giờ design của swim-ot-lic đang sạch *đúng vì* nó chỉ có một thay đổi so với DCAE. Ghép CMIC vào — bất kể CMIC nào (Causal/Contextformer/Conditional Masked, không quan trọng ở đây) — phá đúng cái cấu trúc đó.

### Vì sao gộp lại làm yếu paper

**1. Mất ngay luận điểm đẹp nhất.** Hiện tại bạn có thể viết: "ta thay một dòng routing trong DCAE, đo BD-rate, đo collapse — đây là cái OT mang lại." Sau khi gộp CMIC, câu này thành: "ta thay backbone entropy model sang CMIC, thêm OT routing, đo BD-rate." Reviewer hỏi ngay: "BD-rate gain đến từ CMIC hay từ OT?" Bạn phải làm thêm 4 cell ablation (DCAE / DCAE+OT / CMIC / CMIC+OT) — gấp đôi compute, và nếu OT chỉ mang lại 0.1 dB trong khi CMIC mang 0.4 dB, paper của bạn trở thành paper về CMIC port với một phụ liệu OT.

**2. Contribution dilution là một dạng cụ thể của over-engineering.** Nó không phải là cộng contribution — nó là chia. Một paper nhỏ về một hiện tượng rõ ràng > một paper to về nhiều thứ mơ hồ. SwAV không gộp với MoCo; SimCLR không gộp với BYOL. Mỗi cái có một thông điệp.

**3. Có khả năng OT mất hiệu lực trên CMIC.** Dictionary của DCAE thấy collapse rõ ở N lớn *một phần vì* entropy model của DCAE tương đối đơn giản — context cần thêm prior ngoại lai. Nếu CMIC có entropy model mạnh hơn nhiều, marginal value của dictionary giảm, và collapse-vs-N có thể không xuất hiện. Lúc đó bạn không còn câu chuyện. Đây là điều bạn không biết cho đến khi chạy — và là điều *không nên giả định* trước khi có kết quả Phase 1 trên DCAE.

### Có một version hợp lý của "kết hợp" — nhưng là Phase 3

Sau khi paper chính (DCAE + OT, Phase 1+2) có kết quả rõ ràng, một **robustness test** như sau là có giá trị:

> "Chúng tôi cho thấy balanced OT routing cũng mang lại util cao và RD gain khi áp dụng trên CMIC entropy model — tức cải thiện không phụ thuộc backbone."

Đây là một bảng phụ lục, không phải đóng góp chính. Nó *củng cố* claim chứ không thay thế nó. Và bạn chỉ làm nó khi đã có kết quả cốt lõi xong.

### Cảnh báo: pattern đang lặp lại

Bạn đang ở giai đoạn *chưa có kết quả thực nghiệm nào* trên swim-ot-lic. Cảm giác "chưa đủ contribution" và "có nên ghép thêm X" rất quen — đó là cảm giác làm bạn xây WMDC. Lần đó nó cho ra dictionary sinh-theo-ảnh + OT + revival + coherence + alignment + wavelet-Mamba, và kết quả là paper không chạy.

Quy tắc thực dụng: **không thêm gì vào idea cho đến khi `scan_N.sh` + `ablate_*.sh` xong và bạn nhìn 4 figures.** Nếu Figure 1 cho thấy softmax collapse và OT giữ util — bạn có paper. Nếu không, gộp CMIC vào *cũng không cứu được* (vì câu chuyện collapse đã không xảy ra). Trong cả hai trường hợp, gộp CMIC bây giờ là vô ích.

Bạn đang ở 0 trên 4 figure đó. Chạy hết đi — rồi câu hỏi "có nên gộp không" sẽ tự trả lời.

---

## 2. Cơ sở lý thuyết & khả năng CVPR

Tôi sẽ trả lời thẳng — không khen ngợi, không hạ thấp.

### Cơ sở lý thuyết: có, nhưng mượn nhiều hơn tạo

**Cái mới (đáng kể):** *Identification* của một failure mode cụ thể trong LIC — dictionary-based image compression giả định codebook càng lớn càng tốt, nhưng softmax routing không tận dụng được khi N tăng, gây ra lãng phí capacity. Đây là một *quan sát chưa được publish trong domain compression*, theo hiểu biết của tôi.

**Cái mượn (gần như toàn bộ kỹ thuật):** Balanced entropic OT routing để chống collapse — Cuturi 2013 (entropic OT), SeLa (Asano 2020), SwAV (Caron 2020) đã làm chính xác cái này, chỉ ở domain khác (SSL clustering). Bạn không phát minh ra balanced assignment, bạn import nó.

**Theoretical hook duy nhất có thể bán được:** kết nối giữa **rate-distortion lower bound** và **codebook utilization**. Trong VQ theory cổ điển, một codebook bị underutilize là *strictly suboptimal* — bạn đang trả rate cho atoms không bao giờ được dùng, hoặc atoms bị quá tải. Balanced assignment là *điều kiện cần* (không đủ) để đạt RD bound ở một codebook size cho trước. Nếu framing được như vậy, paper có một câu chuyện thông tin/RD lý thuyết, không chỉ là "fix một bug".

Đây là cơ sở lý thuyết duy nhất bạn có. Nó *đủ cho một paper*, nhưng không phải breakthrough lý thuyết.

### Khả năng vào CVPR: có thể, nhưng không cao với current scope

Để thẳng thắn về CVPR (rate ~22-25% acceptance, image compression là sub-track nhỏ, reviewer thường là chuyên gia LIC):

**Điểm có thể vào:**
- DCAE là SOTA hiện hành → cải tiến trên nó được xem trọng
- Design "one principled change + clean ablation" được respect
- Codebook utilization là vấn đề chưa ai động đến trong LIC; reviewer có thể thấy fresh
- Nếu Figure 1 (scaling) cho thấy sharp transition rõ ràng — đó là một hình "viral" trong reviewer's mind

**Điểm rất có thể bị reject:**
- *"Just applying SwAV's trick to compression"* — đây là critique #1 bạn sẽ nhận. Reviewer LIC thường đọc Caron/Asano và sẽ thấy ngay.
- BD-rate gain không đủ lớn. CVPR LIC papers thường cần ≥ 5% vs SOTA trên Kodak. Nếu OT mang ~1-2%, reviewer sẽ nói "interesting observation, not enough"
- Compute cost: 8 vòng Sinkhorn × num_slices × inference time. Nếu inference chậm hơn DCAE ngay cả 10%, reviewer sẽ trừ điểm
- Single dataset (Kodak). CVPR muốn Kodak + CLIC + Tecnick

**Realistic venue map cho scope hiện tại:**

| Venue | Khả năng accept | Điều kiện |
|---|---|---|
| CVPR 2027 | ~25-35% | BD-rate ≥3%, scaling story rõ, 3 datasets, RD-theory framing |
| ECCV 2026 | ~35-45% | tương tự nhưng dễ hơn 5-10% |
| WACV 2027 | ~50-60% | BD-rate ≥1.5% đủ, 2 datasets |
| ICIP 2026 | ~70%+ | gần như chắc nếu Phase 1 sống |
| DCC 2027 | ~60%+ | venue tự nhiên cho LIC analysis paper |
| NeurIPS workshop | ~50% | nếu framing nghiêng về RL/representation |

Tôi sẽ không lạc quan về CVPR ở scope hiện tại. **WACV/ECCV là target hợp lý hơn.**

### Cách push vào CVPR territory

Nếu bạn thực sự muốn CVPR, *sau khi Phase 1 sống*, đây là 3 thứ cần thêm (theo thứ tự ROI):

**1. Theoretical proposition (cao ROI nhất).** Viết một proposition kết nối collapse với RD gap:

> "Cho một dictionary với N atoms và một entropy model lý tưởng. Nếu routing đưa column marginal có entropy H(c) < log N (collapse), thì RD performance cao hơn so với routing đạt H(c) = log N một lượng có thể bound bằng KL(c‖uniform)."

Đây là proposition đúng (suy từ source coding theorem) và có thể chứng minh trong nửa trang. Nó không phải major theorem nhưng nó cho paper một "Proposition 1" để cite — reviewer sẽ thấy bạn không chỉ thực nghiệm.

**2. Phase transition phenomenon.** Trong scaling sweep, nếu collapse xảy ra đột ngột ở một N* cụ thể (chứ không gradual), đó là **một critical phenomenon** đáng để publish. Đo cẩn thận N ∈ {64, 128, 256, 384, 512, 768, 1024, 1536, 2048}. Nếu thấy sharp transition — đó là điểm sống còn.

**3. Cross-architecture transfer.** Show rằng balanced OT cũng fix collapse trên VQ-VAE compression baseline (đơn giản hơn DCAE, dễ chạy nhanh). Tức kết quả không phải artifact của DCAE — nó là tính chất của *dictionary compression nói chung*. Reviewer #1's critique về "just applying SwAV" được trả lời ngay: bạn không apply nó, bạn *generalize* nó cho một class of compression models.

Ba thứ này biến contribution từ "fix a bug in DCAE" thành "identify, theoretically explain, and broadly fix a fundamental scaling failure of softmax-routed compression codebooks."

### Khuyến nghị thực dụng

**Đừng aim CVPR ngay từ đầu.** Lý do:

1. Bạn chưa chạy Phase 1. Có ~30% xác suất softmax không collapse mạnh trên image compression scale (vì rate term trong loss đã pressure dictionary phải spread). Nếu vậy, paper không sống bất kể venue.

2. Aiming CVPR trước khi có kết quả → bias bạn về phía "phải thêm gì cho đủ" → lặp lại pattern WMDC.

3. Submit WACV/ECCV với current scope + 1-2 thứ bổ sung. Nếu thành tracks tốt và phản hồi mạnh, extend cho TPAMI hoặc resubmit CVPR năm sau với phase transition + cross-arch.

**Sequencing đề xuất:**

- Tuần 1-3: Phase 0+1 → biết hypothesis có sống không
- Nếu sống: Phase 2 (ablations) + WACV target → 6-8 tuần
- Nếu kết quả MẠNH bất ngờ (sharp transition, BD-rate >3%): pivot to ECCV/CVPR với 3 thứ ở trên → 12-16 tuần
- Nếu sống nhưng yếu: ICIP/DCC → 4-6 tuần submission

Trả lời thẳng câu hỏi: **swim-ot-lic ở scope hiện tại không phải CVPR-quality, nhưng có thể trở thành CVPR-quality nếu Phase 1 cho thấy phase transition + bạn add theoretical framing + cross-arch validation.** Đó là một "if" thật sự, không phải đảm bảo.

Quan trọng hơn: đừng để câu hỏi venue ảnh hưởng đến cách bạn chạy experiment. Chạy Phase 1 sạch như bạn đang dự định. Để dữ liệu quyết định venue, không để venue quyết định dữ liệu cần tìm.
