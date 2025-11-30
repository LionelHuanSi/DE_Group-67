# Differential Evolution Variants Comparison  
Repository for comparing 6 Differential Evolution (DE) variants on the Rastrigin optimization problem.

---

## Mô tả bài toán

Mục tiêu là tìm vector biến liên tục **x = (x₁, x₂, ..., xₙ)** sao cho giá trị hàm mục tiêu **f(x)** đạt cực tiểu.

### Hàm Rastrigin (n ≥ 2)

f(x) = A · n + ∑_{i=1}^{n} ( x_i^2 − A · cos(2π x_i) )

- A = 10 (hằng số)
- n = số chiều (ví dụ: 10)
- Miền giá trị: x_i ∈ [−5.12, 5.12]

Hàm này là bài toán tiêu chuẩn để đánh giá các thuật toán tối ưu tiến hóa vì có nhiều cực trị địa phương.


## 🔹 Các biến thể DE được so sánh

Repository tiến hành chạy và so sánh **6 biến thể DE phổ biến**:

- **DE/rand/1/bin** (thuật toán gốc)
- **DE/best/1/bin**
- **DE/current-to-best/1**
- **DE/rand/2/bin**
- **JADE** (biến thể thích ứng thông số)
- **SHADE** (biến thể thích ứng nâng cao)

---

## 📌 Tham số thí nghiệm

| Tham số               | Ký hiệu | Giá trị              |
|----------------------|---------|----------------------|
| Hàm mục tiêu         | f(x)    | Rastrigin            |
| Số chiều không gian  | D       | 10                   |
| Kích thước quần thể  | NP      | 70                   |
| Số thế hệ tối đa     | Gmax    | 1000                 |
| Miền tìm kiếm        | [xmin, xmax] | [-5.12, 5.12]  |
| Số lần chạy độc lập  | Run     | 1                    |

---

## 🚀 Cách chạy mã

File chính của project: de_compare.py

