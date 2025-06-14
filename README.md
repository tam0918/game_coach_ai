# Triple Tile AI Analyst

## 1. Tổng quan

Dự án này là một ứng dụng full-stack mô phỏng game Triple Tile, được trang bị một hệ thống AI để phân tích và đánh giá nước đi của người chơi theo thời gian thực.

Cốt lõi của dự án là một pipeline học máy hoàn chỉnh:
1.  **Tự động chơi game** để tạo ra dữ liệu thô.
2.  **Sử dụng LLM (FPT API)** để gán nhãn và làm giàu dữ liệu, tạo ra một bộ dữ liệu huấn luyện chất lượng cao.
3.  **Huấn luyện một mô hình học máy (PyTorch)** cục bộ để có thể tái tạo lại khả năng phân tích của LLM.
4.  **Tích hợp mô hình** vào một API backend (Flask) để đưa ra dự đoán nhanh chóng.
5.  **Xây dựng một giao diện game (React)** để người dùng có thể chơi và nhận được phân tích AI ngay lập tức.

## 2. Kiến trúc

Dự án được chia thành hai phần chính: `backend` và `frontend`.

### Backend (`/backend`)

-   **`app.py`**: Một API server sử dụng Flask, cung cấp 3 endpoint chính:
    -   `/predict`: Sử dụng mô hình PyTorch cục bộ để đưa ra phân tích nước đi (nhanh, dùng cho frontend).
    -   `/predict_rationale`: Sử dụng mô hình distilled để đưa ra giải thích về nước đi (nhanh, dùng cho frontend).
    -   `/evaluate-move`: Sử dụng FPT API để phân tích (chậm, dùng để tạo dữ liệu huấn luyện mới).
-   **`/simulation`**: Chứa logic game (`game_logic.py`) và một agent dựa trên luật (`heuristic_agent.py`) để tự động chơi và tạo dữ liệu (`run_simulation.py`).
-   **`/trainer`**: Pipeline học máy.
    -   `data_processor.py`: Xử lý và chuyển đổi dữ liệu thô.
    -   `model.py`: Định nghĩa kiến trúc mạng nơ-ron đa đầu ra bằng PyTorch.
    -   `train.py`: Kịch bản huấn luyện chính.
-   **`predictor.py`**: Tải mô hình đã huấn luyện và phục vụ cho việc dự đoán.
-   **`rationale_predictor.py`**: Tải mô hình distilled và cung cấp giải thích chi tiết về các nước đi.
-   **`train_distilled_model.py`**: Script để huấn luyện mô hình giải thích nước đi (rationale prediction).
-   **`requirements.txt`**: Danh sách tất cả các thư viện Python cần thiết.

### Frontend (`/frontend`)

-   Một ứng dụng game được xây dựng bằng React, TypeScript, và Vite.
-   Cung cấp giao diện chơi game Triple Tile trực quan.
-   Sau mỗi nước đi, ứng dụng sẽ gọi đến endpoint `/predict` của backend để nhận và hiển thị phân tích từ AI.
-   Bao gồm tính năng đánh giá sau khi kết thúc trò chơi (Post-Match Review) sử dụng endpoint `/predict_rationale` để cung cấp phản hồi chi tiết về toàn bộ trận đấu.

## 3. Hướng dẫn Cài đặt và Chạy

### Yêu cầu
-   Python 3.10+
-   Node.js và npm (hoặc yarn)
-   Git

### Bước 1: Cài đặt Backend

1.  **Clone a repository**
    ```sh
    git clone https://github.com/dcmgacode/games.git
    ```
2.  **Tạo môi trường ảo**
    Mở terminal trong thư mục gốc của dự án và chạy:
    ```sh
    python -m venv .venv
    ```

3.  **Kích hoạt môi trường ảo**
    -   **Windows (Command Prompt/PowerShell):**
        ```sh
        .\.venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```sh
        source .venv/bin/activate
        ```
    Sau khi kích hoạt, bạn sẽ thấy `(.venv)` ở đầu dòng lệnh.

4.  **Cài đặt các thư viện Python**
    Vẫn trong môi trường ảo, chạy lệnh:
    ```sh
    pip install -r backend/requirements.txt
    ```

### Bước 2: Cài đặt Frontend

1.  Mở một terminal **mới**.
2.  Điều hướng đến thư mục `frontend`:
    ```sh
    cd frontend
    ```
3.  Cài đặt các gói Node.js:
    ```sh
    npm install
    ```
    *(Hoặc `yarn install` nếu bạn dùng Yarn)*

### Bước 3: Chạy ứng dụng

Bạn cần chạy cả backend và frontend cùng một lúc trên hai cửa sổ terminal riêng biệt.

1.  **Chạy Backend Server**
    -   Mở terminal thứ nhất, đảm bảo môi trường ảo `.venv` đã được kích hoạt.
    -   Từ thư mục **gốc** của dự án, chạy lệnh:
        ```sh
        python backend/app.py
        ```
    -   Server sẽ khởi động tại `http://127.0.0.1:5001`. Bạn sẽ thấy thông báo mô hình đã được tải thành công.

2.  **Chạy Frontend App**
    -   Mở terminal thứ hai.
    -   Điều hướng đến thư mục `frontend`:
        ```sh
        cd frontend
        ```
    -   Khởi động server phát triển:
        ```sh
        npm run dev
        ```
    -   Mở trình duyệt và truy cập vào địa chỉ được cung cấp (thường là `http://localhost:5173` hoặc một cổng khác).

Bây giờ bạn đã có thể chơi game và xem kết quả phân tích từ AI!

## 4. Huấn luyện lại Mô hình (Tùy chọn)

Nếu bạn muốn cải tiến hoặc huấn luyện lại mô hình với dữ liệu mới.

### Bước 4.1: Tạo dữ liệu (Yêu cầu FPT API Key)

1.  Tạo một tệp tên là `.env bên trong thư mục `backend`.
2.  Thêm khóa API của bạn vào tệp `.env`:
    ```
    FPT_API_KEY="your_fpt_api_key_here"
    MODEL_NAME="meta/llama3-70b-instruct"
    ```
3.  Chạy kịch bản mô phỏng từ thư mục **gốc** (đảm bảo môi trường `.venv` đã được kích hoạt):
    ```sh
    python backend/simulation/run_simulation.py
    ```
4.  Quá trình này có thể mất nhiều thời gian. Sau khi hoàn tất, tệp `backend/simulation_results.csv` sẽ được tạo ra hoặc cập nhật.

### Bước 4.2: Huấn luyện

1.  Đảm bảo môi trường ảo `.venv` đã được kích hoạt.
2.  Chạy kịch bản huấn luyện từ thư mục **gốc**:
    ```sh
    python backend/trainer/train.py
    ```
3.  Kịch bản sẽ sử dụng dữ liệu trong `simulation_results.csv` để huấn luyện. Mô hình và bộ xử lý dữ liệu mới sẽ được lưu vào `backend/trainer/saved_model_torch/`.

### Bước 4.3: Tải lại mô hình mới

-   Dừng server backend (nhấn `Ctrl + C` trong terminal của nó) và khởi động lại để tải mô hình mới nhất:
    ```sh
    python backend/app.py
    ``` 