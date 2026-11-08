# Tests

Chạy bộ kiểm thử đơn vị từ project root sau khi cài package:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests/unit -v
```

Các kiểm thử logic detector và phân loại nguồn vào không cần camera hoặc model checkpoint được nạp thật. Kiểm thử end-to-end với GPU cần cài dependency ML và chạy trên thiết bị có CUDA.
