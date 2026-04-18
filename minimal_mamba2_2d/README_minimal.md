# Minimal Mamba2 2D (standalone)

这个目录可以独立使用。

- **无需安装整个仓库**；
- 只需把该目录所在路径加入 `PYTHONPATH`，或在该目录同级执行脚本即可。

## 统一导出接口

```python
from minimal_mamba2_2d import Mamba2_2D
```

## 最小示例

```bash
python minimal_mamba2_2d/examples/run_minimal.py
```

脚本会执行：
1. 随机输入的前向；
2. 一次反向传播（`loss.backward()`）。
