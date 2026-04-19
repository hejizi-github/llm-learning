**适用场景**: 需要从零创建含 LaTeX 公式、中文内容、复杂数学的 Jupyter notebook

**步骤**:
1. 写 `/tmp/gen_nb.py`（Python 脚本），用 `json.dump()` 构造 notebook 对象，每个 cell 用函数 `md(source)` / `code(source)` 封装
2. 所有字符串用**单引号**作为外层 Python 字符串定界符，避免中文中的 ASCII 双引号截断字符串
3. 运行 `python3 /tmp/gen_nb.py` 验证脚本本身无 SyntaxError、JSON 写入成功
4. 运行 `jupyter nbconvert --to notebook --execute path/to/nb.ipynb --output executed.ipynb` 验证零错误
5. 删除 executed.ipynb，只提交原始 notebook
6. 写 tests/ 验证 notebook 数学性质 + notebook 可执行性

**注意事项**:
- 中文排版弯引号在 Python 源码里实际是 ASCII 双引号，放入双引号字符串会截断，检查用 grep
- LaTeX 公式中的反斜杠命令（sin, frac 等）在 Python 字符串里需双反斜杠
- 用 `matplotlib.use('Agg')` 防止 headless 环境报错
