import os
import re
import yaml
import json
import ast
import operator as op
from collections import defaultdict
from typing import Any
from typing_extensions import Self

# ----------------------------- 常量 -----------------------------
SAFE_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow,
    ast.BitXor: op.xor, ast.And: lambda a, b: a and b, ast.Or: lambda a, b: a or b,
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
    ast.USub: op.neg, ast.Not: op.not_,
}
SAFE_FUNCTIONS = {
    'min': min, 'max': max, 'abs': abs, 'int': int, 'float': float, 'bool': bool
}

# ----------------------------- 表达式求值 -----------------------------

def _preprocess_expr(expr: str) -> str:
    """将表达式中的“&& / || / ^”替换为 Python 语法可识别形式。"""
    expr = expr.strip()
    expr = re.sub(r"&&", " and ", expr)
    expr = re.sub(r"\|\|", " or ", expr)
    return expr


def _build_attr_path(node: ast.Attribute) -> str:
    """把 Attribute 节点链拼成 cache.all.style → "cache.all.style"""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        raise ValueError("Unsupported attribute chain")
    return ".".join(reversed(parts))


def _eval_expr(expr: str, local_vars: dict):
    expr = _preprocess_expr(expr)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in local_vars:
                return local_vars[node.id]
            if node.id in os.environ:
                return os.environ[node.id]
            raise KeyError(node.id)
        elif isinstance(node, ast.Attribute):
            dotted = _build_attr_path(node)
            if dotted in local_vars:
                return local_vars[dotted]
            # 尝试逐层解析 dict 对象
            base_val = _eval(node.value)
            if isinstance(base_val, dict):
                if node.attr in base_val:
                    return base_val[node.attr]
            raise KeyError(dotted)
        elif isinstance(node, ast.BinOp):
            return SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_eval(v) for v in node.values)
            elif isinstance(node.op, ast.Or):
                return any(_eval(v) for v in node.values)
        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op_, right_ in zip(node.ops, node.comparators):
                if not SAFE_OPERATORS[type(op_)](left, _eval(right_)):
                    return False
                left = _eval(right_)
            return True
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' not allowed in expressions")
            func = SAFE_FUNCTIONS[func_name]
            args = [_eval(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode='eval')
        return _eval(tree)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


# ----------------------------- 主类 -----------------------------
class ConfigResolver:
    VAR_PATTERN = re.compile(r"\$\{(?!\{)([^}]+)\}")  # 排除 ${{ 表达式占位
    EXPR_PATTERN = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.raw_config = self._load_config()
        self.flat_config = self._flatten(self.raw_config)
        self.dependencies = self._extract_dependencies()
        self.resolved = None

    # ------------------------- 公开接口 -------------------------
    def parse(self) -> dict[str, Any]:
        """解析配置文件，返回解析后的嵌套字典。"""
        order = self._topo_sort_with_cycle_check()
        # include all flat keys
        for k in self.flat_config:
            if k not in order:
                order.append(k)

        resolved = {}
        for key in order:
            if key not in self.flat_config:
                continue
            resolved[key] = self._resolve_value_recursively(self.flat_config[key], resolved)
        self.resolved = resolved
        return self._unflatten(resolved)

    def add_variable(self, key: str, value: Any) -> Self:
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self.flat_config[key] = value
        self.dependencies = self._extract_dependencies()
        return self.parse()

    # ----------------------- 内部解析逻辑 -----------------------
    def _resolve_value_recursively(self, value, resolved):
        if isinstance(value, list):
            return [self._resolve_value_recursively(v, resolved) for v in value]
        if isinstance(value, dict):
            return {k: self._resolve_value_recursively(v, resolved) for k, v in value.items()}

        # 变量替换先行
        def substitute_vars(val):
            prev = None
            while isinstance(val, str) and self.VAR_PATTERN.search(val):
                if val == prev:
                    raise ValueError(f"Unresolved variables remain in value: {val}")
                prev = val
                for var in self.VAR_PATTERN.findall(val):
                    if var in resolved:
                        v = resolved[var]
                    elif var in self.flat_config:
                        v = self._resolve_value_recursively(self.flat_config[var], resolved)
                        resolved[var] = v
                    elif var in os.environ:
                        v = os.environ[var]
                    else:
                        raise ValueError(f"Unresolved variable: {var}")
                    val = val.replace(f"${{{var}}}", str(v))
            return val

        value = substitute_vars(value)

        # 表达式求值
        if isinstance(value, str) and self.EXPR_PATTERN.search(value):
            value = self.EXPR_PATTERN.sub(lambda m: str(_eval_expr(m.group(1), resolved)), value)
        return value

    # ----------------------- 工具函数 -----------------------
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Only .yaml, .yml, or .json files are supported.")

    def _flatten(self, d, parent_key='', sep='.'):  # dict → flat
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    def _unflatten(self, d, sep='.'):  # flat → nested
        result = {}
        for k, v in d.items():
            keys = k.split(sep)
            tgt = result
            for sub in keys[:-1]:
                tgt = tgt.setdefault(sub, {})
            tgt[keys[-1]] = v
        return result

    def _extract_dependencies(self):
        deps = defaultdict(set)
        dotted_name_re = r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"

        for key, val in self.flat_config.items():
            if isinstance(val, str):
                deps[key].update(self.VAR_PATTERN.findall(val))
                for expr in self.EXPR_PATTERN.findall(val):
                    deps[key].update(re.findall(dotted_name_re, expr))

        all_nodes = set(deps.keys()).union(*deps.values()) if deps else set()
        for n in all_nodes:
            deps.setdefault(n, set())
        return deps

    def _topo_sort_with_cycle_check(self):
        """拓扑排序 + 循环检测。返回解析顺序（依赖优先）。"""
        visited: dict[str, int] = {}
        order: list[str] = []
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]):
            if visited.get(node) == 1:  # 灰 → 再次遇到 = 环
                cycles.append(path + [node])
                return
            if visited.get(node) == 2:  # 黑，已完结
                return
            visited[node] = 1  # 灰
            for nxt in self.dependencies.get(node, []):
                dfs(nxt, path + [node])
            visited[node] = 2  # 黑
            order.append(node)

        for k in sorted(self.dependencies):
            if visited.get(k, 0) == 0:
                dfs(k, [])

        if cycles:
            readable = [" -> ".join(c) for c in cycles]
            raise ValueError("Cycle(s) detected in variable references: " + str(readable))

        return order  # 依赖节点排在前面，已满足解析顺序
