# traj_lib/utils/register.py

from typing import Any, Callable, Dict, Iterator, List, Sequence, Union
from collections.abc import MutableMapping
from collections import defaultdict
import re
from difflib import get_close_matches


def _normalize(s: str) -> str:
    """弱化大小写、下划线/连字符/空格差异，用于粗略匹配。"""
    return re.sub(r"[^a-z0-9]", "", str(s).casefold())


class CategoryRegistry(MutableMapping[str, Any]):
    """单一分类的注册器：dict-like + 装饰器注册 + 粗略匹配建议。"""

    def __init__(self, category: str) -> None:
        self.category = category
        self._data: Dict[str, Any] = {}

    # --- dict-like 必需接口 ---
    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:  # <- 未命中时给出候选
            suggestions = self.suggest(key, n=3, cutoff=0.6)
            if suggestions:
                msg = (
                    f"{key!r} not registered in {self.category!r}. "
                    f"Did you mean: {', '.join(suggestions)} ?"
                )
            else:
                msg = (
                    f"{key!r} not registered in {self.category!r}. "
                    f"Available: {', '.join(sorted(self._data.keys()))}"
                )
            raise KeyError(msg) from None

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<CategoryRegistry {self.category!r} keys={list(self._data.keys())!r}>"

    # --- 额外能力 ---
    def register(self, name: Union[str, Sequence[str]]) -> Callable[[Any], Any]:
        """
        用作装饰器：
          @registry.register("name")
          @registry.register(["name1", "name2"])
          @registry.register(("name1", "name2"))
        规则：
          - 首次注册：把所有名字都指向同一对象
          - 重复把“同一对象”注册到已存在相同名字：幂等
          - 若已有相同名字但指向“不同对象”：抛 ValueError 防止静默替换
        """
        # 统一成列表
        if isinstance(name, (list, tuple, set)):
            names = list(name)
        else:
            names = [name]

        # 去重但保持相对顺序
        seen: set[str] = set()
        names = list(dict.fromkeys(names))

        def decorator(obj: Any) -> Any:
            # 先做冲突检测，保证“要么全部成功，要么不改任何内容”
            conflicts = []
            for n in names:
                if n in self._data and self._data[n] is not obj:
                    conflicts.append(n)
            if conflicts:
                raise ValueError(
                    f"Name(s) {conflicts!r} already registered in {self.category!r} "
                    f"by different object(s)."
                )

            # 写入（或幂等跳过）
            for n in names:
                if n in self._data:
                    # 同一对象重复注册到同名，幂等
                    continue
                self._data[n] = obj
            return obj

        return decorator

    def suggest(self, target: str, n: int = 3, cutoff: float = 0.6) -> List[str]:
        """
        给出与 target 最相近的 n 个键名。cutoff ∈ [0,1]，越高越严格。
        """
        # 建立 规范化名 -> 原始名列表 的映射，避免规范化后碰撞丢信息
        norm_to_raw: Dict[str, List[str]] = {}
        for k in self._data.keys():
            nk = _normalize(k)
            norm_to_raw.setdefault(nk, []).append(k)

        target_norm = _normalize(target)
        match_norms = get_close_matches(
            target_norm, list(norm_to_raw.keys()), n=n, cutoff=cutoff
        )

        seen, out = set(), []
        for nk in match_norms:
            for raw in norm_to_raw[nk]:
                if raw not in seen:
                    seen.add(raw)
                    out.append(raw)
                    if len(out) >= n:
                        break
            if len(out) >= n:
                break
        return out

    # （可选）显式创建别名的辅助方法
    def alias(self, existing: str, *aliases: str) -> None:
        """
        将已有键 existing 的对象，额外绑定到若干别名 aliases。
        行为同 register 的冲突规则。
        """
        obj = self[existing]  # 复用 __getitem__ 的 KeyError+suggest 提示
        # 先检测
        conflicts = []
        for a in aliases:
            if a in self._data and self._data[a] is not obj:
                conflicts.append(a)
        if conflicts:
            raise ValueError(
                f"Alias name(s) {conflicts!r} already registered in {self.category!r} "
                f"by different object(s)."
            )
        # 再写入
        for a in aliases:
            self._data[a] = obj


class RegistryHub:
    """管理多个分类注册器，按需懒创建。"""

    def __init__(self) -> None:
        self._cats: Dict[str, CategoryRegistry] = {}

    def category(self, name: str) -> CategoryRegistry:
        if name not in self._cats:
            self._cats[name] = CategoryRegistry(name)
        return self._cats[name]

    # 兼容旧的 _register(category, name) 风格（仍只接受单名）
    def register(self, category: str, name: str) -> Callable[[Any], Any]:
        return self.category(category).register(name)

    def __getitem__(self, category: str) -> CategoryRegistry:
        return self.category(category)


# --- 模块级单例（向外暴露的“数个实例化的结果”） ---
_HUB = RegistryHub()

DATALOADER_REGISTRY: CategoryRegistry = _HUB.category("dataloader")
EVAL_REGISTRY: CategoryRegistry = _HUB.category("eval")
VIEW_REGISTRY: CategoryRegistry = _HUB.category("view")
EARLY_STOP_REGISTRY: CategoryRegistry = _HUB.category("early_stop")

# --- 向后兼容：保留旧的装饰器名（现在也接受多个名字） ---
NameArg = Union[str, Sequence[str]]


def register_dataloader(name: NameArg):
    return DATALOADER_REGISTRY.register(name)


def register_eval(name: NameArg):
    return EVAL_REGISTRY.register(name)


def register_view(name: NameArg):
    return VIEW_REGISTRY.register(name)

def register_early_stop(name: NameArg):
    return EARLY_STOP_REGISTRY.register(name)


# --- 向后兼容：保留旧的 _register（内部使用也可继续工作）---
def _register(category: str, name: str) -> Callable[[Any], Any]:
    return _HUB.register(category, name)


# --- 额外：为极端保守的旧代码保留一个 dict-of-dicts 视图（只为读取/迭代设计）---
# 注意：这里的 value 是 *同一份* 底层 dict，读写会与上面三者互通。
_REGISTRY: Dict[str, Dict[str, Any]] = defaultdict(dict)
_REGISTRY["dataloader"] = DATALOADER_REGISTRY._data
_REGISTRY["eval"] = EVAL_REGISTRY._data
_REGISTRY["view"] = VIEW_REGISTRY._data
_REGISTRY["early_stop"] = EARLY_STOP_REGISTRY._data