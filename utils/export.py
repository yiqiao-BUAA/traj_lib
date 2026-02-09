# pip install altair pandas
import altair as alt
import pandas as pd
from typing import List, Dict, Any

from utils.logger import get_logger
logger = get_logger(__name__)

alt.data_transformers.disable_max_rows()

def export_alt(
    items: List[Dict[str, Any]],
    title: str | None = None,
    smooth_alpha: float | None = None,   # None=不平滑；否则 0~1 越大越平滑
    path: str = "metrics.html",
    x_key: str = "x",
    panel_width: int = 520,
    panel_height: int = 320,
    two_cols: bool = True,               # 默认双栏
):
    """
    将 list[dict] 绘制为单页 HTML（path）。
    - 对 list 中每个 dict 生成一个“面板”：单张多折线图（所有数值指标 vs x）
    - 若 dict 含 'title'，面板标题用其值；否则为 'Series i'
    - 若包含 'x'，要求各指标与 x 等长；否则横轴为 1..N（要求各指标同长）
    - 不再对 'loss' 做单独绘图或检测，只有在 items[0] 上用作存在性断言
    """

    def _to_list(v: Any) -> list[float]:
        if isinstance(v, (list, tuple, pd.Series)):
            return [float(x) for x in v]
        try:
            return [float(v)]
        except Exception as e:
            raise ValueError(f"值 {v!r} 不能转为数值序列: {e}") from e

    def _prep_frame(d: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
        # 面板标题
        panel_title = str(d["title"]).strip() if "title" in d else None

        # 收集可数值化字段（排除明显文本，保留 x）
        series_map: Dict[str, list[float]] = {}
        for k, v in d.items():
            key = str(k).strip() if k is not None else ""
            if not key or key == "title":
                continue
            if isinstance(v, str) and key != x_key:
                # 非数值字符串且不是 x，跳过
                continue
            try:
                series_map[key] = _to_list(v)
            except ValueError:
                if key == x_key:
                    raise
                continue

        # 决定横轴
        if x_key in series_map:
            x = series_map.pop(x_key)
            n = len(x)
            keep = {k: s for k, s in series_map.items() if len(s) == n}
            dropped = [k for k, s in series_map.items() if len(s) != n]
            if dropped:
                logger.warning(f"丢弃与 x 长度不一致的指标：{dropped}（期望 {n}）")
            if not keep:
                raise ValueError("没有与 x 同长度的可画指标")
            df = pd.DataFrame({x_key: x, **keep})
        else:
            if not series_map:
                raise ValueError("没有可画的指标")
            lengths = {k: len(v) for k, v in series_map.items()}
            lens = set(lengths.values())
            if len(lens) != 1:
                raise ValueError(f"无 x 时要求各指标长度一致，当前长度：{lengths}")
            n = next(iter(lens))
            df = pd.DataFrame({x_key: list(range(1, n + 1)), **series_map})

        # 数值化
        df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
        if df[x_key].isna().all():
            raise ValueError(f"横轴 {x_key} 全为 NaN")
        for c in df.columns:
            if c == x_key:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if not panel_title:
            panel_title = "Series"

        return df, panel_title

    def _single_chart(df: pd.DataFrame, panel_title: str):
        # 所有数值列（含 loss 在内）一起画
        value_cols = df.select_dtypes(include="number").columns.tolist()
        if x_key in value_cols:
            value_cols.remove(x_key)
        if not value_cols:
            raise ValueError("没有可画的数值列")

        long_df = df[[x_key] + value_cols].melt(
            id_vars=x_key, value_vars=value_cols,
            var_name="metric", value_name="value"
        ).sort_values(["metric", x_key])

        y_field = "value"
        if smooth_alpha is not None:
            long_df["value_smooth"] = (
                long_df.groupby("metric")["value"]
                       .transform(lambda s: s.ewm(alpha=float(smooth_alpha)).mean())
            )
            y_field = "value_smooth"

        hi = alt.selection_point(fields=["metric"], bind="legend")
        chart = (
            alt.Chart(long_df)
              .mark_line(point=True)
              .encode(
                  x=alt.X(f"{x_key}:Q", title=x_key),
                  y=alt.Y(f"{y_field}:Q", title="Value"),
                  color=alt.Color("metric:N", title="Metric"),
                  opacity=alt.condition(hi, alt.value(1.0), alt.value(0.2)),
                  tooltip=[
                      alt.Tooltip("metric:N", title="Metric"),
                      alt.Tooltip(f"{x_key}:Q", title=x_key),
                      alt.Tooltip(f"{y_field}:Q", title="Value", format=".6g"),
                  ],
              ).properties(width=panel_width, height=panel_height, title=panel_title)
              .add_params(hi)
              .interactive()
        )
        return chart

    # 生成所有面板
    panels = []
    for i, d in enumerate(items, 1):
        try:
            df, base_title = _prep_frame(d)
            panel_title = d.get("title", f"{base_title} {i}")
            panels.append(_single_chart(df, str(panel_title)))
        except Exception as e:
            logger.warning(f"跳过第 {i} 个元素：{e}")

    if not panels:
        raise ValueError("没有生成任何图表，请检查输入数据。")

    # 布局：默认双栏
    if two_cols and len(panels) > 1:
        rows = []
        for j in range(0, len(panels), 2):
            row = panels[j]
            if j + 1 < len(panels):
                row = alt.hconcat(row, panels[j + 1], spacing=16)\
                         .resolve_scale(color="independent", y="independent")
            rows.append(row)
        combined = alt.vconcat(*rows, spacing=16)\
                      .resolve_scale(color="independent", y="independent")
    else:
        combined = alt.vconcat(*panels, spacing=16)\
                      .resolve_scale(color="independent", y="independent")

    if title:
        combined = combined.properties(title=title)

    combined.save(path)
