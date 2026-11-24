"""Observatory scaffold. Reads published evidence; does not create models."""
import sys
import tempfile

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

from model_arena.results import Results


st.set_page_config(page_title="Observatory · Model Arena", layout="wide")
st.markdown("<style>.block-container{padding-top:3.5rem}h1,h2,h3{color:#153456}</style>",
            unsafe_allow_html=True)
source = sys.argv[1] if len(sys.argv) > 1 else ""
if not source:
    st.info("Provide a publication reference to launch the monitor.")
    st.stop()


@st.cache_resource
def open_results(source):
    if source.startswith("s3://"):
        from model_arena.aws import materialize_publication
        local = materialize_publication(source, tempfile.mkdtemp(prefix="arena-publication-"))
        return Results.open(local)
    return Results.open(source)


try:
    results = open_results(source)
except Exception as error:
    st.error(str(error))
    st.stop()

st.sidebar.title("Observatory")
st.sidebar.caption("Resultados publicados · solo lectura")
view = st.sidebar.radio("Monitor", ["Historia y segmentos", "Model Arena", "Pronósticos y alertas"])
presentation = st.sidebar.radio("Presentación", ["Gráfico", "Tabla"], horizontal=True)
inventory = results.table("inventory")
st.sidebar.caption(f"Publicación: {results.path.name}")

if view == "Historia y segmentos":
    st.title("Historia y segmentos")
    st.metric("Series", len(inventory))
    candidates = [c for c in inventory if c != "entity" and inventory[c].nunique() <= 30]
    if candidates:
        group = st.selectbox("Agrupar por", candidates)
        counts = inventory.groupby(group, dropna=False).size().rename("series").reset_index()
        if presentation == "Gráfico":
            st.plotly_chart(px.bar(counts, x=group, y="series", text_auto=True), use_container_width=True)
        else:
            st.dataframe(counts, hide_index=True, use_container_width=True)
    st.dataframe(inventory, hide_index=True, use_container_width=True)

elif view == "Model Arena":
    st.title("Model Arena")
    curves = results.table("learning_curves")
    scores = results.table("scores")
    if not scores.empty:
        st.subheader("Comparación temporal")
        st.caption("Evidencia utilizada para selección; no es un test independiente.")
        metric = scores.groupby(["candidate", "input", "target"], as_index=False)[["value", "baseline_value"]].mean()
        metric["mejora"] = metric.baseline_value-metric.value
        if presentation == "Gráfico":
            st.plotly_chart(px.bar(metric, x="candidate", y="mejora", color="input", facet_row="target",
                                   labels={"mejora": "Pinball baseline − modelo"}), use_container_width=True)
        else:
            st.dataframe(metric, hide_index=True, use_container_width=True)
    st.subheader("Loss por época")
    if curves.empty:
        st.info("Esta publicación no contiene curvas de aprendizaje por época.")
        st.stop()
    inp = st.selectbox("Input", sorted(curves.input.unique()))
    folds = st.multiselect("Folds", sorted(curves.fold.unique()),
                           default=[f for f in ["F1", "F2", "F3"] if f in set(curves.fold)])
    candidates = sorted(curves[curves.input == inp].candidate.unique())
    chosen = st.multiselect("Candidatos", candidates, default=candidates[:4])
    segment_cols = [c for c in inventory if c != "entity"]
    scope = st.selectbox("Ámbito", ["total", "entity", *segment_cols])
    filtered = results.learning_curves(scope=scope, input=inp, fold=folds, candidate=chosen)
    if scope == "entity" and not filtered.empty:
        entity = st.selectbox("Serie", sorted(filtered.entity.unique()))
        filtered = filtered[filtered.entity == entity]
    elif scope not in ("entity", "total") and not filtered.empty:
        segment = st.selectbox("Segmento", sorted(filtered[scope].dropna().unique()))
        filtered = filtered[filtered[scope] == segment]
    if presentation == "Tabla":
        st.dataframe(filtered, hide_index=True, use_container_width=True)
    elif not filtered.empty and folds:
        targets = list(filtered.target.unique())
        fig = make_subplots(rows=len(targets), cols=len(folds),
                            subplot_titles=[f"{target} · {fold}" for target in targets for fold in folds])
        colors = px.colors.qualitative.Safe
        for i, target in enumerate(targets, 1):
            for j, fold in enumerate(folds, 1):
                part = filtered[(filtered.target == target) & (filtered.fold == fold)]
                for k, candidate in enumerate(chosen):
                    for split, dash in [("train_fixed", "solid"), ("validation", "dash")]:
                        series = part[(part.candidate == candidate) & (part.split == split)].sort_values("epoch")
                        if series.empty:
                            continue
                        fig.add_trace(go.Scatter(x=series.epoch, y=series.value, mode="lines",
                                                 name=f"{candidate} · {split}",
                                                 legendgroup=candidate+split, showlegend=i == 1 and j == 1,
                                                 line=dict(color=colors[k % len(colors)], dash=dash)), row=i, col=j)
                fig.update_yaxes(title_text="Pinball normalizado", row=i, col=j)
                fig.update_xaxes(title_text="Época", row=i, col=j)
        fig.update_layout(height=300*len(targets)+100, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Línea continua: evaluación sobre entrenamiento fijo. Discontinua: evaluación temporal. "
               "FUTURO carece de validación observada. Solo se muestran scopes efectivamente registrados.")
    st.download_button("Descargar loss filtrada", filtered.to_csv(index=False), "learning_curves.csv")

else:
    st.title("Pronósticos y alertas")
    forecasts = results.table("forecasts")
    if forecasts.empty:
        st.info("Sin pronósticos publicados.")
        st.stop()
    entity = st.selectbox("Serie", sorted(forecasts.entity.unique()))
    roles = st.pills("Familias visibles", ["Baseline", "Winner", "Observado"],
                     selection_mode="multi", default=["Baseline", "Winner", "Observado"])
    assigned = results.table("winners", entity=entity)
    st.dataframe(assigned, hide_index=True, use_container_width=True)
    selected = forecasts[(forecasts.entity == entity) & forecasts.role.isin(roles)]
    if presentation == "Tabla":
        st.dataframe(selected, hide_index=True, use_container_width=True)
    else:
        targets = list(forecasts.target.unique())
        fig = make_subplots(rows=len(targets), cols=1, subplot_titles=targets)
        for i, target in enumerate(targets, 1):
            for role in ["Baseline", "Winner"]:
                for q, values in selected[(selected.target == target) & (selected.role == role)].groupby("quantile"):
                    fig.add_trace(go.Scatter(x=values.ds, y=values.value,
                                             name=f"{role} · Q{q:g}", mode="lines",
                                             line=dict(dash="solid" if q == .5 else "dot")), row=i, col=1)
        spec = __import__('json').loads((results.path / "spec.json").read_text())
        history = results.table("history")
        if "Observado" in roles and not history.empty:
            schema = spec["schema"]
            data = history[history[schema["id_col"]] == entity]
            for i, target in enumerate(targets, 1):
                fig.add_trace(go.Scatter(x=data[schema["time_col"]], y=data[target], name=f"Observado · {target}"),
                              row=i, col=1)
        fig.update_layout(height=300*len(targets), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    alerts = results.table("alerts", entity=entity)
    if alerts.empty:
        st.caption("Sin alertas publicadas para esta serie. El monitor no inventa reglas del dominio.")
    else:
        st.dataframe(alerts, hide_index=True, use_container_width=True)
    st.download_button("Descargar pronósticos", selected.to_csv(index=False), "forecasts.csv")

