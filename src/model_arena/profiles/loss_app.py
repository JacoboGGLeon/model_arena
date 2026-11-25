from io import BytesIO
from pathlib import Path
import os
import polars as pl
import streamlit as st
from model_arena.profiles.loss_queries import curves, figure

st.set_page_config(page_title="Observatorio · entrenamiento", layout="wide")
project = Path(os.environ["OBS_LOSS_PROJECT"])
st.title("Modelo y entrenamiento")
st.caption("Lectura del run: " + str(project) + " · Este visor no entrena ni publica.")
with st.sidebar:
    inp = st.radio("Input", ["X0", "X1", "X2", "X3", "Todos"], horizontal=True)
    fold = st.radio("Ajuste", ["Todos", "F1", "F2", "F3", "FUTURO"])
    scope = st.radio("Desglose", ["Total", "Budget", "Cuenta"])
    value = None
    inventory = project / "run/curvas/inventory.parquet"
    if scope != "Total":
        if not inventory.exists():
            st.info("El inventario todavía no está listo")
            st.stop()
        inv = pl.read_parquet(inventory)
        value = st.selectbox(
            scope,
            sorted(inv["budget" if scope == "Budget" else "account_id"].unique().drop_nulls().to_list()),
        )
    kind = (
        st.radio("Medición", ["Ventanas fijas", "Entrenamiento muestreado"])
        if scope == "Total"
        else "Ventanas fijas"
    )
    presentation = st.radio("Presentación", ["Gráfico", "Tabla"])
    st.button("Actualizar")
    import shutil

    st.metric("Disco libre", f"{shutil.disk_usage(project).free / 2**30:.2f} GiB")
frame = curves(
    project,
    inp=inp,
    folds=None if fold == "Todos" else [fold],
    scope=scope,
    value=value,
    sampled=kind != "Ventanas fijas",
)
if frame.is_empty():
    st.info("Aún no hay curvas guardadas para estos filtros.")
    st.stop()
options = sorted(frame["candidate"].unique().to_list())
highlight = st.selectbox("Resaltar candidato", ["Ninguno", *options])
st.caption(
    "X: época · Y: pinball normalizado, menor es mejor. Cada curva identifica un candidato y ajuste en el hover. Colores: verde = entrenamiento; rojo discontinuo = validación."
)
if kind == "Ventanas fijas":
    st.caption(
        "Las ventanas de medición se mantienen fijas. F1–F3 participan en HPO/selección; no son un test independiente. FUTURO no tiene validación observada."
    )
else:
    st.caption(
        "Se muestrean orígenes distintos en cada época. Esta medición puede ser más variable que la evaluación fija."
    )
if presentation == "Gráfico":
    st.plotly_chart(figure(frame, None if highlight == "Ninguno" else highlight), use_container_width=True)
else:
    st.dataframe(frame, use_container_width=True, hide_index=True)
audit_path = project / "run/curvas/hpo_trials.parquet"
if audit_path.is_file():
    with st.expander("Estados HPO · completados y podados"):
        audit = pl.read_parquet(audit_path)
        audit = audit.with_columns(pl.col("parameters_json").str.json_path_match("$.encoder_kind")
                                  .fill_null("tft_compact").alias("encoder_kind"))
        if inp != "Todos":
            audit = audit.filter(pl.col("input") == inp)
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.caption("PRUNED conserva curvas parciales y no participa en la selección de ganadores.")
buffer = BytesIO()
frame.write_parquet(buffer, compression="zstd")
st.download_button(
    "Descargar loss filtradas · Parquet", data=buffer.getvalue(), file_name="loss_filtradas.parquet"
)
