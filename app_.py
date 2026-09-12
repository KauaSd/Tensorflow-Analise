"""
Dashboard de Análise e Previsão de Vendas

Tecnologias:
    - Python
    - Pandas
    - NumPy
    - TensorFlow / Keras
    - Streamlit

Funcionalidades:
    - Dataset em formato de dicionário Python
    - Validação automática
    - Limpeza de dados
    - Análise estatística
    - Indicadores de vendas
    - Visualização interativa
    - Feature Engineering para séries temporais
    - Modelo de regressão com TensorFlow
    - Avaliação do modelo
    - Previsão de vendas futuras
    - Comparação entre vendas reais e previstas

Executar:

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# CONFIGURAÇÕES
# ============================================================

APP_TITLE = "SalesVision AI"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# CONFIGURAÇÃO DO STREAMLIT
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATASET
# ============================================================
#
# Substitua este dicionário pelo seu dataset.
#
# IMPORTANTE:
# Todas as colunas precisam possuir a mesma quantidade
# de elementos.
#
# Cada posição representa uma linha:
#
# data[0]   -> vendas[0]
# data[1]   -> vendas[1]
# data[2]   -> vendas[2]
#
# ============================================================

DATASET = {
    "data": [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-10",
        "2026-01-11",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-17",
        "2026-01-18",
        "2026-01-19",
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
        "2026-01-24",
        "2026-01-25",
        "2026-01-26",
        "2026-01-27",
        "2026-01-28",
        "2026-01-29",
        "2026-01-30",
        "2026-01-31",
        "2026-02-01",
        "2026-02-02",
        "2026-02-03",
        "2026-02-04",
        "2026-02-05",
        "2026-02-06",
        "2026-02-07",
        "2026-02-08",
        "2026-02-09",
        "2026-02-10",
        "2026-02-11",
        "2026-02-12",
        "2026-02-13",
        "2026-02-14",
        "2026-02-15",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
        "2026-02-19",
        "2026-02-20",
        "2026-02-21",
        "2026-02-22",
        "2026-02-23",
        "2026-02-24",
        "2026-02-25",
        "2026-02-26",
        "2026-02-27",
        "2026-02-28",
        "2026-03-01",
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
        "2026-03-05",
        "2026-03-06",
        "2026-03-07",
        "2026-03-08",
        "2026-03-09",
        "2026-03-10",
    ],
    "vendas": [
        120,
        128,
        135,
        118,
        142,
        150,
        147,
        155,
        162,
        158,
        170,
        175,
        168,
        180,
        187,
        192,
        185,
        198,
        205,
        201,
        214,
        220,
        215,
        228,
        235,
        231,
        242,
        248,
        252,
        258,
        263,
        270,
        275,
        269,
        283,
        290,
        286,
        297,
        305,
        300,
        312,
        318,
        315,
        328,
        335,
        330,
        342,
        350,
        346,
        357,
        365,
        360,
        372,
        380,
        375,
        388,
        395,
        390,
        402,
        410,
        405,
        418,
        425,
        420,
        432,
        440,
        435,
        448,
        455,
    ],
}


# ============================================================
# FUNÇÕES DE DADOS
# ============================================================


def validate_dataset(dataset: dict) -> None:
    """
    Valida a estrutura básica do dataset.
    """

    if not isinstance(dataset, dict):
        raise ValueError(
            "O dataset precisa ser um dicionário Python."
        )

    if len(dataset) == 0:
        raise ValueError("O dataset está vazio.")

    sizes = {}

    for column, values in dataset.items():
        if not hasattr(values, "__len__"):
            raise ValueError(
                f"A coluna '{column}' não é uma lista válida."
            )

        sizes[column] = len(values)

    unique_sizes = set(sizes.values())

    if len(unique_sizes) > 1:
        details = "\n".join(
            f"- {column}: {size} valores"
            for column, size in sizes.items()
        )

        raise ValueError(
            "As colunas possuem tamanhos diferentes.\n\n"
            f"{details}\n\n"
            "Todas as colunas precisam ter o mesmo número "
            "de elementos."
        )

    if "data" not in dataset:
        raise ValueError(
            "O dataset precisa possuir a coluna 'data'."
        )

    if "vendas" not in dataset:
        raise ValueError(
            "O dataset precisa possuir a coluna 'vendas'."
        )


@st.cache_data
def load_dataset(dataset: dict) -> pd.DataFrame:
    """
    Converte o dicionário para DataFrame e realiza
    a limpeza inicial.
    """

    validate_dataset(dataset)

    df = pd.DataFrame(dataset)

    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    df["vendas"] = pd.to_numeric(
        df["vendas"],
        errors="coerce",
    )

    # Remove registros inválidos.
    df = df.dropna(
        subset=["data", "vendas"]
    ).copy()

    # Não permitimos vendas negativas.
    df = df[df["vendas"] >= 0].copy()

    if df.empty:
        raise ValueError(
            "Não existem registros válidos após a limpeza."
        )

    # Ordenação temporal.
    df = df.sort_values("data").reset_index(drop=True)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features para previsão de séries temporais.

    Features principais:

        lag_1
        lag_2
        lag_3
        lag_7
        rolling_mean_7
        rolling_std_7
        dia_semana
        mes
        dia_do_ano
    """

    data = df.copy()

    data["dia_semana"] = (
        data["data"].dt.dayofweek
    )

    data["mes"] = (
        data["data"].dt.month
    )

    data["dia_do_ano"] = (
        data["data"].dt.dayofyear
    )

    data["lag_1"] = (
        data["vendas"].shift(1)
    )

    data["lag_2"] = (
        data["vendas"].shift(2)
    )

    data["lag_3"] = (
        data["vendas"].shift(3)
    )

    data["lag_7"] = (
        data["vendas"].shift(7)
    )

    data["rolling_mean_7"] = (
        data["vendas"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["rolling_std_7"] = (
        data["vendas"]
        .shift(1)
        .rolling(7)
        .std()
    )

    data = data.dropna().reset_index(drop=True)

    return data


# ============================================================
# MACHINE LEARNING
# ============================================================


FEATURES = [
    "dia_semana",
    "mes",
    "dia_do_ano",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
]


def build_model(input_shape: int) -> tf.keras.Model:
    """
    Cria o modelo de regressão com TensorFlow.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.Input(
                shape=(input_shape,)
            ),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                0.10
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
    )

    return model


def train_model(
    feature_df: pd.DataFrame,
    epochs: int,
) -> tuple[
    tf.keras.Model,
    dict,
    pd.DataFrame,
]:
    """
    Treina o modelo mantendo a ordem cronológica.

    Não utilizamos shuffle porque estamos trabalhando
    com uma série temporal.
    """

    x = feature_df[FEATURES].astype(
        np.float32
    ).values

    y = feature_df["vendas"].astype(
        np.float32
    ).values

    if len(x) < 20:
        raise ValueError(
            "É necessário possuir pelo menos 20 "
            "registros válidos para o treinamento."
        )

    split_index = int(len(x) * 0.80)

    x_train = x[:split_index]
    y_train = y[:split_index]

    x_test = x[split_index:]
    y_test = y[split_index:]

    # Normalização.
    normalizer = tf.keras.layers.Normalization(
        axis=-1
    )

    normalizer.adapt(x_train)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(
                shape=(len(FEATURES),)
            ),
            normalizer,

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                0.10
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=8,
        shuffle=False,
        verbose=0,
        callbacks=[early_stopping],
    )

    evaluation = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    predictions = model.predict(
        x_test,
        verbose=0,
    ).reshape(-1)

    predictions = np.maximum(
        predictions,
        0,
    )

    mae = float(evaluation[1])

    # MAPE seguro para evitar divisão por zero.
    denominator = np.maximum(
        np.abs(y_test),
        1e-7,
    )

    mape = float(
        np.mean(
            np.abs(
                (y_test - predictions)
                / denominator
            )
        )
        * 100
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (y_test - predictions) ** 2
            )
        )
    )

    test_results = feature_df.iloc[
        split_index:
    ].copy()

    test_results["previsao"] = predictions

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "epochs": len(
            history.history["loss"]
        ),
        "treino": len(x_train),
        "teste": len(x_test),
    }

    return model, metrics, test_results


def predict_future(
    model: tf.keras.Model,
    original_df: pd.DataFrame,
    days: int,
) -> pd.DataFrame:
    """
    Faz previsão recursiva para os próximos dias.

    A previsão de amanhã passa a alimentar
    a previsão do dia seguinte.
    """

    if days <= 0:
        raise ValueError(
            "A quantidade de dias precisa ser maior que zero."
        )

    data = original_df.copy()

    predictions = []

    for _ in range(days):
        next_date = (
            data["data"].max()
            + pd.Timedelta(days=1)
        )

        previous_sales = (
            data["vendas"]
            .astype(float)
            .values
        )

        if len(previous_sales) < 7:
            raise ValueError(
                "São necessários pelo menos 7 registros "
                "para realizar previsão."
            )

        lag_1 = previous_sales[-1]
        lag_2 = previous_sales[-2]
        lag_3 = previous_sales[-3]
        lag_7 = previous_sales[-7]

        rolling_values = previous_sales[-7:]

        rolling_mean = float(
            np.mean(rolling_values)
        )

        rolling_std = float(
            np.std(rolling_values)
        )

        features = np.array(
            [
                [
                    next_date.dayofweek,
                    next_date.month,
                    next_date.dayofyear,
                    lag_1,
                    lag_2,
                    lag_3,
                    lag_7,
                    rolling_mean,
                    rolling_std,
                ]
            ],
            dtype=np.float32,
        )

        prediction = float(
            model.predict(
                features,
                verbose=0,
            )[0][0]
        )

        prediction = max(
            prediction,
            0,
        )

        predictions.append(
            {
                "data": next_date,
                "previsao_vendas": prediction,
            }
        )

        # Adiciona a previsão ao histórico para que ela
        # seja utilizada na próxima iteração.
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "data": [next_date],
                        "vendas": [prediction],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


# ============================================================
# FUNÇÕES DE ANÁLISE
# ============================================================


def calculate_statistics(
    df: pd.DataFrame,
) -> dict:
    """
    Calcula indicadores principais.
    """

    total = float(
        df["vendas"].sum()
    )

    average = float(
        df["vendas"].mean()
    )

    median = float(
        df["vendas"].median()
    )

    minimum = float(
        df["vendas"].min()
    )

    maximum = float(
        df["vendas"].max()
    )

    standard_deviation = float(
        df["vendas"].std()
    )

    best_day = df.loc[
        df["vendas"].idxmax()
    ]

    worst_day = df.loc[
        df["vendas"].idxmin()
    ]

    return {
        "total": total,
        "media": average,
        "mediana": median,
        "minimo": minimum,
        "maximo": maximum,
        "desvio": standard_deviation,
        "melhor_data": best_day["data"],
        "melhor_valor": best_day["vendas"],
        "pior_data": worst_day["data"],
        "pior_valor": worst_day["vendas"],
    }


def calculate_growth(
    df: pd.DataFrame,
) -> float:
    """
    Calcula crescimento percentual entre o primeiro
    e o último registro.
    """

    first_value = float(
        df.iloc[0]["vendas"]
    )

    last_value = float(
        df.iloc[-1]["vendas"]
    )

    if first_value == 0:
        return 0.0

    return (
        (last_value - first_value)
        / first_value
    ) * 100


# ============================================================
# SIDEBAR
# ============================================================


def render_sidebar() -> tuple[int, int]:
    """
    Renderiza controles laterais.
    """

    with st.sidebar:
        st.title("⚙️ Configurações")

        st.subheader("Modelo")

        epochs = st.slider(
            "Épocas de treinamento",
            min_value=50,
            max_value=500,
            value=250,
            step=50,
        )

        forecast_days = st.slider(
            "Dias para prever",
            min_value=1,
            max_value=30,
            value=7,
        )

        st.divider()

        st.subheader("Dataset")

        st.info(
            "O dataset atual está definido diretamente "
            "no arquivo app.py."
        )

        st.divider()

        st.caption(
            "SalesVision AI • TensorFlow + Pandas + Streamlit"
        )

    return epochs, forecast_days


# ============================================================
# DASHBOARD
# ============================================================


def render_header() -> None:
    """
    Cabeçalho principal.
    """

    st.title("📈 SalesVision AI")

    st.markdown(
        """
        ### Análise e previsão de vendas com Machine Learning

        O sistema transforma os dados históricos em informações
        úteis para responder três perguntas:

        **O que aconteceu? → O que está acontecendo? → O que pode acontecer?**
        """
    )


def render_overview(
    df: pd.DataFrame,
) -> None:
    """
    Exibe os principais indicadores.
    """

    stats = calculate_statistics(df)

    growth = calculate_growth(df)

    columns = st.columns(5)

    with columns[0]:
        st.metric(
            "💰 Total vendido",
            f"{stats['total']:,.0f}",
        )

    with columns[1]:
        st.metric(
            "📊 Média diária",
            f"{stats['media']:,.1f}",
        )

    with columns[2]:
        st.metric(
            "🚀 Crescimento",
            f"{growth:+.1f}%",
        )

    with columns[3]:
        st.metric(
            "🏆 Maior venda",
            f"{stats['maximo']:,.0f}",
        )

    with columns[4]:
        st.metric(
            "📦 Registros",
            f"{len(df)}",
        )


def render_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Aba de análise exploratória.
    """

    st.header("🔎 Análise dos dados")

    stats = calculate_statistics(df)

    column_left, column_right = st.columns(2)

    with column_left:
        st.subheader("Distribuição das vendas")

        st.line_chart(
            df.set_index("data")[
                ["vendas"]
            ]
        )

    with column_right:
        st.subheader("Vendas por dia da semana")

        weekday_names = {
            0: "Segunda",
            1: "Terça",
            2: "Quarta",
            3: "Quinta",
            4: "Sexta",
            5: "Sábado",
            6: "Domingo",
        }

        weekday_df = df.copy()

        weekday_df["dia_semana"] = (
            weekday_df["data"].dt.dayofweek
        )

        weekday_df["dia_nome"] = (
            weekday_df["dia_semana"]
            .map(weekday_names)
        )

        weekday_sales = (
            weekday_df
            .groupby(
                ["dia_semana", "dia_nome"]
            )["vendas"]
            .mean()
            .reset_index()
            .sort_values("dia_semana")
        )

        st.bar_chart(
            weekday_sales.set_index(
                "dia_nome"
            )["vendas"]
        )

    st.divider()

    st.subheader("📋 Estatísticas")

    statistics_df = pd.DataFrame(
        {
            "Métrica": [
                "Total",
                "Média",
                "Mediana",
                "Mínimo",
                "Máximo",
                "Desvio padrão",
            ],
            "Valor": [
                stats["total"],
                stats["media"],
                stats["mediana"],
                stats["minimo"],
                stats["maximo"],
                stats["desvio"],
            ],
        }
    )

    st.dataframe(
        statistics_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🗓️ Dias extremos")

    extreme_left, extreme_right = st.columns(2)

    with extreme_left:
        st.success(
            "🏆 Melhor dia"
        )

        st.write(
            f"**Data:** "
            f"{stats['melhor_data'].strftime('%d/%m/%Y')}"
        )

        st.write(
            f"**Vendas:** {stats['melhor_valor']:,.2f}"
        )

    with extreme_right:
        st.warning(
            "📉 Pior dia"
        )

        st.write(
            f"**Data:** "
            f"{stats['pior_data'].strftime('%d/%m/%Y')}"
        )

        st.write(
            f"**Vendas:** {stats['pior_valor']:,.2f}"
        )


def render_data_table(
    df: pd.DataFrame,
) -> None:
    """
    Exibe os dados brutos.
    """

    st.header("🗂️ Dataset")

    st.caption(
        "Dados utilizados pela aplicação após a limpeza."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_training_metrics(
    metrics: dict,
) -> None:
    """
    Exibe métricas do modelo.
    """

    st.header("🧠 Resultado do modelo")

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "MAE",
            f"{metrics['mae']:.2f}",
        )

    with columns[1]:
        st.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}",
        )

    with columns[2]:
        st.metric(
            "MAPE",
            f"{metrics['mape']:.2f}%",
        )

    with columns[3]:
        st.metric(
            "Épocas",
            f"{metrics['epochs']}",
        )

    st.caption(
        """
        **MAE:** erro médio absoluto.

        **RMSE:** penaliza erros grandes com maior intensidade.

        **MAPE:** erro percentual médio.
        """
    )


def render_model_results(
    test_results: pd.DataFrame,
) -> None:
    """
    Exibe comparação entre valores reais e previstos.
    """

    st.subheader("🎯 Real x previsto")

    comparison = test_results[
        [
            "data",
            "vendas",
            "previsao",
        ]
    ].copy()

    comparison = comparison.set_index(
        "data"
    )

    comparison.columns = [
        "Vendas reais",
        "Previsão",
    ]

    st.line_chart(
        comparison
    )

    st.dataframe(
        comparison.reset_index(),
        use_container_width=True,
        hide_index=True,
    )


def render_future_prediction(
    prediction_df: pd.DataFrame,
) -> None:
    """
    Exibe previsão futura.
    """

    st.header("🔮 Previsão futura")

    total_prediction = float(
        prediction_df[
            "previsao_vendas"
        ].sum()
    )

    average_prediction = float(
        prediction_df[
            "previsao_vendas"
        ].mean()
    )

    maximum_prediction = float(
        prediction_df[
            "previsao_vendas"
        ].max()
    )

    columns = st.columns(3)

    with columns[0]:
        st.metric(
            "📦 Total previsto",
            f"{total_prediction:,.0f}",
        )

    with columns[1]:
        st.metric(
            "📊 Média diária",
            f"{average_prediction:,.1f}",
        )

    with columns[2]:
        st.metric(
            "🚀 Pico previsto",
            f"{maximum_prediction:,.1f}",
        )

    st.subheader(
        "Evolução prevista"
    )

    chart_df = prediction_df.copy()

    chart_df = chart_df.set_index(
        "data"
    )

    st.line_chart(
        chart_df[
            ["previsao_vendas"]
        ]
    )

    st.subheader(
        "📅 Próximos dias"
    )

    display_df = prediction_df.copy()

    display_df["data"] = (
        display_df["data"]
        .dt.strftime("%d/%m/%Y")
    )

    display_df["previsao_vendas"] = (
        display_df["previsao_vendas"]
        .round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Função principal da aplicação.
    """

    render_header()

    epochs, forecast_days = (
        render_sidebar()
    )

    # --------------------------------------------------------
    # CARREGAMENTO
    # --------------------------------------------------------

    try:
        df = load_dataset(DATASET)

    except ValueError as error:
        st.error(
            f"❌ Erro no dataset: {error}"
        )

        st.stop()

    except Exception as error:
        st.error(
            "❌ Erro inesperado ao carregar o dataset."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    render_overview(df)

    st.divider()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab_analysis, tab_data, tab_model = st.tabs(
        [
            "📊 Análise",
            "🗂️ Dados",
            "🧠 Machine Learning",
        ]
    )

    # --------------------------------------------------------
    # ANÁLISE
    # --------------------------------------------------------

    with tab_analysis:
        render_analysis(df)

    # --------------------------------------------------------
    # DADOS
    # --------------------------------------------------------

    with tab_data:
        render_data_table(df)

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    with tab_model:
        st.header("🧠 Machine Learning")

        st.markdown(
            """
            O modelo utiliza o histórico para encontrar padrões
            temporais nas vendas.

            As principais informações utilizadas são:

            - **Lag 1:** venda do dia anterior
            - **Lag 2:** venda de dois dias atrás
            - **Lag 3:** venda de três dias atrás
            - **Lag 7:** venda de uma semana atrás
            - **Média móvel:** comportamento recente
            - **Desvio padrão:** volatilidade recente
            - **Dia da semana**
            - **Mês**
            - **Dia do ano**
            """
        )

        feature_df = create_features(df)

        if len(feature_df) < 20:
            st.warning(
                "⚠️ Poucos dados disponíveis. "
                "Adicione mais registros para obter "
                "um treinamento mais confiável."
            )

        st.subheader(
            "Features utilizadas"
        )

        st.dataframe(
            feature_df[
                [
                    "data",
                    "vendas",
                    *FEATURES,
                ]
            ].tail(10),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        if st.button(
            "🚀 Treinar modelo",
            type="primary",
            use_container_width=True,
        ):
            progress = st.progress(
                0,
                text="Preparando treinamento..."
            )

            try:
                progress.progress(
                    25,
                    text="Criando modelo..."
                )

                model, metrics, test_results = (
                    train_model(
                        feature_df,
                        epochs,
                    )
                )

                progress.progress(
                    100,
                    text="Treinamento concluído."
                )

                st.session_state[
                    "model"
                ] = model

                st.session_state[
                    "metrics"
                ] = metrics

                st.session_state[
                    "test_results"
                ] = test_results

                st.session_state[
                    "trained"
                ] = True

                st.success(
                    "✅ Modelo treinado com sucesso!"
                )

            except Exception as error:
                progress.empty()

                st.error(
                    "❌ Não foi possível treinar "
                    "o modelo."
                )

                st.exception(error)

        # ----------------------------------------------------
        # RESULTADOS DO TREINAMENTO
        # ----------------------------------------------------

        if st.session_state.get(
            "trained",
            False,
        ):
            metrics = st.session_state[
                "metrics"
            ]

            test_results = st.session_state[
                "test_results"
            ]

            render_training_metrics(
                metrics
            )

            render_model_results(
                test_results
            )

            st.divider()

            # ------------------------------------------------
            # PREVISÃO FUTURA
            # ------------------------------------------------

            if st.button(
                "🔮 Gerar previsão",
                type="secondary",
                use_container_width=True,
            ):
                model = st.session_state[
                    "model"
                ]

                try:
                    prediction_df = (
                        predict_future(
                            model,
                            df,
                            forecast_days,
                        )
                    )

                    st.session_state[
                        "prediction"
                    ] = prediction_df

                    st.success(
                        "✅ Previsão gerada!"
                    )

                except Exception as error:
                    st.error(
                        "❌ Erro ao gerar previsão."
                    )

                    st.exception(error)

            if (
                "prediction"
                in st.session_state
            ):
                render_future_prediction(
                    st.session_state[
                        "prediction"
                    ]
                )


# ============================================================
# EXECUÇÃO
# ============================================================


if __name__ == "__main__":
    main()