"""
Logistic Regression con TensorFlow para predecir dirección del precio de acciones
de empresas alimenticias considerando variables climáticas.

El modelo clasifica si el precio subirá (1) o bajará/se mantendrá (0) al día siguiente.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Generación de datos sintéticos
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Simula datos históricos de precios de acciones de empresas alimenticias
    junto con variables climáticas que afectan la cadena de suministro agrícola.

    Variables climáticas incluidas:
      - temperatura_media (°C)
      - precipitacion_mm (mm)
      - indice_sequía (0-10)
      - eventos_extremos (0/1): heladas, inundaciones, etc.
      - radiacion_solar (MJ/m²)

    Variables financieras:
      - precio_actual
      - volumen_transacciones
      - precio_commodity (precio del commodity agrícola principal)
      - retorno_anterior (rendimiento día previo)
    """
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="B")

    # --- Variables climáticas ---
    estacion = (dates.month % 12) // 3  # 0=invierno, 1=primavera, 2=verano, 3=otoño
    temperatura_base = np.where(estacion == 2, 28, np.where(estacion == 0, 5, 18))
    temperatura_media = temperatura_base + np.random.normal(0, 4, n_samples)

    precipitacion_mm = np.abs(
        np.random.normal(40, 25, n_samples) + 20 * (estacion == 1)
    )
    indice_sequia = np.clip(
        10 - precipitacion_mm / 10 + np.random.normal(0, 1, n_samples), 0, 10
    )
    eventos_extremos = (
        (temperatura_media > 38) | (temperatura_media < -2) | (precipitacion_mm > 100)
    ).astype(int)
    radiacion_solar = np.clip(
        15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
        + np.random.normal(0, 2, n_samples),
        2, 30,
    )

    # --- Variables financieras ---
    precio_actual = np.cumsum(np.random.normal(0.05, 1.2, n_samples)) + 100
    precio_actual = np.abs(precio_actual)
    volumen_transacciones = np.abs(np.random.normal(1_000_000, 300_000, n_samples))

    # El precio del commodity agrícola correlaciona con clima
    precio_commodity = (
        50
        + 0.8 * indice_sequia
        - 0.3 * precipitacion_mm / 10
        + 1.5 * eventos_extremos * 5
        + np.random.normal(0, 3, n_samples)
    )
    retorno_anterior = np.roll(np.diff(precio_actual, prepend=precio_actual[0]), 1)

    # --- Variable objetivo: subida de precio al día siguiente ---
    retorno_siguiente = np.roll(
        np.diff(precio_actual, prepend=precio_actual[0]), -1
    )
    # Factores climáticos negativos tienden a presionar precios al alza
    # (escasez de suministros) o a la baja (menor demanda en extremos)
    logit = (
        0.3 * retorno_anterior
        - 0.05 * indice_sequia
        + 0.02 * precipitacion_mm / 10
        - 0.4 * eventos_extremos
        + 0.01 * (temperatura_media - 20)
        + np.random.normal(0, 0.5, n_samples)
    )
    probabilidad = 1 / (1 + np.exp(-logit))
    precio_sube = (retorno_siguiente > 0).astype(int)

    df = pd.DataFrame(
        {
            "fecha": dates,
            "temperatura_media": temperatura_media,
            "precipitacion_mm": precipitacion_mm,
            "indice_sequia": indice_sequia,
            "eventos_extremos": eventos_extremos,
            "radiacion_solar": radiacion_solar,
            "precio_actual": precio_actual,
            "volumen_transacciones": volumen_transacciones,
            "precio_commodity": precio_commodity,
            "retorno_anterior": retorno_anterior,
            "probabilidad_teorica": probabilidad,
            "precio_sube": precio_sube,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------

FEATURES = [
    "temperatura_media",
    "precipitacion_mm",
    "indice_sequia",
    "eventos_extremos",
    "radiacion_solar",
    "precio_actual",
    "volumen_transacciones",
    "precio_commodity",
    "retorno_anterior",
]
TARGET = "precio_sube"


def preprocess(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 3. Modelo de Regresión Logística con TensorFlow/Keras
# ---------------------------------------------------------------------------

def build_logistic_model(n_features: int) -> tf.keras.Model:
    """
    Regresión logística implementada como red neuronal de una sola capa densa
    con activación sigmoide — equivalente exacto de regresión logística.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
                name="logistic_layer",
            ),
        ],
        name="LogisticRegression_StockWeather",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Entrenamiento
# ---------------------------------------------------------------------------

def train(model: tf.keras.Model, X_train, y_train, X_test, y_test):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    return history


# ---------------------------------------------------------------------------
# 5. Evaluación y visualización
# ---------------------------------------------------------------------------

def evaluate(model: tf.keras.Model, X_test, y_test, history):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("  RESULTADOS DE EVALUACIÓN")
    print("=" * 60)

    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = ["loss", "accuracy", "auc", "precision", "recall"]
    for name, val in zip(metric_names, results):
        print(f"  {name:>12}: {val:.4f}")

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  ROC-AUC (sklearn): {auc:.4f}")

    print("\n  Reporte de Clasificación:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Baja/Estable", "Sube"]
        )
    )

    print("  Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[VP={cm[0,0]}  FN={cm[0,1]}]")
    print(f"   [FP={cm[1,0]}  VN={cm[1,1]}]]")

    _plot_results(history, y_test, y_prob, cm)


def _plot_results(history, y_test, y_prob, cm):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Regresión Logística: Predicción de Acciones Alimenticias + Clima",
        fontsize=13,
        fontweight="bold",
    )

    # Curva de aprendizaje
    ax = axes[0]
    ax.plot(history.history["loss"], label="Train Loss", color="#1f77b4")
    ax.plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--")
    ax.set_title("Curva de Pérdida (Binary Crossentropy)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Matriz de confusión
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Baja/Estable", "Sube"])
    ax.set_yticklabels(["Baja/Estable", "Sube"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)

    # Distribución de probabilidades predichas
    ax = axes[2]
    ax.hist(
        y_prob[y_test == 0], bins=40, alpha=0.6, label="Baja/Estable", color="#d62728"
    )
    ax.hist(
        y_prob[y_test == 1], bins=40, alpha=0.6, label="Sube", color="#2ca02c"
    )
    ax.axvline(0.5, color="black", linestyle="--", label="Umbral 0.5")
    ax.set_title("Distribución de Probabilidades Predichas")
    ax.set_xlabel("P(precio sube)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados_modelo.png", dpi=150, bbox_inches="tight")
    print("\n  Gráfico guardado: resultados_modelo.png")


# ---------------------------------------------------------------------------
# 6. Interpretación de coeficientes
# ---------------------------------------------------------------------------

def interpret_coefficients(model: tf.keras.Model):
    weights, bias = model.get_layer("logistic_layer").get_weights()
    coef = weights.flatten()

    print("\n" + "=" * 60)
    print("  IMPORTANCIA DE VARIABLES (Coeficientes del Modelo)")
    print("=" * 60)
    importance = sorted(zip(FEATURES, coef), key=lambda x: abs(x[1]), reverse=True)
    for feature, coef_val in importance:
        bar = "█" * int(abs(coef_val) * 20)
        sign = "+" if coef_val > 0 else "-"
        print(f"  {feature:>25}: {sign}{abs(coef_val):.4f}  {bar}")
    print(f"\n  Sesgo (bias): {bias[0]:.4f}")


# ---------------------------------------------------------------------------
# 7. Predicción de ejemplo
# ---------------------------------------------------------------------------

def predict_example(model: tf.keras.Model, scaler: StandardScaler):
    print("\n" + "=" * 60)
    print("  PREDICCIÓN DE EJEMPLO")
    print("=" * 60)

    escenarios = {
        "Día normal (otoño)": [18.0, 35.0, 4.0, 0, 15.0, 105.0, 950_000, 52.0, 0.3],
        "Sequía severa": [35.0, 2.0, 9.5, 1, 28.0, 105.0, 1_200_000, 65.0, -0.8],
        "Lluvia excesiva / inundación": [12.0, 150.0, 1.0, 1, 8.0, 98.0, 1_500_000, 58.0, -1.2],
        "Condiciones ideales": [22.0, 55.0, 3.0, 0, 18.0, 110.0, 900_000, 50.0, 1.1],
    }

    for nombre, valores in escenarios.items():
        x = scaler.transform([valores])
        prob = model.predict(x, verbose=0)[0][0]
        decision = "SUBE" if prob >= 0.5 else "BAJA/ESTABLE"
        confianza = prob if prob >= 0.5 else 1 - prob
        print(f"\n  Escenario: {nombre}")
        print(f"    Temperatura: {valores[0]}°C | Precipitación: {valores[1]}mm | "
              f"Sequía: {valores[2]:.1f} | Evento extremo: {'Sí' if valores[3] else 'No'}")
        print(f"    → Predicción: {decision}  (P={prob:.3f}, confianza {confianza:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  REGRESIÓN LOGÍSTICA CON TENSORFLOW")
    print("  Predicción de Acciones - Empresas Alimenticias + Clima")
    print("=" * 60)

    print("\n[1/5] Generando datos sintéticos...")
    df = generate_synthetic_data(n_samples=2000)
    print(f"      {len(df)} muestras | Distribución objetivo: "
          f"{df[TARGET].value_counts().to_dict()}")

    print("\n[2/5] Preprocesando datos...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[3/5] Construyendo modelo...")
    model = build_logistic_model(n_features=len(FEATURES))
    model.summary()

    print("\n[4/5] Entrenando modelo...")
    history = train(model, X_train, y_train, X_test, y_test)
    print(f"      Entrenamiento completado en {len(history.history['loss'])} épocas.")

    print("\n[5/5] Evaluando modelo...")
    evaluate(model, X_test, y_test, history)

    interpret_coefficients(model)
    predict_example(model, scaler)

    model.save("modelo_logistico_acciones_clima.keras")
    print("\n  Modelo guardado: modelo_logistico_acciones_clima.keras")
    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
