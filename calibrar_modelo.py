import joblib
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from preprocesamiento import preparar_datos

# Rango permitido para la probabilidad calibrada de fuga
PROB_MIN = 0.01
PROB_MAX = 0.99


def calibrar():
    """
    Calibra las probabilidades de la red SIN reentrenarla.

    Al entrenar con class_weight='balanced' la red sobreestima la probabilidad de fuga.
    Ajustamos una regresion isotonica que traduce la probabilidad de la red a una
    probabilidad cercana a la frecuencia real de abandono.
    """
    X_train, X_test, y_train, y_test, _, _ = preparar_datos()
    modelo = tf.keras.models.load_model('modelo_churn.keras')

    # Keras reservo el ultimo 20% de X_train como validacion durante el entrenamiento
    # (validation_split=0.20). Calibramos con ese tramo y evaluamos con el test, que
    # la red y el calibrador nunca vieron.
    corte = int(len(X_train) * 0.80)
    X_cal, y_cal = X_train[corte:], y_train[corte:]

    p_cal = modelo.predict(X_cal, verbose=0).flatten()
    p_test = modelo.predict(X_test, verbose=0).flatten()

    # La isotonica es una funcion escalonada y puede devolver exactamente 0 o 1, que no son
    # probabilidades creibles para un cliente individual. Limitamos el rango a [0.01, 0.99].
    calibrador = IsotonicRegression(out_of_bounds='clip', y_min=PROB_MIN, y_max=PROB_MAX)
    calibrador.fit(p_cal, y_cal)
    p_test_cal = calibrador.predict(p_test)

    antes = {
        'roc_auc': roc_auc_score(y_test, p_test),
        'pr_auc': average_precision_score(y_test, p_test),
        'brier': brier_score_loss(y_test, p_test),
    }
    despues = {
        'roc_auc': roc_auc_score(y_test, p_test_cal),
        'pr_auc': average_precision_score(y_test, p_test_cal),
        'brier': brier_score_loss(y_test, p_test_cal),
    }

    print(f"Fuga real en test:       {y_test.mean():.4f}")
    print(f"Prob. media ANTES:       {p_test.mean():.4f}")
    print(f"Prob. media DESPUES:     {p_test_cal.mean():.4f}")
    print(f"Brier   antes / despues: {antes['brier']:.4f} / {despues['brier']:.4f}")
    print(f"ROC-AUC antes / despues: {antes['roc_auc']:.4f} / {despues['roc_auc']:.4f}")
    print(f"PR-AUC  antes / despues: {antes['pr_auc']:.4f} / {despues['pr_auc']:.4f}")

    joblib.dump(calibrador, 'calibrador.joblib')
    print("Guardado: calibrador.joblib")

    # Actualizamos las metricas que muestra la app con las de las probabilidades calibradas,
    # conservando las originales por trazabilidad.
    try:
        metricas = joblib.load('metricas_tecnicas.joblib')
    except FileNotFoundError:
        metricas = {}
    for clave, valor in (('roc_auc', antes['roc_auc']), ('pr_auc', antes['pr_auc']),
                         ('brier_score', antes['brier']), ('calibracion', 1.0 - antes['brier'])):
        metricas.setdefault(clave + '_sin_calibrar', float(valor))
    metricas['roc_auc'] = float(despues['roc_auc'])
    metricas['pr_auc'] = float(despues['pr_auc'])
    metricas['brier_score'] = float(despues['brier'])
    metricas['calibracion'] = float(1.0 - despues['brier'])
    metricas['prob_media_test_sin_calibrar'] = float(p_test.mean())
    metricas['prob_media_test_calibrada'] = float(p_test_cal.mean())
    metricas['fuga_real_test'] = float(y_test.mean())
    joblib.dump(metricas, 'metricas_tecnicas.joblib')
    print("Actualizado: metricas_tecnicas.joblib")


if __name__ == '__main__':
    calibrar()
