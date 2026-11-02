# Verificación en R de la Metodología de Superficie de Respuesta (RSM)
# Compara estos resultados con los de la página "RSM" de la app.
# Uso: Rscript verificacion_rsm.R   (ejecutar desde la carpeta verificacion/)

# Instala el paquete "rsm" solo si no está disponible en este equipo
if (!requireNamespace("rsm", quietly = TRUE)) {
  install.packages("rsm", repos = "https://cloud.r-project.org")
}

# Carga el paquete: aporta rsm(), steepest() y canonical()
library(rsm)

# ---------------------------------------------------------------------------
# PARTE 1: modelo de primer orden (diseño factorial 2^2 con puntos centrales)
# ---------------------------------------------------------------------------

# Lee los datos con variables ya codificadas (x1, x2) y la respuesta (eficiencia)
datos1 <- read.csv("../data/rsm1_mi_variante.csv")

# Ajusta el plano de primer orden: eficiencia = b0 + b1*x1 + b2*x2
# FO(x1, x2) indica "first order" (solo efectos lineales)
modelo1 <- rsm(eficiencia ~ FO(x1, x2), data = datos1)

# Resumen: coeficientes (b0, b1, b2), R², y el análisis de varianza con falta de ajuste
# (compara b0, b1, b2 y R² con la pestaña "1. Primer orden" de la app)
print(summary(modelo1))

# Trayectoria de máximo ascenso: pasos en la dirección de mayor aumento de la respuesta
# dist = distancias desde el centro (en unidades codificadas) donde se calcula cada paso
cat("\nTrayectoria de máximo ascenso:\n")
print(steepest(modelo1, dist = seq(0, 5, by = 1)))

# ---------------------------------------------------------------------------
# PARTE 2: modelo de segundo orden (diseño central compuesto)
# ---------------------------------------------------------------------------

# Lee los datos del diseño central compuesto (x1, x2 codificadas y eficiencia)
datos2 <- read.csv("../data/ccd_mi_variante.csv")

# Ajusta el modelo cuadrático completo:
# eficiencia = b0 + b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2 + b12*x1*x2
# SO(x1, x2) indica "second order" (lineales, cuadráticos e interacción)
modelo2 <- rsm(eficiencia ~ SO(x1, x2), data = datos2)

# Resumen: coeficientes, R², R² ajustado y prueba de falta de ajuste
# (compara con la pestaña "3. Segundo orden" de la app)
print(summary(modelo2))

# Análisis canónico: punto estacionario (xs), respuesta en ese punto (ys)
# y valores propios (lambda); todos negativos = máximo, todos positivos = mínimo,
# signos distintos = punto de silla (compara con la pestaña "4. Punto estacionario")
cat("\nAnálisis canónico:\n")
print(canonical(modelo2))

# Superficie 3D del modelo de segundo orden: x1 en un eje, x2 en el otro, respuesta en altura
# (con Rscript los gráficos se guardan en el archivo Rplots.pdf)
persp(modelo2, x2 ~ x1, col = "lightblue", zlab = "Eficiencia")

# Curvas de nivel del modelo de segundo orden: líneas de igual respuesta predicha
contour(modelo2, ~ x1 + x2, image = TRUE)
