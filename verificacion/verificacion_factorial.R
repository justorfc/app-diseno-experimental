# Verificación en R del diseño factorial (carga x pretratamiento)
# Compara estos resultados con los que muestra la página "Factoriales" de la app.
# Uso: Rscript verificacion_factorial.R   (ejecutar desde la carpeta verificacion/)

# Lee el archivo CSV (separado por comas) y lo guarda en la tabla "datos"
datos <- read.csv("../data/factorial_mi_variante.csv")

# Convierte los factores en categorías: R los trata como grupos, no como números
datos$carga <- factor(datos$carga)
datos$pretratamiento <- factor(datos$pretratamiento)

# Ajusta el modelo factorial completo: efecto de la carga, del pretratamiento y de su interacción
# (el símbolo * equivale a carga + pretratamiento + carga:pretratamiento)
modelo <- aov(humedad ~ factor(carga) * pretratamiento, data = datos)

# Imprime la tabla ANOVA: gl, suma de cuadrados, cuadrado medio, F y valor p de cada efecto
# (se mira primero la interacción: si p < 0.05, los efectos principales no se interpretan solos)
print(summary(modelo))

# Calcula la humedad media de cada combinación carga x pretratamiento (medias de celda)
medias_celda <- tapply(datos$humedad, list(datos$carga, datos$pretratamiento), mean)
cat("\nMedias de celda (filas = carga, columnas = pretratamiento):\n")
print(medias_celda)

# Gráfica de interacción: una línea por pretratamiento, con la carga en el eje x
# Líneas paralelas = sin interacción; líneas que se cruzan o se abren = hay interacción
# (con Rscript el gráfico se guarda en el archivo Rplots.pdf)
interaction.plot(
  x.factor = datos$carga,            # factor del eje x
  trace.factor = datos$pretratamiento, # una línea por cada nivel de este factor
  response = datos$humedad,          # variable que se grafica (media por celda)
  xlab = "Carga", ylab = "Humedad media (%)", trace.label = "Pretratamiento"
)

# Prueba de Tukey sobre la interacción: compara todos los pares de combinaciones (celdas)
# Si p adj < 0.05 (o el intervalo lwr-upr no incluye 0), esas dos celdas difieren
# Usa el CM Error del modelo, igual que la pestaña "Comparaciones" de la app
cat("\nTukey sobre la interacción carga:pretratamiento:\n")
print(TukeyHSD(modelo, which = "factor(carga):pretratamiento"))
