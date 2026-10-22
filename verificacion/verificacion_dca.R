# Verificación en R del DCA (Diseño Completamente al Azar)
# Compara estos resultados con los que muestra la página "DCA" de la app.
# Uso: Rscript verificacion_dca.R   (ejecutar desde la carpeta verificacion/)

# Lee el archivo CSV (separado por comas) y lo guarda en la tabla "datos"
datos <- read.csv("../data/dca_mi_variante.csv")

# Convierte secador en factor: R lo trata como grupos (tratamientos), no como número
datos$secador <- factor(datos$secador)

# Ajusta el modelo ANOVA de un factor: ¿la humedad media cambia según el secador?
modelo <- aov(humedad ~ factor(secador), data = datos)

# Imprime la tabla ANOVA: suma de cuadrados, F y valor p
# (un valor p < 0.05 indica que al menos un secador difiere de los demás)
print(summary(modelo))

# Calcula la humedad media de cada secador
medias <- tapply(datos$humedad, datos$secador, mean)
cat("\nMedias por secador:\n")
print(medias)

# Cuadrado medio del error (CME): variación no explicada por los secadores
cme <- deviance(modelo) / df.residual(modelo)

# Coeficiente de variación (%) = raíz del CME / media general * 100
# Mide qué tan precisos son los datos; menor CV significa mejor control del ensayo
cv <- sqrt(cme) / mean(datos$humedad) * 100
cat("\nCoeficiente de variación (%):", round(cv, 2), "\n")

# Prueba de Tukey: compara todos los pares de secadores
# Si el intervalo (lwr, upr) no incluye 0 o p adj < 0.05, esos dos secadores difieren
cat("\nComparaciones múltiples de Tukey:\n")
print(TukeyHSD(modelo))
