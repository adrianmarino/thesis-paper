import numpy as np
from scipy.stats import shapiro, levene
from scipy.stats import ttest_ind
import data.plot as dpl


class TStudentTestSummary:
    def __init__(
        self,
        shapiro_stat1,
        shapiro_p_value1,
        shapiro_stat2,
        shapiro_p_value2,
        levene_stat,
        levene_p_value,
        t_student_stat,
        t_student_p_value,
        normality_assumption,
        variance_homogeneity_assumption,
        significant_difference,
        dist1,
        dist2,
        figsize
    ):
        self.shapiro_stat1         = shapiro_stat1
        self.shapiro_stat2         = shapiro_stat2
        self.shapiro_p_value1      = shapiro_p_value1
        self.shapiro_p_value2      = shapiro_p_value2
        self.levene_stat           = levene_stat
        self.levene_p_value        = levene_p_value
        self.stat                  = t_student_stat
        self.p_value               = t_student_p_value
        self.normality_assumption  = normality_assumption
        self.variance_homogeneity_assumption = variance_homogeneity_assumption
        self.significant_difference          = significant_difference
        self.dist1 = dist1
        self.dist2 = dist2
        self.figsize = figsize

    def __str__(self):
        dpl.describe_num_var_array(self.dist1, "Distribucion 1", figsize=self.figsize, title_fontsize=12)
        dpl.describe_num_var_array(self.dist2, "Distribucion 2", figsize=self.figsize, title_fontsize=12)

        if self.significant_difference:
            result = 'Se rechaza la hipótesis nula. Hay una diferencia estadísticamente significativa entre las medias de las distribuciones.'
        else:
            result = 'No se puede rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que existe una diferencia significativa entre las medias de las distribuciones.'

        return f"""
Prueba T-Student de diferencia de medias entre las distribuciones 1 y 2:

- Tamaño de las muestras: Dist1({len(self.dist1)}), Dist2({len(self.dist2)}).
- {'Se cumples los supuestos de normalidad y homogeneidad de las varianzas' if self.variance_homogeneity_assumption and self.normality_assumption else 'No se cumples todos los supuestos'}.
- Estadístico: {self.levene_stat}
- P-value: {self.levene_p_value}
- {result}

Supuestos:

    Prueba de normalidad para distribución 1:
    - Estadístico: {self.shapiro_stat1}
    - P-value: {self.shapiro_p_value1}

    Prueba de normalidad para distribución 2:
    - Estadístico: {self.shapiro_stat2}
    - P-value: {self.shapiro_p_value2}

    Prueba de homogeneidad de varianzas:
    - Estadístico: {self.levene_stat}
    - P-value: {self.levene_p_value}

    Resultados:
    - {'Se cumple el supuesto de normalidad de ambas distribuciones' if self.normality_assumption else 'No se cumple el supuesto de normalidad una o ambas distribuciones'}.
    - {'Se cumple el supuesto homogeneidad de las varianzas de las distribuciones' if self.variance_homogeneity_assumption else 'No se cumple el supuesto homogeneidad de las varianzas de las distribuciones'}.
"""

    def __repr__(self): return str(self)






class TStudentTest:
    def __init__(
        self,
        shapiro_significance_level    = 0.05,
        levene_significance_level     = 0.05,
        t_student_significance_level  = 0.05,
        figsize                       =(10, 4)
    ):
        self.shapiro_significance_level    = shapiro_significance_level
        self.levene_significance_level     = levene_significance_level
        self.t_student_significance_level  = t_student_significance_level
        self.figsize = figsize


    def __call__(self, dist1, dist2):
        # Normality test
        shapiro_stat1, shapiro_p_value1 = shapiro(dist1)
        shapiro_stat2, shapiro_p_value2 = shapiro(dist2)

        # Variance homogeneity test
        levene_stat, levene_p_value     = levene(dist1, dist2)

        # Distributions differences test
        t_student_stat, t_student_p_value = ttest_ind(dist1, dist2)

        return TStudentTestSummary(
            shapiro_stat1,
            shapiro_p_value1,
            shapiro_stat2,
            shapiro_p_value2,
            levene_stat,
            levene_p_value,
            t_student_stat,
            t_student_p_value,
            normality_assumption            = shapiro_p_value1  > self.shapiro_significance_level and shapiro_p_value2 > self.shapiro_significance_level,
            variance_homogeneity_assumption = levene_p_value    > self.levene_significance_level,
            significant_difference          = t_student_p_value < self.t_student_significance_level,
            dist1 = dist1,
            dist2 = dist2,
            figsize = self.figsize
        )