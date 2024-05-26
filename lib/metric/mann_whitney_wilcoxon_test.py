from scipy.stats import mannwhitneyu

class MannWhitneyWilcoxonTestResult:
    def __init__(
        self,
        u_stat,
        p_value,
        significance_level,
        dist1,
        dist2,
    ):
        self.u_stat = u_stat
        self.p_value = p_value
        self.significance_level = significance_level
        self.dist1 = dist1
        self.dist2 = dist2

    def __str__(self):
        if self.p_value < self.significance_level:
            output = 'Se rechaza la hipótesis nula. Hay una diferencia estadísticamente significativa entre las medianas de los grupos.'
            p_value_desc = f'(<{self.significance_level})'
        else:
            output = 'No se puede rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que existe una diferencia estadísticamente significativa entre las medianas de los grupos.'
            p_value_desc = f'(>={self.significance_level})'

        return f"""
Prueba de Mann Whiney - Wilcoxon para la comparación de medianas:
==========================================================

- Tamaño de las muestras: Dist1({len(self.dist1)}), Dist2({len(self.dist2)}).
- Estadístico: {self.u_stat}
- P-value: {self.p_value} {p_value_desc}
- {output}
"""

    def __repr__(self): return str(self)


class MannWhitneyWilcoxonTest:
    def __init__(self, significance_level = 0.05):
        self.significance_level = significance_level

    def __call__(self, dist1, dist2):
        u_stat, p_value = mannwhitneyu(dist1, dist2)
        return MannWhitneyWilcoxonTestResult(
            u_stat,
            p_value,
            self.significance_level,
            dist1,
            dist2
        )