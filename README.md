[![Ver sitio](https://img.shields.io/badge/Ver%20sitio-GitHub%20Pages-7aa2ff?style=for-the-badge&logo=github)](https://tupxz.github.io/ciclos_economicos/site/index.html)

# Ciclos Económicos — Macroeconomía 3 (EAFIT)

**Curso:** Macroeconomía 3: Ciclos Económicos (EAFIT)  
**Profesor:** Álvaro Arturo Hurtado  
**Autor:** Santiago Tupaz  
**Monitor:** Diego Alejandro Quintero  

## Sitio del proyecto (GitHub Pages)
Página con entregables y outputs del taller:

- **https://tupxz.github.io/ciclos_economicos/site/index.html**

---

## Qué contiene este repositorio

Repositorio para desarrollar actividades, talleres y ejercicios cuantitativos del curso.  
El foco principal (hasta ahora) es **Taller 1**, implementado de forma reproducible:

- **Pipeline de datos** (FRED → dataset trimestral limpio)
- **Exploración descriptiva + visualizaciones**
- **Filtro Hodrick–Prescott (HP)**
  - Implementación manual (álgebra lineal)
  - Implementación con `statsmodels`
- **Hechos estilizados** (volatilidad, correlaciones contemporáneas y móviles)
- **Análisis Insumo–Producto (I–O)**
  - Coeficientes técnicos, inversa de Leontief
  - Encadenamientos backward/forward (directos, totales e “indirectos puros”)
  - Mapa de encadenamientos (scatter)

---

## Estructura del proyecto

```text
ciclos_economicos/
  data/
    raw/           # insumos crudos (FRED xlsx, matriz IO, etc.)
    processed/     # dataset limpio (dataset_taller.*)
  notebooks/
    Taller_1.ipynb # notebook principal (organizado por fases)
  src/
    data_prep.py   # ingesta + estandarización del dataset
    hp_filter.py   # HP manual (implementación propia)
    cycle_stats.py # HP statsmodels + stats (volatilidad/correlaciones)
    io_tools.py    # herramientas IO (A, L, encadenamientos)
  outputs/
    figures/       # figuras exportadas (01_...png)
    tables/        # tablas exportadas (xlsx/csv)
    exports/       # entregables finales (Excel consolidado)
  site/            # GitHub Pages (reportes, outputs copiados y landing)
  requirements.txt
  README.md
```

## Entregables (Taller 1)
Los principales outputs del taller se generan automáticamente al ejecutar el notebook:

- **Excel final consolidado:** `outputs/exports/entrega_taller_1.xlsx`
- **Tablas auxiliares:** `outputs/tables/`
- **Figuras:** `outputs/figures/`
- **Reportes HTML/PDF:** publicados en `site/` (GitHub Pages)

---

## Cómo reproducir

### 1) Crear ambiente e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```
### 2) Generar dataset

```bash
python -m src.data_prep data/raw data/processed
```
### 3) Ejecutar el notebook
Abrir y ejecutar:
	•	notebooks/Taller_1.ipynb (Run All)

### Sobre mí

Santiago Tupaz
- Estudiante de Economía – EAFIT
- Estudiante de Estadística – Universidad Nacional de Colombia

linkedin: https://www.linkedin.com/in/santiago-tupaz-ram%C3%ADrez-2b87082b3/


