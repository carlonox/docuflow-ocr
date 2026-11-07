# Procesar un corpus grande con DocuFlow

Guía para lotes de miles de documentos: qué mirar antes, cómo no pagar OCR de
más y cómo reanudar sin repetir trabajo. Está escrita con lo aprendido en un
corpus real de 35.748 PDFs (10,6 GB) del que solo el **18 %** necesitaba OCR.

## 1. Primero, mídelo

Antes de lanzar nada, separa los documentos en dos clases:

| Clase | Cómo se detecta | Coste |
|---|---|---|
| **Con capa de texto** | `page.get_text()` devuelve texto | ~10-30 ms por PDF |
| **Solo imagen (escaneo)** | la capa está vacía | OCR real |

En el corpus de referencia, menos de una quinta parte no tenía capa; el resto se
resolvía con extracción directa. Lanzar OCR a todo habría multiplicado el tiempo por cinco
sin mejorar nada.

```python
import fitz
doc = fitz.open(path)
tiene_capa = any(p.get_text().strip() for p in doc)
```

## 2. La cascada hace el resto

Para los que sí necesitan OCR, `CascadeOCR` ejecuta primero el proveedor rápido
(RapidOCR, ~0,1-2 s) y solo escala al preciso (Surya) cuando la confianza baja
del umbral. En pruebas reales, ~90 % de las páginas se quedan en el primer
escalón.

```python
from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider

cascade = CascadeOCR(
    fast_provider=RapidOCRProvider(),
    accurate_provider=None,          # añade SuryaProvider si hay GPU
    config=CascadeConfig(fast_threshold=0.90),
)
result = cascade.extract("pagina.png")   # .text, .confidence
```

## 3. Lote reanudable

`examples/batch_corpus.py` implementa el patrón completo:

```bash
# 1) corpus de prueba reproducible (sin datos de nadie)
python examples/make_sample_corpus.py --out samples --n 40

# 2) lote
python examples/batch_corpus.py --root samples --out demo.jsonl --sleep 0.3

# 3) informe
python scripts/dashboard.py demo.jsonl -o dashboard.html --titulo "samples"
```

- Escribe **una línea JSONL por documento** con el texto de cada página.
- `done.txt` permite relanzar el comando: salta lo ya procesado.
- `--sleep` deja respirar a la CPU; `--workers > 1` está bloqueado salvo
  `--allow-workers` (con RapidOCR y CPU, más procesos empeoran el tiempo y
  saturan la máquina).
- Prioridad de proceso **BELOW_NORMAL** y `OMP_NUM_THREADS=2` por defecto.

## 4. Aprender el formato repetido

La cascada paga el modelo caro página a página. El siguiente escalón es aprender
por **formato**: si un tipo de documento (factura bancaria, acta, cédula) se
resuelve siempre con confianza alta en el motor rápido, no hace falta volver a
probarlo.

`LearningEngine` ya hace la parte difícil —minar confusiones (1↔7, 0↔O),
comparar características de imagen de los fallos y proponer ajustes— pero está
pensado para campos con referencia. Para texto libre el equivalente es:

1. Agrupar páginas por firma de formato (primera línea + proporción de campos).
2. Guardar confianza media por formato.
3. Si la confianza es alta, marcar el formato como «resuelto por el rápido».
4. Las páginas dudosas alimentan `ManualFeedback` → `LearningEngine.analyze()`.

Mientras eso se implementa, el dashboard ya muestra qué proveedor resolvió cada
página y qué documentos quedaron por debajo de 0.80.

## 5. Resultados de referencia

Corpus sintético de `make_sample_corpus.py` (40 PDFs: 20 con capa, 20 escaneados):

| Métrica | Valor |
|---|---|
| Documentos | 40 |
| Con capa de texto | 20 (50 %) |
| OCR real | 20 |
| Velocidad (1 proceso, CPU) | ~0,25 PDF/s |
| Confianza media (RapidOCR) | ver dashboard |

En un corpus de decenas de miles de PDFs con ~18 % escaneados, la estimación con un
solo proceso es de varias horas; con GPU, entre una y tres.
