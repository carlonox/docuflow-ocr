"""docuflow — self-learning document processing framework.

Extract text from degraded documents with OCR engines that learn from their
own successes and failures. Built around a cascade pipeline (fast -> accurate
-> LLM correction) and a learning subsystem that tunes extraction parameters
automatically per document field.
"""

__version__ = "0.1.0"
__author__ = "Carlos Javier Cuervo Baracaldo"
__license__ = "MIT"
