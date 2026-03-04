"""
Módulo: nlp_pipeline.py
Responsabilidad: Pipeline de Inteligencia de Riesgos mediante procesamiento de texto (HuggingFace).

Procesa noticias en crudo y evalúa cuantitativamente un riesgo con heurísticas 
híbridas (NLP model confidence x Severidad de Palabras Clave).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Manejo seguro para evitar roturas si hay envenenamiento de entorno
try:
    from transformers import pipeline, Pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers no detectado. El pipeline no podrá instanciarse.")
    _TRANSFORMERS_AVAILABLE = False


class RiskIntelligence:
    """
    Motor híbrido de Inteligencia de Riesgos sobre texto desestructurado.
    
    Funciona en dos pasos:
    1. Base NLP Sentiment: Uso de un modelo preentrenado (HF lightweight) para
       extraer el vector de sentimiento crudo.
    2. Multiplicadores Léxicos: Aumentos asimétricos del riesgo basados en
       vocabularios críticos orientados a cadenas de suministro.
    """

    # Keyword lists for severe multiplier penalties
    CRITICAL_KEYWORDS = [
        "strike", "war", "blockade", "sanctions", "typhoon",
        "earthquake", "flooding", "union", "embargo", "tsunami",
        "drought", "storm", "tariff", "tariffs", "military"
    ]

    # Geopolitical fuzzy matching registry (MVP)
    KNOWN_LOCATIONS = [
        "suez", "panama", "shanghai", "rotterdam", "busan", 
        "felixstowe", "valencia", "singapore", "los angeles", 
        "antwerp", "hamburg", "taiwan", "hormuz", "malacca"
    ]

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        """
        Inicializa el pipeline de Machine Learning cargando pesos en RAM.
        Utiliza DestilBert por razones de eficiencia (modelo < 300MB).
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Se requiere 'transformers' y 'torch'. Revisa requirements.txt.")

        self.model_name = model_name
        logger.info("Iniciando carga del NLP Transformer en RAM: %s", self.model_name)
        try:
            # HuggingFace Pipeline wrapper abstractions (Sentiment)
            self._analyzer: Pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name,
                truncation=True,
                max_length=512
            )
            logger.info("✔️ Modelo NLP '%s' cargado y activo.", self.model_name)
        except Exception as exc:
            logger.error("❌ Fallo crítico de MLOps cargando %s: %s", self.model_name, exc)
            raise

    def assess_risk(self, text: str) -> float:
        """
        Recibe un texto y devuelve un risk_score continuo [0.0 - 1.0].
        
        Lógica:
        - Si el NLP clasifica la oración como NEGATIVE -> parte de score = confidence * 0.5.
        - Escaneo Regex/Fuzzy para detectar multiplicadores. 
          Ej: Un texto "Earthquake at Suez" disparará ambos NLP penalty y lexical multipliers.
        """
        if not text or not text.strip():
            return 0.0
            
        text_lower = text.lower()
        
        try:
            inference = self._analyzer(text[:512])[0]
            label: str = inference["label"]    # "POSITIVE" or "NEGATIVE"
            confidence: float = inference["score"]
        except Exception as e:
            logger.warning("Fallo en inferencia NLP sobre string truncado. Retornando neutro. Detalle: %s", e)
            return 0.1
            
        # 1. Base Score calculation
        if label == "NEGATIVE":
            # El NLP está seguro de la negatividad (Max Base 0.5)
            base_risk = confidence * 0.5 
        else:
            # Positivo, el riesgo es residual/inverso a la confidencia
            base_risk = 0.1 * (1.0 - confidence)

        # 2. Multiplicador de Severidad Geopolítica/Logística
        multiplier = 1.0
        for kw in self.CRITICAL_KEYWORDS:
            if kw in text_lower:
                # Cada concepto crítico magnifica el riesgo en un 50%
                multiplier += 0.5 
                
        # 3. Aggregation step
        final_risk = base_risk * multiplier
        
        # Sigmoid bind para mantener entre [0.0 y 1.0]
        return min(max(final_risk, 0.0), 1.0)

    def extract_location(self, text: str) -> List[str]:
        """
        Fuzzy matcher simplificado como sustituto rápido de un motor NER completo,
        para extracción MVP de corredores y clústers afectados.
        """
        if not text:
            return []
            
        text_lower = text.lower()
        found_nodes = []
        for loc in self.KNOWN_LOCATIONS:
            if loc in text_lower:
                found_nodes.append(loc.title())
                
        return sorted(list(set(found_nodes)))
