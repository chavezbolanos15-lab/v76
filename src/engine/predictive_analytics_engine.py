#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - Predictive Analytics Engine
Motor de Análise Preditiva e Insights Profundos Ultra-Avançado
"""
import os
import logging
import json
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings("ignore")
# Imports condicionais para análise avançada
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
try:
    import gensim
    from gensim import corpora, models
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
from services.auto_save_manager import salvar_etapa, salvar_erro
logger = logging.getLogger(__name__)

class PredictiveAnalyticsEngine:
    """Motor de Análise Preditiva e Insights Profundos Ultra-Avançado"""

    def __init__(self):
        """Inicializa o motor de análise preditiva"""
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.tfidf_vectorizer = None
        self.topic_model = None
        # Configurações de análise
        self.config = {
            "min_text_length": 100,
            "max_features_tfidf": 1000,
            "n_topics_lda": 10,
            "n_clusters_kmeans": 5,
            "confidence_threshold": 0.7,
            "prediction_horizon_days": 90,
            "min_data_points_prediction": 5
        }
        self._initialize_models()
        logger.info("🔮 Predictive Analytics Engine Ultra-Avançado inicializado")

    def _initialize_models(self):
        """Inicializa modelos de ML e NLP"""
        # Carrega modelo SpaCy para português
        if HAS_SPACY:
            try:
                self.nlp_model = spacy.load("pt_core_news_sm")
                logger.info("✅ Modelo SpaCy português carregado")
            except OSError:
                try:
                    self.nlp_model = spacy.load("pt_core_news_lg")
                    logger.info("✅ Modelo SpaCy português (large) carregado")
                except OSError:
                    logger.warning("⚠️ Modelo SpaCy não encontrado. Execute: python -m spacy download pt_core_news_sm")
                    self.nlp_model = None
        # Inicializa analisador de sentimento
        if HAS_VADER:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("✅ Analisador de sentimento VADER carregado")
        # Inicializa TF-IDF
        if HAS_SKLEARN:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config["max_features_tfidf"],
                stop_words=self._get_portuguese_stopwords(),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            logger.info("✅ TF-IDF Vectorizer configurado")

    def _get_portuguese_stopwords(self) -> List[str]:
        """Retorna lista de stopwords em português"""
        return [
            "a", "o", "e", "é", "de", "do", "da", "em", "um", "uma", "para", "com", "não", "que", "se", "na", "por",
            "mais", "as", "os", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
            "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
            "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse",
            "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm",
            "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas",
            "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas"
        ]

    async def analyze_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Analisa todos os dados disponíveis de uma sessão para gerar insights preditivos ultra-avançados
        Args:
            session_id: ID da sessão
        Returns:
            Dict com insights preditivos completos
        """
        logger.info(f"🔮 INICIANDO ANÁLISE PREDITIVA ULTRA-AVANÇADA para sessão: {session_id}")
        session_dir = Path(f"analyses_data/{session_id}")
        if not session_dir.exists():
            logger.error(f"❌ Diretório da sessão não encontrado: {session_dir}")
            return {"success": False, "error": "Diretório da sessão não encontrado"}

        # Estrutura de insights ultra-completa
        insights = {
            "session_id": session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "success": True,
            "methodology": "ARQV30_PREDICTIVE_ULTRA_v3.0",
            # Análises principais
            "textual_insights": {},
            "temporal_trends": {},
            "visual_insights": {},
            "network_analysis": {},
            "sentiment_dynamics": {},
            "topic_evolution": {},
            "engagement_patterns": {},
            # Previsões e cenários
            "predictions": {},
            "scenarios": {},
            "risk_assessment": {},
            "opportunity_mapping": {},
            # Métricas de confiança
            "confidence_metrics": {},
            "data_quality_assessment": {},
            # Recomendações estratégicas
            "strategic_recommendations": {},
            "action_priorities": {}
        }

        try:
            # FASE 1: Análise Textual Ultra-Profunda
            logger.info("🧠 FASE 1: Análise textual ultra-profunda...")
            insights["textual_insights"] = await self._perform_ultra_textual_analysis(session_dir)

            # FASE 2: Análise de Tendências Temporais
            logger.info("📈 FASE 2: Análise de tendências temporais...")
            insights["temporal_trends"] = await self._perform_temporal_analysis(session_dir)

            # FASE 3: Análise Visual Avançada (OCR + Computer Vision)
            logger.info("👁️ FASE 3: Análise visual avançada...")
            insights["visual_insights"] = await self._perform_advanced_visual_analysis(session_dir)

            # FASE 4: Análise de Rede e Conectividade
            logger.info("🕸️ FASE 4: Análise de rede e conectividade...")
            insights["network_analysis"] = await self._perform_network_analysis(session_dir)

            # FASE 5: Dinâmica de Sentimentos
            logger.info("💭 FASE 5: Análise de dinâmica de sentimentos...")
            insights["sentiment_dynamics"] = await self._analyze_sentiment_dynamics(session_dir)

            # FASE 6: Evolução de Tópicos
            logger.info("🔄 FASE 6: Análise de evolução de tópicos...")
            insights["topic_evolution"] = await self._analyze_topic_evolution(session_dir)

            # FASE 7: Padrões de Engajamento
            logger.info("📊 FASE 7: Análise de padrões de engajamento...")
            insights["engagement_patterns"] = await self._analyze_engagement_patterns(session_dir)

            # FASE 8: Geração de Previsões Ultra-Avançadas
            logger.info("🔮 FASE 8: Geração de previsões ultra-avançadas...")
            insights["predictions"] = await self._generate_ultra_predictions(insights)

            # FASE 9: Modelagem de Cenários Complexos
            logger.info("🗺️ FASE 9: Modelagem de cenários complexos...")
            insights["scenarios"] = await self._model_complex_scenarios(insights)

            # FASE 10: Avaliação de Riscos e Oportunidades
            logger.info("⚖️ FASE 10: Avaliação de riscos e oportunidades...")
            insights["risk_assessment"] = await self._assess_risks_and_opportunities(insights)

            # FASE 11: Mapeamento de Oportunidades
            logger.info("🎯 FASE 11: Mapeamento estratégico de oportunidades...")
            insights["opportunity_mapping"] = await self._map_strategic_opportunities(insights)

            # FASE 12: Métricas de Confiança
            logger.info("📏 FASE 12: Cálculo de métricas de confiança...")
            insights["confidence_metrics"] = await self._calculate_confidence_metrics(insights)

            # FASE 13: Avaliação de Qualidade dos Dados
            logger.info("🔍 FASE 13: Avaliação de qualidade dos dados...")
            insights["data_quality_assessment"] = await self._assess_data_quality(session_dir)

            # FASE 14: Recomendações Estratégicas
            logger.info("💡 FASE 14: Geração de recomendações estratégicas...")
            insights["strategic_recommendations"] = await self._generate_strategic_recommendations(insights)

            # FASE 15: Priorização de Ações
            logger.info("🎯 FASE 15: Priorização de ações...")
            insights["action_priorities"] = await self._prioritize_actions(insights)

            # Salva insights preditivos
            insights_path = session_dir / "insights_preditivos.json"
            with open(insights_path, "w", encoding="utf-8") as f:
                json.dump(insights, f, ensure_ascii=False, indent=2)

            # Salva também como etapa
            salvar_etapa("insights_preditivos_completos", insights, categoria="analise_preditiva", session_id=session_id)

            logger.info(f"✅ ANÁLISE PREDITIVA ULTRA-AVANÇADA CONCLUÍDA: {insights_path}")
            return insights

        except Exception as e:
            logger.error(f"❌ Erro crítico na análise preditiva: {e}")
            salvar_erro("predictive_analytics_critical", e, contexto={"session_id": session_id})
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_content_chunk(self, text_content: str) -> Dict[str, Any]:
        """Analisa um chunk de conteúdo textual para extrair insights."""
        logger.info("🧠 Analisando chunk de conteúdo textual...")
        results = {
            "text_length": len(text_content),
            "sentiment_analysis": {},
            "key_phrases": [],
            "summary": ""
        }

        if not text_content or len(text_content) < self.config["min_text_length"]:
            results["summary"] = "Conteúdo muito curto para análise detalhada."
            return results

        # Análise de sentimento
        if HAS_VADER and self.sentiment_analyzer:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text_content)
            results["sentiment_analysis"] = sentiment_scores

        # Extração de palavras-chave/frases-chave (usando TF-IDF ou SpaCy)
        if HAS_SKLEARN and self.tfidf_vectorizer:
            try:
                # Fit e transform em um único documento pode ser problemático para TF-IDF
                # Idealmente, TF-IDF é treinado em um corpus maior.
                # Para um único chunk, podemos extrair as palavras mais frequentes após remover stopwords.
                words = [word for word in re.findall(r'\b\w+\b', text_content.lower()) if word not in self._get_portuguese_stopwords()]
                word_counts = Counter(words)
                results["key_phrases"] = [word for word, count in word_counts.most_common(10)]
            except Exception as e:
                logger.warning(f"⚠️ Erro na extração de key_phrases para chunk: {e}")
        elif HAS_SPACY and self.nlp_model:
            try:
                doc = self.nlp_model(text_content[:100000]) # Limita para performance
                word_freq = Counter([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                results["key_phrases"] = [word for word, count in word_freq.most_common(10)]
            except Exception as e:
                logger.warning(f"⚠️ Erro na extração de key_phrases com SpaCy para chunk: {e}")

        # Geração de resumo (simplificado)
        results["summary"] = text_content[:200] + "..." if len(text_content) > 200 else text_content

        logger.info("✅ Análise de chunk de conteúdo concluída.")
        return results

    async def analyze_data_quality(self, massive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia a qualidade dos dados coletados."""
        logger.info("🔍 Avaliando qualidade dos dados...")
        quality_metrics = {
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "timeliness_score": 0.0,
            "relevance_score": 0.0,
            "overall_quality_score": 0.0,
            "issues_detected": []
        }

        total_sources = massive_data.get("statistics", {}).get("total_sources", 0)
        if total_sources == 0:
            quality_metrics["issues_detected"].append("Nenhuma fonte de dados coletada.")
            return quality_metrics

        # Completude: % de campos essenciais preenchidos
        completed_fields = 0
        total_fields = 0
        for content_item in massive_data.get("extracted_content", []):
            total_fields += 1 # Cada item é uma "fonte"
            if content_item.get("content") or content_item.get("text") or content_item.get("title"):
                completed_fields += 1
        quality_metrics["completeness_score"] = (completed_fields / total_sources) * 100 if total_sources > 0 else 0

        # Consistência: verificar duplicatas ou formatos inconsistentes (simplificado)
        all_contents = [json.dumps(item, sort_keys=True) for item in massive_data.get("extracted_content", [])]
        unique_contents = len(set(all_contents))
        quality_metrics["consistency_score"] = (unique_contents / total_sources) * 100 if total_sources > 0 else 0
        if unique_contents < total_sources:
            quality_metrics["issues_detected"].append(f"Detectadas {total_sources - unique_contents} duplicatas de conteúdo.")

        # Temporalidade: quão recentes são os dados (simplificado)
        collection_timestamp_str = massive_data.get("collection_started")
        if collection_timestamp_str:
            collection_time = datetime.fromisoformat(collection_timestamp_str)
            age_in_days = (datetime.now() - collection_time).days
            # Exemplo: dados com menos de 7 dias são 100%, mais antigos diminuem
            quality_metrics["timeliness_score"] = max(0, 100 - (age_in_days * 5)) # Reduz 5% por dia
        else:
            quality_metrics["issues_detected"].append("Timestamp de coleta não disponível para avaliar temporalidade.")

        # Relevância: (difícil de automatizar sem feedback humano, aqui é um placeholder)
        # Poderia ser baseado em quão bem as palavras-chave da query aparecem no conteúdo.
        query = massive_data.get("query", "")
        if query and all_contents:
            relevance_score_sum = 0
            query_words = set(query.lower().split())
            for content_item in massive_data.get("extracted_content", []):
                text = str(content_item.get("content", "")) + str(content_item.get("snippet", ""))
                if text:
                    text_words = set(text.lower().split())
                    common_words = len(query_words.intersection(text_words))
                    relevance_score_sum += (common_words / len(query_words)) if len(query_words) > 0 else 0
            quality_metrics["relevance_score"] = (relevance_score_sum / total_sources) * 100 if total_sources > 0 else 0
        else:
            quality_metrics["issues_detected"].append("Query ou conteúdo insuficiente para avaliar relevância.")

        # Cálculo do score geral
        quality_metrics["overall_quality_score"] = np.mean([
            quality_metrics["completeness_score"],
            quality_metrics["consistency_score"],
            quality_metrics["timeliness_score"],
            quality_metrics["relevance_score"]
        ])

        logger.info(f"✅ Avaliação de qualidade dos dados concluída. Score: {quality_metrics['overall_quality_score']:.2f}")
        return quality_metrics

    async def refine_search_queries(self, original_query: str, search_results: Dict[str, Any]) -> List[str]:
        """Refina queries de busca com base nos resultados iniciais."""
        logger.info(f"🔄 Refinando queries de busca para: {original_query}")
        refined_queries = [original_query] # Sempre inclui a original

        all_snippets = []
        for provider_results in search_results.get("all_results", []):
            for result in provider_results.get("results", []):
                if result.get("snippet"):
                    all_snippets.append(result["snippet"])

        if not all_snippets:
            logger.warning("⚠️ Nenhum snippet encontrado para refinar queries.")
            return refined_queries

        # Concatena snippets para análise de tópicos/palavras-chave
        combined_text = " ".join(all_snippets)

        # Extrai palavras-chave e entidades usando SpaCy (se disponível)
        if HAS_SPACY and self.nlp_model:
            try:
                doc = self.nlp_model(combined_text[:100000]) # Limita para performance
                keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_space and token.is_alpha]
                entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "LOC", "PRODUCT"]]
                # Combina e seleciona as mais relevantes
                all_terms = Counter(keywords + entities)
                top_terms = [term for term, count in all_terms.most_common(5)]
                for term in top_terms:
                    if term.lower() not in original_query.lower():
                        refined_queries.append(f"{original_query} {term}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao usar SpaCy para refinar queries: {e}")

        # Fallback simples: extrair bigramas frequentes
        if not HAS_SPACY or not self.nlp_model:
            words = [word for word in re.findall(r'\b\w+\b', combined_text.lower()) if word not in self._get_portuguese_stopwords()]
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            top_bigrams = [bigram for bigram, count in bigram_counts.most_common(3)]
            for bigram in top_bigrams:
                if bigram.lower() not in original_query.lower():
                    refined_queries.append(f"{original_query} {bigram}")

        # Remove duplicatas e limita o número de queries
        refined_queries = list(dict.fromkeys(refined_queries))[:5]

        logger.info(f"✅ Queries refinadas: {refined_queries}")
        return refined_queries

    async def _perform_ultra_textual_analysis(self, session_dir: Path) -> Dict[str, Any]:
        """Realiza análise textual ultra-profunda em todo o conteúdo coletado."""
        logger.info("📝 Realizando análise textual ultra-profunda...")
        textual_insights = {
            "total_words": 0,
            "unique_words": 0,
            "readability_score": 0.0,
            "sentiment_distribution": {},
            "top_keywords": [],
            "top_entities": {},
            "topic_modeling": {},
            "content_summaries": {}
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return textual_insights

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        all_text_content = []
        for item in massive_data.get("extracted_content", []):
            text = str(item.get("content", "")) + str(item.get("snippet", "")) + str(item.get("title", ""))
            if text:
                all_text_content.append(text)

        if not all_text_content:
            logger.warning("⚠️ Nenhum conteúdo textual para análise.")
            return textual_insights

        combined_text = " ".join(all_text_content)
        words = re.findall(r'\b\w+\b', combined_text.lower())
        textual_insights["total_words"] = len(words)
        textual_insights["unique_words"] = len(set(words))

        # Readability (simplificado)
        textual_insights["readability_score"] = len(words) / len(set(re.findall(r'\b\w+\b', combined_text))) if len(set(re.findall(r'\b\w+\b', combined_text))) > 0 else 0

        # Análise de Sentimento Distribuída
        if HAS_VADER and self.sentiment_analyzer:
            sentiment_scores = [self.sentiment_analyzer.polarity_scores(text) for text in all_text_content]
            pos_count = sum(1 for s in sentiment_scores if s["compound"] >= 0.05)
            neg_count = sum(1 for s in sentiment_scores if s["compound"] <= -0.05)
            neu_count = len(sentiment_scores) - pos_count - neg_count
            total_sentiments = len(sentiment_scores)
            textual_insights["sentiment_distribution"] = {
                "positive": (pos_count / total_sentiments) * 100 if total_sentiments > 0 else 0,
                "negative": (neg_count / total_sentiments) * 100 if total_sentiments > 0 else 0,
                "neutral": (neu_count / total_sentiments) * 100 if total_sentiments > 0 else 0
            }

        # Tópicos e Entidades (se SpaCy e Gensim disponíveis)
        if HAS_SPACY and self.nlp_model and HAS_GENSIM:
            try:
                processed_docs = []
                all_entities = defaultdict(int)
                for text in all_text_content:
                    doc = self.nlp_model(text[:100000]) # Limita para performance
                    processed_docs.append([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                    for ent in doc.ents:
                        all_entities[ent.label_] += 1
                textual_insights["top_entities"] = dict(all_entities)

                # Topic Modeling (LDA)
                if processed_docs:
                    dictionary = corpora.Dictionary(processed_docs)
                    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                    lda_model = models.LdaModel(corpus, num_topics=self.config["n_topics_lda"], id2word=dictionary, passes=15)
                    topics = []
                    for idx, topic in lda_model.print_topics(-1):
                        topics.append({"id": idx, "keywords": topic})
                    textual_insights["topic_modeling"] = {"topics": topics}
            except Exception as e:
                logger.warning(f"⚠️ Erro na modelagem de tópicos/entidades: {e}")

        # Top Keywords (TF-IDF ou simples contagem)
        if HAS_SKLEARN and self.tfidf_vectorizer and all_text_content:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_text_content)
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                # Soma os scores TF-IDF para cada palavra
                sums = tfidf_matrix.sum(axis=0)
                data = []
                for col, term in enumerate(feature_names):
                    data.append((term, sums[0, col]))
                sorted_keywords = sorted(data, key=lambda x: x[1], reverse=True)
                textual_insights["top_keywords"] = [kw[0] for kw in sorted_keywords[:20]]
            except Exception as e:
                logger.warning(f"⚠️ Erro na extração de top_keywords com TF-IDF: {e}")
        elif all_text_content:
            words_no_stopwords = [word for word in words if word not in self._get_portuguese_stopwords()]
            word_counts = Counter(words_no_stopwords)
            textual_insights["top_keywords"] = [word for word, count in word_counts.most_common(20)]

        logger.info("✅ Análise textual ultra-profunda concluída.")
        return textual_insights

    async def _perform_temporal_analysis(self, session_dir: Path) -> Dict[str, Any]:
        """Realiza análise de tendências temporais."""
        logger.info("⏰ Realizando análise de tendências temporais...")
        temporal_trends = {
            "content_over_time": [],
            "sentiment_over_time": [],
            "topic_frequency_over_time": {},
            "prediction_models": {},
            "future_projections": {}
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return temporal_trends

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        # Coleta dados com timestamp
        dated_content = []
        for item in massive_data.get("extracted_content", []):
            timestamp_str = item.get("published_at") or item.get("timestamp") or massive_data.get("collection_started")
            if timestamp_str:
                try:
                    # Tenta parsear vários formatos de data
                    dt = None
                    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(timestamp_str, fmt)
                            break
                        except ValueError:
                            continue
                    if dt:
                        dated_content.append({
                            "date": dt.date(),
                            "text": str(item.get("content", "")) + str(item.get("snippet", "")) + str(item.get("title", "")),
                            "source": item.get("source", "unknown")
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao parsear data {timestamp_str}: {e}")

        if not dated_content:
            logger.warning("⚠️ Nenhum conteúdo com data para análise temporal.")
            return temporal_trends

        df = pd.DataFrame(dated_content)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Frequência de conteúdo ao longo do tempo
        content_counts = df.groupby("date").size().reset_index(name="count")
        temporal_trends["content_over_time"] = content_counts.to_dict(orient="records")

        # Sentimento ao longo do tempo
        if HAS_VADER and self.sentiment_analyzer:
            df["sentiment_compound"] = df["text"].apply(lambda x: self.sentiment_analyzer.polarity_scores(x)["compound"])
            sentiment_over_time = df.groupby("date")["sentiment_compound"].mean().reset_index(name="average_sentiment")
            temporal_trends["sentiment_over_time"] = sentiment_over_time.to_dict(orient="records")

        # Frequência de tópicos ao longo do tempo (simplificado)
        if HAS_SPACY and self.nlp_model:
            topic_freq_data = defaultdict(lambda: defaultdict(int))
            for _, row in df.iterrows():
                doc = self.nlp_model(row["text"][:100000])
                keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
                for keyword in Counter(keywords).most_common(5):
                    topic_freq_data[str(row["date"])][keyword[0]] += keyword[1]
            temporal_trends["topic_frequency_over_time"] = topic_freq_data

        # Modelagem Preditiva com Prophet (se disponível e dados suficientes)
        if HAS_PROPHET and len(content_counts) >= self.config["min_data_points_prediction"]:
            try:
                # Previsão de volume de conteúdo
                df_prophet_content = content_counts.rename(columns={"date": "ds", "count": "y"})
                model_content = Prophet(daily_seasonality=True)
                model_content.fit(df_prophet_content)
                future_content = model_content.make_future_dataframe(periods=self.config["prediction_horizon_days"])
                forecast_content = model_content.predict(future_content)
                temporal_trends["prediction_models"]["content_volume"] = forecast_content[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
                temporal_trends["future_projections"]["content_volume"] = {
                    "trend": "increasing" if forecast_content["yhat"].iloc[-1] > forecast_content["yhat"].iloc[0] else "decreasing",
                    "value_in_90_days": forecast_content["yhat"].iloc[-1]
                }

                # Previsão de sentimento (se dados suficientes)
                if HAS_VADER and len(sentiment_over_time) >= self.config["min_data_points_prediction"]:
                    df_prophet_sentiment = sentiment_over_time.rename(columns={"date": "ds", "average_sentiment": "y"})
                    model_sentiment = Prophet(daily_seasonality=True)
                    model_sentiment.fit(df_prophet_sentiment)
                    future_sentiment = model_sentiment.make_future_dataframe(periods=self.config["prediction_horizon_days"])
                    forecast_sentiment = model_sentiment.predict(future_sentiment)
                    temporal_trends["prediction_models"]["average_sentiment"] = forecast_sentiment[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
                    temporal_trends["future_projections"]["average_sentiment"] = {
                        "trend": "positive" if forecast_sentiment["yhat"].iloc[-1] > 0 else "negative",
                        "value_in_90_days": forecast_sentiment["yhat"].iloc[-1]
                    }
            except Exception as e:
                logger.warning(f"⚠️ Erro na modelagem preditiva com Prophet: {e}")
        else:
            logger.info("Prophet não disponível ou dados insuficientes para previsão temporal.")

        logger.info("✅ Análise de tendências temporais concluída.")
        return temporal_trends

    async def _perform_advanced_visual_analysis(self, session_dir: Path) -> Dict[str, Any]:
        """Realiza análise visual avançada em screenshots (OCR + Computer Vision)."""
        logger.info("🖼️ Realizando análise visual avançada...")
        visual_insights = {
            "total_screenshots_analyzed": 0,
            "text_extracted_from_images": {},
            "object_detection_summary": {},
            "dominant_colors": {},
            "visual_trends": []
        }

        screenshots_dir = session_dir / "screenshots"
        if not screenshots_dir.exists():
            logger.warning(f"⚠️ Diretório de screenshots não encontrado: {screenshots_dir}")
            return visual_insights

        screenshot_files = list(screenshots_dir.glob("*.png")) + list(screenshots_dir.glob("*.jpg"))
        visual_insights["total_screenshots_analyzed"] = len(screenshot_files)

        if not screenshot_files:
            logger.warning("⚠️ Nenhuma imagem para análise visual.")
            return visual_insights

        for img_path in screenshot_files:
            try:
                # OCR (se disponível)
                if HAS_OCR:
                    try:
                        # CORRIGIDO: Linha 695
                        text = pytesseract.image_to_string(Image.open(img_path), lang='por')
                        visual_insights["text_extracted_from_images"][img_path.name] = text[:500] + "..." if len(text) > 500 else text
                    except Exception as e:
                        logger.warning(f"⚠️ Erro OCR em {img_path.name}: {e}")

                # Análise de cores dominantes (simplificado)
                if HAS_OPENCV:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            pixels = img.reshape(-1, 3)
                            # K-Means para cores dominantes (simplificado)
                            if HAS_SKLEARN:
                                kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(pixels)
                                colors = [tuple(map(int, c)) for c in kmeans.cluster_centers_]
                                visual_insights["dominant_colors"][img_path.name] = colors
                    except Exception as e:
                        logger.warning(f"⚠️ Erro na análise de cores em {img_path.name}: {e}")
            except Exception as e:
                logger.error(f"❌ Erro ao processar imagem {img_path.name}: {e}")

        logger.info("✅ Análise visual avançada concluída.")
        return visual_insights

    async def _perform_network_analysis(self, session_dir: Path) -> Dict[str, Any]:
        """Realiza análise de rede e conectividade (ex: links entre fontes)."""
        logger.info("🔗 Realizando análise de rede e conectividade...")
        network_analysis = {
            "total_nodes": 0,
            "total_edges": 0,
            "top_connected_nodes": [],
            "community_detection": {},
            "influencer_detection": []
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return network_analysis

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        if not HAS_NETWORKX:
            logger.warning("⚠️ NetworkX não disponível para análise de rede.")
            return network_analysis

        G = nx.DiGraph() # Grafo direcionado para links

        # Adiciona nós e arestas com base em URLs e menções
        urls = set()
        for item in massive_data.get("extracted_content", []):
            url = item.get("url")
            if url:
                G.add_node(url, type="url", title=item.get("title", ""))
                urls.add(url)
                # Tenta encontrar outros links no conteúdo
                content = str(item.get("content", "")) + str(item.get("snippet", ""))
                found_urls = re.findall(r'https?://[\w\d\./\-]+', content)
                for found_url in found_urls:
                    if found_url != url: # Evita auto-loops
                        G.add_edge(url, found_url, type="mentions")
                        urls.add(found_url)

        network_analysis["total_nodes"] = G.number_of_nodes()
        network_analysis["total_edges"] = G.number_of_edges()

        if G.number_of_nodes() > 0:
            # Centralidade de Grau (Top Connected Nodes)
            degree_centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
            network_analysis["top_connected_nodes"] = sorted_nodes[:10]

            # Detecção de Comunidades (simplificado, para grafos pequenos)
            if G.number_of_nodes() < 100 and HAS_NETWORKX:
                try:
                    communities = list(nx.community.label_propagation_communities(G.to_undirected()))
                    network_analysis["community_detection"] = {f"community_{i}": list(c) for i, c in enumerate(communities)}
                except Exception as e:
                    logger.warning(f"⚠️ Erro na detecção de comunidades: {e}")

            # Detecção de Influenciadores (PageRank)
            try:
                pagerank = nx.pagerank(G)
                sorted_influencers = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
                network_analysis["influencer_detection"] = sorted_influencers[:10]
            except Exception as e:
                logger.warning(f"⚠️ Erro no cálculo de PageRank: {e}")

        logger.info("✅ Análise de rede e conectividade concluída.")
        return network_analysis

    async def _analyze_sentiment_dynamics(self, session_dir: Path) -> Dict[str, Any]:
        """Analisa a dinâmica de sentimentos ao longo do tempo e por tópico."""
        logger.info("📈 Analisando dinâmica de sentimentos...")
        sentiment_dynamics = {
            "overall_sentiment_trend": {},
            "sentiment_by_source": {},
            "sentiment_by_topic": {},
            "sentiment_shifts": []
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return sentiment_dynamics

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        if not HAS_VADER or not self.sentiment_analyzer:
            logger.warning("⚠️ VADER Sentiment Analyzer não disponível.")
            return sentiment_dynamics

        dated_content = []
        for item in massive_data.get("extracted_content", []):
            timestamp_str = item.get("published_at") or item.get("timestamp") or massive_data.get("collection_started")
            text = str(item.get("content", "")) + str(item.get("snippet", "")) + str(item.get("title", ""))
            source = item.get("source", "unknown")
            if timestamp_str and text:
                try:
                    dt = None
                    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(timestamp_str, fmt)
                            break
                        except ValueError:
                            continue
                    if dt:
                        sentiment_score = self.sentiment_analyzer.polarity_scores(text)["compound"]
                        dated_content.append({
                            "date": dt.date(),
                            "text": text,
                            "sentiment": sentiment_score,
                            "source": source
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao parsear data ou sentimento: {e}")

        if not dated_content:
            logger.warning("⚠️ Nenhum conteúdo com data e sentimento para análise.")
            return sentiment_dynamics

        df = pd.DataFrame(dated_content)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Sentimento geral ao longo do tempo
        overall_sentiment_trend = df.groupby("date")["sentiment"].mean().reset_index(name="average_sentiment")
        sentiment_dynamics["overall_sentiment_trend"] = overall_sentiment_trend.to_dict(orient="records")

        # Sentimento por fonte
        sentiment_by_source = df.groupby("source")["sentiment"].mean().reset_index(name="average_sentiment")
        sentiment_dynamics["sentiment_by_source"] = sentiment_by_source.to_dict(orient="records")

        # Sentimento por tópico (requer topic modeling prévio ou aqui)
        if HAS_SPACY and self.nlp_model and HAS_GENSIM:
            try:
                processed_docs = []
                for text in df["text"]:
                    doc = self.nlp_model(text[:100000])
                    processed_docs.append([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                if processed_docs:
                    dictionary = corpora.Dictionary(processed_docs)
                    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                    lda_model = models.LdaModel(corpus, num_topics=self.config["n_topics_lda"], id2word=dictionary, passes=15)
                    sentiment_by_topic_data = defaultdict(lambda: defaultdict(float))
                    topic_counts = defaultdict(int)
                    for i, doc_bow in enumerate(corpus):
                        doc_topics = lda_model.get_document_topics(doc_bow)
                        for topic_id, prob in doc_topics:
                            sentiment_by_topic_data[f"topic_{topic_id}"]["sum_sentiment"] += df.iloc[i]["sentiment"] * prob
                            sentiment_by_topic_data[f"topic_{topic_id}"]["count"] += prob
                            topic_counts[f"topic_{topic_id}"] += 1
                    for topic_id, data in sentiment_by_topic_data.items():
                        if data["count"] > 0:
                            sentiment_dynamics["sentiment_by_topic"][topic_id] = data["sum_sentiment"] / data["count"]
                            # Adiciona palavras-chave do tópico para contexto
                            sentiment_dynamics["sentiment_by_topic"][topic_id + "_keywords"] = [word for word, _ in lda_model.show_topic(int(topic_id.split("_")[1]))]
            except Exception as e:
                logger.warning(f"⚠️ Erro na análise de sentimento por tópico: {e}")

        # Detecção de mudanças de sentimento (simplificado)
        if len(overall_sentiment_trend) > 1:
            df_sentiment_diff = overall_sentiment_trend.set_index("date")["average_sentiment"].diff().dropna()
            significant_shifts = df_sentiment_diff[df_sentiment_diff.abs() > 0.1] # Mudança > 0.1
            sentiment_dynamics["sentiment_shifts"] = significant_shifts.to_dict()

        logger.info("✅ Análise de dinâmica de sentimentos concluída.")
        return sentiment_dynamics

    async def _analyze_topic_evolution(self, session_dir: Path) -> Dict[str, Any]:
        """Analisa a evolução dos tópicos ao longo do tempo."""
        logger.info("🔄 Analisando evolução de tópicos...")
        topic_evolution = {
            "overall_topics": [],
            "topic_trends_over_time": {},
            "emerging_topics": [],
            "declining_topics": []
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return topic_evolution

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        all_text_content = []
        dated_content = []
        for item in massive_data.get("extracted_content", []):
            text = str(item.get("content", "")) + str(item.get("snippet", "")) + str(item.get("title", ""))
            timestamp_str = item.get("published_at") or item.get("timestamp") or massive_data.get("collection_started")
            if text and timestamp_str:
                all_text_content.append(text)
                try:
                    dt = None
                    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(timestamp_str, fmt)
                            break
                        except ValueError:
                            continue
                    if dt:
                        dated_content.append({"date": dt.date(), "text": text})
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao parsear data para evolução de tópicos: {e}")

        if not all_text_content or not dated_content:
            logger.warning("⚠️ Nenhum conteúdo textual ou datado para análise de evolução de tópicos.")
            return topic_evolution

        if not HAS_SPACY or not self.nlp_model or not HAS_GENSIM:
            logger.warning("⚠️ SpaCy ou Gensim não disponíveis para modelagem de tópicos.")
            return topic_evolution

        try:
            processed_docs = []
            for text in all_text_content:
                doc = self.nlp_model(text[:100000])
                processed_docs.append([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
            if not processed_docs:
                return topic_evolution

            dictionary = corpora.Dictionary(processed_docs)
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            lda_model = models.LdaModel(corpus, num_topics=self.config["n_topics_lda"], id2word=dictionary, passes=15)
            overall_topics = []
            for idx, topic in lda_model.print_topics(-1):
                overall_topics.append({"id": idx, "keywords": topic})
            topic_evolution["overall_topics"] = overall_topics

            # Tendência de tópicos ao longo do tempo
            df_dated = pd.DataFrame(dated_content)
            df_dated["date"] = pd.to_datetime(df_dated["date"])
            df_dated = df_dated.sort_values("date")
            topic_frequency_over_time = defaultdict(lambda: defaultdict(float))
            for i, row in df_dated.iterrows():
                doc = self.nlp_model(row["text"][:100000])
                bow = dictionary.doc2bow([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                doc_topics = lda_model.get_document_topics(bow)
                for topic_id, prob in doc_topics:
                    topic_frequency_over_time[str(row["date"])][f"topic_{topic_id}"] += prob
            topic_evolution["topic_trends_over_time"] = topic_frequency_over_time

            # Identificação de tópicos emergentes e em declínio (simplificado)
            # Compara a frequência do tópico no início e no final do período
            if len(df_dated["date"].unique()) > 1:
                dates = sorted(df_dated["date"].unique())
                first_period_data = df_dated[df_dated["date"] == dates[0]]
                last_period_data = df_dated[df_dated["date"] == dates[-1]]
                first_period_topics = Counter()
                for text in first_period_data["text"]:
                    doc = self.nlp_model(text[:100000])
                    bow = dictionary.doc2bow([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                    for topic_id, prob in lda_model.get_document_topics(bow):
                        first_period_topics[f"topic_{topic_id}"] += prob
                last_period_topics = Counter()
                for text in last_period_data["text"]:
                    doc = self.nlp_model(text[:100000])
                    bow = dictionary.doc2bow([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
                    for topic_id, prob in lda_model.get_document_topics(bow):
                        last_period_topics[f"topic_{topic_id}"] += prob
                for topic_id, initial_freq in first_period_topics.items():
                    final_freq = last_period_topics.get(topic_id, 0)
                    change = final_freq - initial_freq
                    if change > 0.1: # Aumento significativo
                        topic_evolution["emerging_topics"].append({"topic": topic_id, "change": change, "keywords": [word for word, _ in lda_model.show_topic(int(topic_id.split("_")[1]))]})
                    elif change < -0.1: # Declínio significativo
                        topic_evolution["declining_topics"].append({"topic": topic_id, "change": change, "keywords": [word for word, _ in lda_model.show_topic(int(topic_id.split("_")[1]))]})
        except Exception as e:
            logger.error(f"❌ Erro na análise de evolução de tópicos: {e}")

        logger.info("✅ Análise de evolução de tópicos concluída.")
        return topic_evolution

    async def _analyze_engagement_patterns(self, session_dir: Path) -> Dict[str, Any]:
        """Analisa padrões de engajamento em redes sociais e outras fontes."""
        logger.info("📊 Analisando padrões de engajamento...")
        engagement_patterns = {
            "engagement_by_platform": {},
            "engagement_by_content_type": {},
            "top_engaging_content": [],
            "engagement_prediction": {}
        }

        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return engagement_patterns

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        social_data = massive_data.get("social_media_data", {}).get("all_platforms_data", {}).get("platforms", {})
        all_engagements = []
        for platform_name, platform_info in social_data.items():
            if isinstance(platform_info, dict) and "results" in platform_info:
                for item in platform_info["results"]:
                    likes = item.get("likes", 0)
                    comments = item.get("comments", 0)
                    shares = item.get("shares", 0)
                    views = item.get("views", 0)
                    engagement_score = likes + comments * 2 + shares * 3 + views * 0.1 # Exemplo de score
                    all_engagements.append({
                        "platform": platform_name,
                        "content_type": item.get("type", "post"), # Ex: video, image, text
                        "engagement_score": engagement_score,
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at")
                    })

        if not all_engagements:
            logger.warning("⚠️ Nenhum dado de engajamento para análise.")
            return engagement_patterns

        df_engagement = pd.DataFrame(all_engagements)

        # Engajamento por plataforma
        engagement_by_platform = df_engagement.groupby("platform")["engagement_score"].mean().to_dict()
        engagement_patterns["engagement_by_platform"] = engagement_by_platform

        # Engajamento por tipo de conteúdo
        engagement_by_content_type = df_engagement.groupby("content_type")["engagement_score"].mean().to_dict()
        engagement_patterns["engagement_by_content_type"] = engagement_by_content_type

        # Top conteúdos com maior engajamento
        top_engaging_content = df_engagement.sort_values("engagement_score", ascending=False).head(10).to_dict(orient="records")
        engagement_patterns["top_engaging_content"] = top_engaging_content

        # Previsão de engajamento (simplificado com regressão linear)
        if HAS_SKLEARN and len(df_engagement) >= self.config["min_data_points_prediction"]:
            try:
                df_engagement["published_at"] = pd.to_datetime(df_engagement["published_at"])
                df_engagement["days_since_epoch"] = (df_engagement["published_at"] - datetime(1970, 1, 1)).dt.days
                X = df_engagement[["days_since_epoch"]].values.reshape(-1, 1)
                y = df_engagement["engagement_score"].values
                if len(X) > 1 and len(np.unique(X)) > 1: # Verifica se há variância suficiente
                    model = LinearRegression()
                    model.fit(X, y)
                    # Projeta para o futuro (ex: 30 dias)
                    last_day = df_engagement["days_since_epoch"].max()
                    future_days = np.array([[last_day + i] for i in range(1, 31)])
                    future_engagement = model.predict(future_days)
                    engagement_patterns["engagement_prediction"] = {
                        "model_coefficients": model.coef_.tolist(),
                        "model_intercept": model.intercept_,
                        "future_30_days_projection": future_engagement.tolist()
                    }
                else:
                    logger.warning("⚠️ Dados insuficientes para regressão linear de engajamento.")
            except Exception as e:
                logger.warning(f"⚠️ Erro na previsão de engajamento: {e}")

        logger.info("✅ Análise de padrões de engajamento concluída.")
        return engagement_patterns

    async def _generate_ultra_predictions(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Gera previsões ultra-avançadas com base em todos os insights."""
        logger.info("🔮 Gerando previsões ultra-avançadas...")
        predictions = {
            "market_trend_forecast": {},
            "sentiment_shift_prediction": {},
            "viral_content_potential": {},
            "sales_conversion_outlook": {},
            "competitor_response_prediction": {},
            "overall_market_outlook": ""
        }

        # Exemplo: Previsão de tendência de mercado baseada em volume de conteúdo e sentimento
        content_volume_forecast = insights.get("temporal_trends", {}).get("future_projections", {}).get("content_volume", {})
        avg_sentiment_forecast = insights.get("temporal_trends", {}).get("future_projections", {}).get("average_sentiment", {})
        if content_volume_forecast and avg_sentiment_forecast:
            predictions["market_trend_forecast"] = {
                "content_volume_90_days": content_volume_forecast.get("value_in_90_days"),
                "sentiment_90_days": avg_sentiment_forecast.get("value_in_90_days"),
                "overall_trend": "Crescimento positivo" if content_volume_forecast.get("trend") == "increasing" and avg_sentiment_forecast.get("trend") == "positive" else "Atenção necessária"
            }

        # Previsão de potencial de conteúdo viral
        top_engaging_content = insights.get("engagement_patterns", {}).get("top_engaging_content", [])
        if top_engaging_content:
            predictions["viral_content_potential"] = {
                "highest_engagement_score": top_engaging_content[0].get("engagement_score"),
                "most_viral_platform": top_engaging_content[0].get("platform"),
                "viral_drivers": insights.get("textual_insights", {}).get("top_keywords", [])[:5] # Keywords associadas
            }

        # Previsão de mudança de sentimento
        sentiment_shifts = insights.get("sentiment_dynamics", {}).get("sentiment_shifts", {})
        if sentiment_shifts:
            predictions["sentiment_shift_prediction"] = {
                "recent_shifts": sentiment_shifts,
                "potential_impact": "Alto" if any(abs(v) > 0.2 for v in sentiment_shifts.values()) else "Moderado"
            }

        # Outlook geral do mercado (combinação de tudo)
        overall_outlook = "O mercado apresenta tendências de "
        if predictions["market_trend_forecast"].get("overall_trend") == "Crescimento positivo":
            overall_outlook += "crescimento e sentimento positivo. "
        else:
            overall_outlook += "estabilidade, mas com pontos de atenção. "

        # CORRIGIDO: Linha 1142 - Indentação e aspas
        if predictions["viral_content_potential"]:
            overall_outlook += f"Há um alto potencial para conteúdo viral, especialmente em {predictions['viral_content_potential']['most_viral_platform']}. "

        predictions["overall_market_outlook"] = overall_outlook

        logger.info("✅ Previsões ultra-avançadas geradas.")
        return predictions

    async def _model_complex_scenarios(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Modela cenários complexos (otimista, pessimista, realista)."""
        logger.info("🗺️ Modelando cenários complexos...")
        scenarios = {
            "optimistic": {},
            "realistic": {},
            "pessimistic": {}
        }

        # Cenário Otimista: Tudo melhora
        scenarios["optimistic"] = {
            "market_growth": "Acelerado",
            "sentiment": "Altamente positivo",
            "viral_reach": "Máximo",
            "recommendation": "Investir agressivamente em expansão e inovação."
        }

        # Cenário Realista: Continuação das tendências atuais
        scenarios["realistic"] = {
            "market_growth": "Moderado",
            "sentiment": "Estável a ligeiramente positivo",
            "viral_reach": "Consistente",
            "recommendation": "Manter estratégias atuais com otimizações contínuas."
        }

        # Cenário Pessimista: Tendências negativas se acentuam
        scenarios["pessimistic"] = {
            "market_growth": "Estagnação ou declínio",
            "sentiment": "Negativo",
            "viral_reach": "Baixo",
            "recommendation": "Focar em retenção, otimização de custos e mitigação de riscos."
        }

        # Ajusta cenários com base em insights reais
        if insights.get("predictions", {}).get("overall_market_outlook"):
            outlook = insights["predictions"]["overall_market_outlook"]
            if "crescimento positivo" in outlook:
                scenarios["optimistic"]["justification"] = "Baseado nas fortes tendências de crescimento e sentimento positivo identificadas."
            elif "estabilidade" in outlook:
                scenarios["realistic"]["justification"] = "Baseado na estabilidade atual do mercado e padrões de engajamento."

        logger.info("✅ Cenários complexos modelados.")
        return scenarios

    async def _assess_risks_and_opportunities(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia riscos e oportunidades com base nos insights."""
        logger.info("⚖️ Avaliando riscos e oportunidades...")
        risk_assessment = {
            "identified_risks": [],
            "identified_opportunities": [],
            "risk_score": 0.0,
            "opportunity_score": 0.0
        }

        # Riscos
        if insights.get("data_quality_assessment", {}).get("issues_detected"):
            risk_assessment["identified_risks"].append({"type": "Data Quality", "description": "Problemas na qualidade dos dados podem afetar a precisão das previsões.", "severity": "Médio"})
            risk_assessment["risk_score"] += 0.2

        if insights.get("sentiment_dynamics", {}).get("sentiment_shifts"):
            if any(abs(v) > 0.2 for v in insights["sentiment_dynamics"]["sentiment_shifts"].values()):
                risk_assessment["identified_risks"].append({"type": "Sentiment Shift", "description": "Mudanças abruptas no sentimento podem indicar instabilidade no mercado ou na percepção da marca.", "severity": "Alto"})
                risk_assessment["risk_score"] += 0.3

        if insights.get("topic_evolution", {}).get("declining_topics"):
            risk_assessment["identified_risks"].append({"type": "Declining Topics", "description": "Tópicos em declínio podem indicar perda de interesse ou saturação do mercado.", "severity": "Médio"})
            risk_assessment["risk_score"] += 0.2

        # Oportunidades
        if insights.get("topic_evolution", {}).get("emerging_topics"):
            risk_assessment["identified_opportunities"].append({"type": "Emerging Topics", "description": "Tópicos emergentes representam novas áreas de interesse e potencial de crescimento.", "impact": "Alto"})
            risk_assessment["opportunity_score"] += 0.3

        if insights.get("engagement_patterns", {}).get("top_engaging_content"):
            risk_assessment["identified_opportunities"].append({"type": "High Engagement Content", "description": "Conteúdo de alto engajamento pode ser replicado ou adaptado para maximizar o alcance.", "impact": "Médio"})
            risk_assessment["opportunity_score"] += 0.2

        if insights.get("predictions", {}).get("viral_content_potential"):
            risk_assessment["identified_opportunities"].append({"type": "Viral Potential", "description": "Identificação de drivers de viralidade para campanhas futuras.", "impact": "Alto"})
            risk_assessment["opportunity_score"] += 0.3

        logger.info("✅ Avaliação de riscos e oportunidades concluída.")
        return risk_assessment

    async def _map_strategic_opportunities(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Mapeia oportunidades estratégicas detalhadas."""
        logger.info("🎯 Mapeando oportunidades estratégicas...")
        opportunity_mapping = {
            "product_development": [],
            "marketing_campaigns": [],
            "content_strategy": [],
            "partnership_potential": []
        }

        # Oportunidades de Desenvolvimento de Produto
        emerging_topics = insights.get("topic_evolution", {}).get("emerging_topics", [])
        for topic in emerging_topics:
            # CORRIGIDO: Linha 1244
            opportunity_mapping["product_development"].append({"description": f"Desenvolver funcionalidades ou produtos relacionados ao tópico emergente: {topic.get('keywords', '')}", "relevance": "Alta"})

        # Oportunidades de Campanhas de Marketing
        sentiment_by_topic = insights.get("sentiment_dynamics", {}).get("sentiment_by_topic", {})
        for topic, sentiment in sentiment_by_topic.items():
            if sentiment > 0.1: # Tópicos com sentimento positivo
                opportunity_mapping["marketing_campaigns"].append({"description": f"Lançar campanhas focadas no tópico {topic} aproveitando o sentimento positivo.", "relevance": "Média"})

        top_engaging_content = insights.get("engagement_patterns", {}).get("top_engaging_content", [])
        if top_engaging_content:
            # CORRIGIDO: Linha 1253
            opportunity_mapping["marketing_campaigns"].append({"description": f"Analisar e replicar o sucesso do conteúdo de maior engajamento: {top_engaging_content[0].get('title', '')}", "relevance": "Alta"})

        # Oportunidades de Estratégia de Conteúdo
        top_keywords = insights.get("textual_insights", {}).get("top_keywords", [])
        for keyword in top_keywords[:5]:
            opportunity_mapping["content_strategy"].append({"description": f"Criar conteúdo aprofundado sobre a palavra-chave: {keyword}", "relevance": "Alta"})

        # Potencial de Parcerias (simplificado, baseado em influenciadores)
        influencers = insights.get("network_analysis", {}).get("influencer_detection", [])
        for influencer, score in influencers[:3]:
            opportunity_mapping["partnership_potential"].append({"description": f"Explorar parceria com influenciador: {influencer} (Score: {score:.2f})", "relevance": "Média"})

        logger.info("✅ Oportunidades estratégicas mapeadas.")
        return opportunity_mapping

    async def _calculate_confidence_metrics(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas de confiança para as previsões e insights."""
        logger.info("📏 Calculando métricas de confiança...")
        confidence_metrics = {
            "data_coverage_score": 0.0,
            "model_accuracy_score": 0.0,
            "prediction_stability_score": 0.0,
            "overall_confidence_score": 0.0
        }

        # Cobertura de dados (baseado na quantidade de dados coletados)
        total_sources = insights.get("data_quality_assessment", {}).get("total_sources", 1) # Evita divisão por zero
        if total_sources > 0:
            confidence_metrics["data_coverage_score"] = min(100, (total_sources / 100) * 100) # Ex: 100 fontes = 100%

        # Precisão do modelo (simplificado, baseado na qualidade dos dados e existência de previsões)
        data_quality_score = insights.get("data_quality_assessment", {}).get("overall_quality_score", 0)
        if insights.get("predictions", {}).get("market_trend_forecast"):
            confidence_metrics["model_accuracy_score"] = data_quality_score * 0.8 + 20 # 80% da qualidade dos dados + 20% base
        else:
            confidence_metrics["model_accuracy_score"] = data_quality_score * 0.5 # Menos confiança sem previsões

        # Estabilidade da previsão (simplificado, baseado na variância das projeções)
        content_forecast = insights.get("temporal_trends", {}).get("prediction_models", {}).get("content_volume", [])
        if content_forecast and len(content_forecast) > 1:
            yhat_values = [item["yhat"] for item in content_forecast]
            std_dev = np.std(yhat_values)
            mean_val = np.mean(yhat_values)
            if mean_val != 0:
                cv = std_dev / mean_val # Coeficiente de variação
                confidence_metrics["prediction_stability_score"] = max(0, 100 - (cv * 100)) # Quanto menor CV, maior estabilidade
            else:
                confidence_metrics["prediction_stability_score"] = 50 # Neutro se média zero
        else:
            confidence_metrics["prediction_stability_score"] = 0

        # Score de confiança geral
        confidence_metrics["overall_confidence_score"] = np.mean([
            confidence_metrics["data_coverage_score"],
            confidence_metrics["model_accuracy_score"],
            confidence_metrics["prediction_stability_score"]
        ])

        logger.info("✅ Métricas de confiança calculadas.")
        return confidence_metrics

    async def _assess_data_quality(self, session_dir: Path) -> Dict[str, Any]:
        """Avalia a qualidade dos dados brutos coletados (chamada interna)."""
        logger.info("🔍 Avaliando qualidade dos dados brutos...")
        massive_data_path = session_dir / "massive_data_collected.json"
        if not massive_data_path.exists():
            logger.warning(f"⚠️ massive_data_collected.json não encontrado em {session_dir}")
            return {"success": False, "error": "Dados brutos não encontrados"}

        with open(massive_data_path, "r", encoding="utf-8") as f:
            massive_data = json.load(f)

        return await self.analyze_data_quality(massive_data) # Reutiliza o método existente

    async def _generate_strategic_recommendations(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Gera recomendações estratégicas acionáveis."""
        logger.info("💡 Gerando recomendações estratégicas...")
        strategic_recommendations = {
            "short_term": [],
            "medium_term": [],
            "long_term": []
        }

        # Curto Prazo: Baseado em oportunidades imediatas e riscos urgentes
        emerging_topics = insights.get("opportunity_mapping", {}).get("content_strategy", [])
        if emerging_topics:
            # CORRIGIDO: Linha 1429
            strategic_recommendations["short_term"].append({"action": f"Focar na criação de conteúdo sobre os tópicos emergentes: {emerging_topics[0].get('description', '')}", "priority": "Alta"})

        sentiment_shifts = insights.get("risk_assessment", {}).get("identified_risks", [])
        for risk in sentiment_shifts:
            if risk.get("type") == "Sentiment Shift" and risk.get("severity") == "Alto":
                strategic_recommendations["short_term"].append({"action": "Monitorar e responder proativamente a mudanças negativas de sentimento.", "priority": "Urgente"})

        # Médio Prazo: Baseado em tendências e padrões de engajamento
        if insights.get("predictions", {}).get("viral_content_potential"):
            # CORRIGIDO: Linha 1434
            strategic_recommendations["medium_term"].append({"action": f"Desenvolver campanhas de marketing com base nos drivers de viralidade identificados em {insights['predictions']['viral_content_potential']['most_viral_platform']}.", "priority": "Média"})

        if insights.get("engagement_patterns", {}).get("engagement_by_platform"):
            top_platform = max(insights["engagement_patterns"]["engagement_by_platform"], key=insights["engagement_patterns"]["engagement_by_platform"].get)
            strategic_recommendations["medium_term"].append({"action": f"Otimizar a presença e o conteúdo na plataforma de maior engajamento: {top_platform}.", "priority": "Média"})

        # Longo Prazo: Baseado em cenários e oportunidades de desenvolvimento de produto
        product_opportunities = insights.get("opportunity_mapping", {}).get("product_development", [])
        if product_opportunities:
            # CORRIGIDO: Linha 1440
            strategic_recommendations["long_term"].append({"action": f"Investir em pesquisa e desenvolvimento para novos produtos/serviços alinhados com {product_opportunities[0].get('description', '')}.", "priority": "Alta"})

        if insights.get("scenarios", {}).get("optimistic"):
            strategic_recommendations["long_term"].append({"action": insights["scenarios"]["optimistic"]["recommendation"], "priority": "Flexível"})

        logger.info("✅ Recomendações estratégicas geradas.")
        return strategic_recommendations

    async def _prioritize_actions(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Prioriza ações com base em impacto e urgência."""
        logger.info("🎯 Priorizando ações...")
        action_priorities = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        # Combina todas as recomendações
        all_recommendations = []
        for term in ["short_term", "medium_term", "long_term"]:
            for rec in insights.get("strategic_recommendations", {}).get(term, []):
                all_recommendations.append(rec)

        # Lógica de priorização (exemplo simples)
        for rec in all_recommendations:
            priority = rec.get("priority", "Média")
            if priority == "Urgente":
                action_priorities["critical"].append(rec)
            elif priority == "Alta":
                action_priorities["high"].append(rec)
            elif priority == "Média":
                action_priorities["medium"].append(rec)
            else:
                action_priorities["low"].append(rec)

        logger.info("✅ Ações priorizadas.")
        return action_priorities

# Instância global
predictive_analytics_engine = PredictiveAnalyticsEngine()
