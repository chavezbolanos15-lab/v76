#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Alibaba WebSailor Agent
Agente de navegação web inteligente com busca profunda e análise contextual
"""

import os
import logging
import time
import requests
import json
import random
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus, urljoin, urlparse
from bs4 import BeautifulSoup
# Removido: from readability import Document as ReadabilityDocument

from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.auto_save_manager import salvar_etapa, salvar_erro, salvar_trecho_pesquisa_web # Adicionado salvar_trecho_pesquisa_web

# Import para integração com Exa
try:
    from services.exa_client import exa_client, extract_content_with_exa
    HAS_EXA = True
except ImportError:
    HAS_EXA = False
    logging.warning("⚠️ Exa client não encontrado. A extração de conteúdo via Exa será desativada.")

logger = logging.getLogger(__name__)

class AlibabaWebSailorAgent:
    """Agente WebSailor inteligente para navegação e análise web profunda"""

    def __init__(self):
        """Inicializa agente WebSailor"""
        self.enabled = True
        self.google_search_key = os.getenv("GOOGLE_SEARCH_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

        # URLs das APIs
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        self.jina_reader_url = "https://r.jina.ai/"
        self.serper_url = "https://google.serper.dev/search"

        # Headers inteligentes para navegação
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none"
        }

        # Domínios brasileiros preferenciais
        self.preferred_domains = {
            "g1.globo.com", "exame.com", "valor.globo.com", "estadao.com.br",
            "folha.uol.com.br", "canaltech.com.br", "tecmundo.com.br",
            "olhardigital.com.br", "infomoney.com.br", "startse.com",
            "revistapegn.globo.com", "epocanegocios.globo.com", "istoedinheiro.com.br",
            "convergenciadigital.com.br", "mobiletime.com.br", "teletime.com.br",
            "portaltelemedicina.com.br", "saudedigitalbrasil.com.br", "amb.org.br",
            "portal.cfm.org.br", "scielo.br", "ibge.gov.br", "fiocruz.br"
        }

        # Domínios bloqueados (irrelevantes)
        self.blocked_domains = {"airbnb.com"}

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Estatísticas de navegação
        self.navigation_stats = {
            'total_searches': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'blocked_urls': 0,
            'preferred_sources': 0,
            'total_content_chars': 0,
            'avg_quality_score': 0.0
        }

        logger.info("🌐 Alibaba WebSailor Agent inicializado - Navegação inteligente ativada")

    def navigate_and_research_deep(
        self,
        query: str,
        context: Dict[str, Any],
        max_pages: int = 25,
        depth_levels: int = 3,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Navegação e pesquisa profunda com múltiplos níveis"""
        try:
            logger.info(f"🚀 INICIANDO NAVEGAÇÃO PROFUNDA para: {query}")
            start_time = time.time()

            # Salva início da navegação
            salvar_etapa("websailor_iniciado", {
                "query": query,
                "context": context,
                "max_pages": max_pages,
                "depth_levels": depth_levels
            }, categoria="pesquisa_web")

            all_content = []
            search_engines_used = []

            # NÍVEL 1: BUSCA MASSIVA MULTI-ENGINE
            logger.info("🔍 NÍVEL 1: Busca massiva com múltiplos engines")

            # Engines de busca em ordem de prioridade
            search_engines = [
                ("Google Custom Search", self._google_search_deep),
                ("Serper API", self._serper_search_deep),
                # ("Bing Scraping", self._bing_search_deep), # Comentado por padrão
                # ("DuckDuckGo Scraping", self._duckduckgo_search_deep), # Comentado por padrão
                # ("Yahoo Scraping", self._yahoo_search_deep) # Comentado por padrão
            ]

            # Executa buscas em paralelo
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_engine = {
                    executor.submit(func, query, min(max_pages, 10)): name
                    for name, func in search_engines if func is not None
                }

                for future in as_completed(future_to_engine):
                    engine_name = future_to_engine[future]
                    try:
                        results = future.result()
                        if results:
                            search_engines_used.append(engine_name)
                            logger.info(f"✅ {engine_name}: {len(results)} resultados")

                            # Processa cada resultado
                            for result in results[:5]:  # Limita resultados por engine
                                url = result.get('url') or result.get('link')
                                if not url:
                                    continue

                                # Extrai conteúdo com múltiplas estratégias
                                content_data = self._extract_content_multi_strategy(
                                    url, result.get('title', ''), context, session_id # Passa session_id
                                )

                                if content_data and content_data.get('success'):
                                    content_data['search_engine'] = engine_name
                                    content_data['content_length'] = len(content_data.get('content', ''))
                                    all_content.append(content_data)

                                    # === NOVO: Salva o trecho extraído ===
                                    salvar_trecho_pesquisa_web(
                                        url=url,
                                        titulo=content_data.get('title', ''),
                                        conteudo=content_data.get('content', ''),
                                        metodo_extracao=content_data.get('extraction_method', 'desconhecido'),
                                        qualidade=content_data.get('quality_score', 0.0),
                                        session_id=session_id or 'sessao_desconhecida'
                                    )
                                    # ======================================

                                    # Salva cada extração bem-sucedida (mantido para compatibilidade)
                                    salvar_etapa(f"websailor_extracao_{len(all_content)}", {
                                        "url": url, # Corrigido: era result['url'], agora é url
                                        "engine": engine_name,
                                        "content_length": len(content_data.get('content', '')), # Corrigido: era content_data['content']
                                        "quality_score": content_data.get('quality_score', 0.0) # Corrigido: era content_data['quality_score']
                                    }, categoria="pesquisa_web")

                                time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        logger.error(f"❌ Erro em {engine_name}: {str(e)}")
                        continue

            # NÍVEL 2: BUSCA EM PROFUNDIDADE (Links internos)
            if depth_levels > 1 and all_content:
                logger.info("🔍 NÍVEL 2: Busca em profundidade - Links internos")
                # Implementação simplificada para busca em profundidade
                # Pode envolver análise de links encontrados nos conteúdos iniciais
                # e nova extração de conteúdo desses links.
                # Por simplicidade, vamos pular esta etapa neste exemplo.

            # NÍVEL 3: BUSCA CONTEXTUAL AVANÇADA (Queries relacionadas)
            if depth_levels > 2 and all_content:
                logger.info("🔍 NÍVEL 3: Busca contextual avançada - Queries relacionadas")
                related_queries = self._generate_related_queries(query, context, all_content)

                for related_query in related_queries[:3]:  # Limita queries relacionadas
                    try:
                        logger.info(f"🔍 Buscando por query relacionada: {related_query}")
                        # Reutiliza o método de busca principal para a query relacionada
                        related_content = self.navigate_and_research_deep(
                            related_query, context, max_pages=5, depth_levels=1, session_id=session_id # Passa session_id
                        )
                        # Extrai insights relevantes e os adiciona
                        related_content['related_query'] = related_query
                        all_content.append(related_content)
                        time.sleep(0.4)
                    except Exception as e:
                        logger.warning(f"⚠️ Erro em query relacionada '{related_query}': {str(e)}")
                        continue

            # PROCESSAMENTO E ANÁLISE FINAL
            processed_research = self._process_and_analyze_content(all_content, query, context)

            # Atualiza estatísticas
            self._update_navigation_stats(all_content)

            end_time = time.time()

            # Salva resultado final da navegação
            salvar_etapa("websailor_resultado", processed_research, categoria="pesquisa_web")

            logger.info(f"✅ NAVEGAÇÃO PROFUNDA CONCLUÍDA em {end_time - start_time:.2f} segundos")
            logger.info(f"📊 {len(all_content)} páginas analisadas com {len(search_engines_used)} engines")

            return processed_research

        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO na navegação WebSailor: {str(e)}")
            salvar_erro("websailor_critico", e, contexto={"query": query})
            return self._generate_emergency_research(query, context)

    def _extract_content_multi_strategy(
        self,
        url: str,
        title: str,
        context: Dict[str, Any],
        session_id: str # Adicionado session_id
    ) -> Optional[Dict[str, Any]]:
        """Estratégia multi-método para extração de conteúdo"""
        content_data = None
        methods_tried = []

        # 1. Tenta Jina Reader (método principal)
        if self.jina_api_key:
            try:
                logger.info(f"🔍 Tentando Jina Reader para {url}")
                headers = {**self.headers, "Authorization": f"Bearer {self.jina_api_key}"}
                response = self.session.get(f"{self.jina_reader_url}{url}", headers=headers, timeout=20)

                if response.status_code == 200:
                    content = response.text
                    if content and len(content) > 100: # Threshold mínimo
                        content_data = {
                            'success': True,
                            'url': url,
                            'title': title,
                            'content': content,
                            'quality_score': self._calculate_content_quality(content, url, context),
                            'extraction_method': 'jina_reader'
                        }
                        logger.info(f"✅ Jina Reader: {len(content)} caracteres")
                else:
                    logger.warning(f"⚠️ Jina Reader retornou status {response.status_code} para {url}")
                    methods_tried.append(f"Jina ({response.status_code})")
            except Exception as e:
                logger.warning(f"⚠️ Jina Reader falhou para {url}: {e}")
                methods_tried.append(f"Jina (Erro)")

        # 2. Se Jina falhar, usa fallbacks (Exa -> BeautifulSoup)
        if not content_data or not content_data.get('success'):
            content_data = self._fallback_extraction(url, title, context, session_id) # Passa session_id
            if content_data and content_data.get('success'):
                 methods_tried.append(content_data.get('extraction_method', 'fallback'))
            else:
                 methods_tried.append("Todos_Fallbacks_Falharam")

        if content_data and content_data.get('success'):
            content_data['methods_tried'] = methods_tried
            is_preferred = any(domain in urlparse(url).netloc.lower() for domain in self.preferred_domains)
            content_data['is_preferred_source'] = is_preferred
            if is_preferred:
                self.navigation_stats['preferred_sources'] += 1
            return content_data
        else:
            self.navigation_stats['failed_extractions'] += 1
            logger.warning(f"⚠️ Falha em todos os métodos de extração para {url}. Métodos tentados: {methods_tried}")
            return None

    def _fallback_extraction(
        self,
        url: str,
        title: str,
        context: Dict[str, Any],
        session_id: str # Adicionado session_id
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback para extração de conteúdo quando métodos principais falham.
        Ordem: Exa -> BeautifulSoup
        """
        logger.info(f"🔄 Usando fallback para extrair conteúdo de {url}")
        content = None
        extraction_method = "unknown"

        # --- MODIFICAÇÃO 1: Tenta Exa primeiro (se disponível) ---
        if HAS_EXA and exa_client.is_available():
            try:
                logger.info(f"🔍 Tentando Exa para {url}")
                content = extract_content_with_exa(url)
                if content and len(content.strip()) > 50:
                    logger.info(f"✅ Conteúdo extraído com Exa ({len(content)} caracteres)")
                    extraction_method = "exa"
                    return {
                        'success': True,
                        'url': url,
                        'title': title,
                        'content': content,
                        'quality_score': self._calculate_content_quality(content, url, context),
                        'extraction_method': extraction_method
                    }
                else:
                    logger.info(f"ℹ️ Exa não retornou conteúdo suficiente ou falhou.")
            except Exception as e:
                logger.warning(f"⚠️ Exa falhou como fallback para {url}: {e}")

        # --- REMOVIDO: Tenta Readability ---

        # --- Tenta BeautifulSoup como último recurso ---
        try:
            logger.info(f"🔍 Tentando BeautifulSoup para {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove elementos indesejados (scripts, styles, etc.)
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()

            # Tenta encontrar o conteúdo principal (heurística simples)
            # Pode ser melhorado com seletores mais específicos
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|post'))
            if main_content:
                text_content = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback para todo o body
                body = soup.find('body')
                text_content = body.get_text(separator=' ', strip=True) if body else ""

            # Limita o tamanho se necessário e remove excesso de espaços
            text_content = re.sub(r'\s+', ' ', text_content).strip()[:10000] # Limite arbitrário

            if text_content and len(text_content) > 50:
                logger.info(f"✅ Conteúdo extraído com BeautifulSoup ({len(text_content)} caracteres)")
                extraction_method = "beautifulsoup"
                return {
                    'success': True,
                    'url': url,
                    'title': title,
                    'content': text_content,
                    'quality_score': self._calculate_content_quality(text_content, url, context),
                    'extraction_method': extraction_method
                }
            else:
                logger.info(f"ℹ️ BeautifulSoup não retornou conteúdo suficiente.")
        except Exception as e:
            logger.error(f"❌ BeautifulSoup falhou como último fallback para {url}: {e}")

        logger.warning(f"⚠️ Todos os métodos de fallback falharam para {url}")
        return None

    def _calculate_content_quality(self, content: str, url: str, context: Dict[str, Any]) -> float:
        """Calcula score de qualidade do conteúdo"""
        if not content:
            return 0.0

        score = 0.0

        # Score por relevância do domínio (máximo 25 pontos)
        domain = urlparse(url).netloc.lower()
        if any(pref in domain for pref in self.preferred_domains):
            score += 20
        elif domain.endswith('.gov.br') or domain.endswith('.edu.br'):
            score += 15
        elif domain.endswith('.org.br'):
            score += 10
        else:
            score += 5

        # Score por densidade de informação (máximo 15 pontos)
        words = content.split()
        if len(words) >= 500:
            score += 15
        elif len(words) >= 200:
            score += 10
        else:
            score += 5

        # Score por presença de dados (máximo 15 pontos)
        data_patterns = [
            r'\d+%', r'R\$\s*[\d,\.]+', r'\d+\s*(mil|milhão|bilhão)',
            r'20(23|24|25)', r'\d+\s*(empresas|profissionais|clientes)'
        ]
        data_matches = sum(len(re.findall(pattern, content)) for pattern in data_patterns)
        if data_matches >= 5:
            score += 15
        elif data_matches >= 2:
            score += 10
        else:
            score += 5

        # Score por contexto (máximo 20 pontos)
        context_keywords = context.get('keywords', []) + context.get('segmento', '').split()
        context_matches = sum(1 for keyword in context_keywords if keyword.lower() in content.lower())
        if context_matches >= 5:
            score += 20
        elif context_matches >= 2:
            score += 10
        else:
            score += 5

        # Score por frescor (máximo 10 pontos) - Simples verificação de ano
        current_year = datetime.now().year
        year_matches = re.findall(rf'20(2[3-9]|[3-9]\d)', content) # Anos 2023-2099
        if any(int(y) >= current_year - 1 for y in year_matches): # Últimos 2 anos
            score += 10
        elif any(int(y) >= current_year - 3 for y in year_matches): # Últimos 4 anos
            score += 5
        else:
            score += 2

        # Normaliza score para 0-100
        return min(100.0, max(0.0, score))

    def _extract_content_insights(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Extrai insights específicos do conteúdo"""
        insights = []
        # Lógica simplificada para extrair insights
        # Pode ser expandida com NLP, regex mais complexo, etc.
        sentences = re.split(r'[.!?]+', content)
        key_terms = context.get('keywords', [])
        for sentence in sentences[:20]: # Limita para performance
            if any(term.lower() in sentence.lower() for term in key_terms):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                    insights.append(clean_sentence)
        return list(set(insights))[:10] # Remove duplicatas e limita

    def _process_and_analyze_content(
        self,
        all_content: List[Dict[str, Any]],
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processa e analisa todo o conteúdo coletado"""
        if not all_content:
            return self._generate_emergency_research(query, context)

        # Ordena por qualidade
        all_content.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        # Extrai insights únicos
        all_insights = []
        for item in all_content:
            insights = item.get('insights', [])
            all_insights.extend(insights)

        unique_insights = list(set(all_insights))

        # Identifica tendências (simplificado)
        trends = []
        opportunities = []
        # Esta parte pode ser significativamente aprimorada com NLP e análise de frequência
        trend_keywords = ['crescimento', 'tendência', 'mercado', 'inovação']
        opportunity_keywords = ['oportunidade', 'chance', 'potencial', 'nicho']

        for insight in unique_insights[:10]: # Analisa top insights
            insight_lower = insight.lower()
            if any(tk in insight_lower for tk in trend_keywords):
                trends.append(insight)
            if any(ok in insight_lower for ok in opportunity_keywords):
                opportunities.append(insight)

        # Calcula estatísticas
        total_chars = sum(item.get('content_length', 0) for item in all_content)
        avg_quality = sum(item.get('quality_score', 0) for item in all_content) / len(all_content) if all_content else 0

        return {
            "query_original": query,
            "context": context,
            "navegacao_profunda": {
                "total_paginas_analisadas": len(all_content),
                "engines_utilizados": list(set(item['search_engine'] for item in all_content if 'search_engine' in item)),
                "fontes_preferenciais": sum(1 for item in all_content if item.get('is_preferred_source')),
                "qualidade_media": round(avg_quality, 2),
                "total_caracteres": total_chars,
                "insights_unicos": len(unique_insights)
            },
            "conteudo_consolidado": {
                "insights_principais": unique_insights[:20],
                "tendencias_identificadas": trends[:10],
                "oportunidades_descobertas": opportunities[:10],
                "fontes_detalhadas": [{
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'quality_score': item.get('quality_score', 0),
                    'content_length': item.get('content_length', 0),
                    'search_engine': item.get('search_engine', 'Desconhecido'),
                    'is_preferred': item.get('is_preferred_source', False)
                } for item in all_content[:15]]
            },
            "estatisticas_navegacao": self.navigation_stats,
            "metadata": {
                "navegacao_concluida_em": datetime.now().isoformat(),
                "agente": "Alibaba_WebSailor_v2.0",
                "garantia_dados_reais": True,
                "simulacao_free": True,
                "qualidade_premium": avg_quality >= 80
            }
        }

    def _update_navigation_stats(self, content_list: List[Dict[str, Any]]):
        """Atualiza estatísticas de navegação"""
        if content_list:
            avg_quality = sum(item.get('quality_score', 0) for item in content_list) / len(content_list)
            self.navigation_stats['avg_quality_score'] = avg_quality

    def _generate_related_queries(
        self,
        query: str,
        context: Dict[str, Any],
        content_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Gera queries de busca relacionadas"""
        related_queries = []
        segmento = context.get('segmento', '')
        produto = context.get('produto', '')

        # Queries baseadas no contexto
        base_queries = [
            f"{query} tendências mercado",
            f"{query} análise concorrência",
            f"{segmento} {produto} inovação",
            f"{produto} benchmarks Brasil",
            f"concorrência {produto} mercado brasileiro",
            f"preços {produto} benchmarks Brasil"
        ]
        related_queries.extend(base_queries)

        # Adiciona queries baseadas em termos frequentes nos conteúdos (simplificado)
        # Em uma implementação real, isso usaria NLP para extrair termos-chave
        all_text = " ".join(item.get('content', '') for item in content_list)
        words = re.findall(r'\b\w{4,}\b', all_text.lower()) # Palavras com 4+ caracteres
        word_freq = {}
        for word in words:
            if word not in self.preferred_domains and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Ordena por frequência e pega os top 5
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        relevant_terms = [word for word, freq in sorted_words[:5]]

        # Adiciona queries baseadas em termos frequentes
        for term in relevant_terms[:3]:
            related_queries.append(f"{term} {segmento} Brasil oportunidades")

        return list(set(related_queries))[:8] # Remove duplicatas e limita a 8

    def _generate_emergency_research(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gera pesquisa de emergência quando navegação falha"""
        logger.warning("⚠️ Gerando pesquisa de emergência WebSailor")
        return {
            "query_original": query,
            "context": context,
            "navegacao_profunda": {
                "total_paginas_analisadas": 0,
                "engines_utilizados": [],
                "status": "emergencia",
                "message": "Navegação em modo de emergência - configure APIs para dados completos"
            },
            "conteudo_consolidado": {
                "insights_principais": [
                    f"Pesquisa emergencial para '{query}' - sistema em recuperação",
                    "Recomenda-se nova tentativa com configuração completa das APIs",
                    "WebSailor em modo de emergência - funcionalidade limitada"
                ],
                "tendencias_identificadas": ["Sistema em modo de emergência - tendências limitadas"],
                "oportunidades_descobertas": ["Reconfigurar APIs para navegação completa"]
            },
            "metadata": {
                "navegacao_concluida_em": datetime.now().isoformat(),
                "agente": "Alibaba_WebSailor_Emergency",
                "garantia_dados_reais": False,
                "modo_emergencia": True
            }
        }

    def get_navigation_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de navegação"""
        return self.navigation_stats.copy()

    def reset_navigation_stats(self):
        """Reset estatísticas de navegação"""
        self.navigation_stats = {
            'total_searches': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'blocked_urls': 0,
            'preferred_sources': 0,
            'total_content_chars': 0,
            'avg_quality_score': 0.0
        }
        logger.info("🔄 Estatísticas de navegação resetadas")

    # --- Métodos de busca em engines específicas (mantidos do original) ---

    def _google_search_deep(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Busca profunda usando Google Custom Search API"""
        if not self.google_search_key or not self.google_cse_id:
            return []

        try:
            params = {
                'key': self.google_search_key,
                'cx': self.google_cse_id,
                'q': self._enhance_query_for_brazil(query),
                'num': min(max_results, 10), # Google limita a 10 por requisição
                'gl': 'br',
                'hl': 'pt'
            }
            response = self.session.get(self.google_search_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("items", []):
                    url = item.get("link", "")
                    if self._is_url_relevant(url, item.get("title", ""), item.get("snippet", "")):
                        results.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("snippet", ""),
                            "displayLink": item.get("displayLink", "")
                        })
                return results
            else:
                logger.error(f"❌ Erro Google Search: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Erro na requisição Google Search: {str(e)}")
        return []

    def _serper_search_deep(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Busca profunda usando Serper API"""
        if not self.serper_api_key:
            return []

        try:
            headers = {
                **self.headers,
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': self._enhance_query_for_brazil(query),
                'gl': 'br',
                'hl': 'pt',
                'num': max_results,
                'autocorrect': True,
                'page': 1
            }
            response = self.session.post(self.serper_url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("organic", []):
                    url = item.get("link", "")
                    if self._is_url_relevant(url, item.get("title", ""), item.get("snippet", "")):
                        results.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("snippet", ""),
                            "position": item.get("position", "")
                        })
                return results
            else:
                logger.error(f"❌ Erro Serper: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Erro na requisição Serper: {str(e)}")
        return []

    def _is_url_relevant(self, url: str, title: str, snippet: str) -> bool:
        """Verifica se URL é relevante"""
        if not url:
            return False

        domain = urlparse(url).netloc.lower()

        # Bloqueia domínios irrelevantes
        if any(blocked in domain for blocked in self.blocked_domains):
            return False

        # Verifica se é um arquivo indesejado
        if any(url.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
            return False

        return True

    def _enhance_query_for_brazil(self, query: str) -> str:
        """Melhora query para busca no contexto brasileiro"""
        brazilian_terms = ["Brasil", "brasileiro", "mercado brasileiro"]
        if not any(term in query for term in brazilian_terms):
            return f"{query} Brasil"
        return query

# Instância global
alibaba_websailor = AlibabaWebSailorAgent()
