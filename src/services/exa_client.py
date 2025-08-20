#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - Exa Client
Cliente para integração com Exa API para pesquisa avançada
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ExaClient:
    """Cliente para integração com Exa API"""
    
    def __init__(self):
        """Inicializa cliente Exa"""
        self.api_key = os.getenv("EXA_API_KEY")
        # CORREÇÃO 1: Remover espaço extra na URL
        self.base_url = "https://api.exa.ai"
        
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # CORREÇÃO 2: Verificação mais robusta da disponibilidade
        self.available = bool(self.api_key) and self.api_key != "sua-chave-aqui" # Assumindo 'sua-chave-aqui' como placeholder
        
        if self.available:
            logger.info("✅ Exa client inicializado com sucesso")
        else:
            logger.warning("⚠️ Exa API key não configurada ou inválida")

    def is_available(self) -> bool:
        """Verifica se o cliente está disponível"""
        return self.available

    def search(
        self,
        query: str,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None, # CORREÇÃO 3: Tipo correto
        exclude_domains: Optional[List[str]] = None, # CORREÇÃO 3: Tipo correto
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: bool = True,
        type: str = "neural"
    ) -> Optional[Dict[str, Any]]:
        """Realiza busca usando Exa API"""

        if not self.available:
            logger.warning("Exa não está disponível - API key não configurada")
            return None

        try:
            payload = {
                "query": query,
                "numResults": min(num_results, 100), # Limitar conforme a API
                "useAutoprompt": use_autoprompt,
                "type": type
            }

            # Adicionar parâmetros opcionais somente se forem fornecidos
            if include_domains:
                payload["includeDomains"] = include_domains

            if exclude_domains:
                payload["excludeDomains"] = exclude_domains

            if start_crawl_date:
                payload["startCrawlDate"] = start_crawl_date

            if end_crawl_date:
                payload["endCrawlDate"] = end_crawl_date

            if start_published_date:
                payload["startPublishedDate"] = start_published_date

            if end_published_date:
                payload["endPublishedDate"] = end_published_date

            # CORREÇÃO 4: Tratamento de erro mais específico
            response = requests.post(
                f"{self.base_url}/search",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status() # Lança exceção para códigos de status HTTP 4xx/5xx

            data = response.json()
            logger.info(f"✅ Exa search: {len(data.get('results', []))} resultados para query '{query}'")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro HTTP na requisição Exa search: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao decodificar JSON da resposta Exa search: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado na requisição Exa search: {str(e)}")
            return None

    def get_contents(
        self,
        ids: List[str], # IDs são obrigatórios para esta função
        text: bool = True,
        highlights: bool = False,
        summary: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Obtém conteúdo detalhado dos resultados"""

        if not self.available:
            logger.warning("Exa não está disponível - API key não configurada")
            return None

        if not ids:
             logger.warning("Nenhum ID fornecido para get_contents")
             return None

        try:
            payload = {
                "ids": ids,
                "text": text,
                "highlights": highlights,
                "summary": summary
            }

            # CORREÇÃO 4: Tratamento de erro mais específico
            response = requests.post(
                f"{self.base_url}/contents",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            logger.info(f"✅ Exa contents: {len(data.get('results', []))} conteúdos obtidos")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro HTTP na requisição Exa contents: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao decodificar JSON da resposta Exa contents: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao obter conteúdos Exa: {str(e)}")
            return None

    def find_similar(
        self,
        url: str,
        num_results: int = 10,
        exclude_source_domain: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Encontra páginas similares"""

        if not self.available:
            logger.warning("Exa não está disponível - API key não configurada")
            return None

        if not url:
             logger.warning("Nenhuma URL fornecida para find_similar")
             return None

        try:
            payload = {
                "url": url,
                "numResults": min(num_results, 100), # Limitar conforme a API
                "excludeSourceDomain": exclude_source_domain
            }

            # CORREÇÃO 4: Tratamento de erro mais específico
            response = requests.post(
                f"{self.base_url}/findSimilar",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            logger.info(f"✅ Exa findSimilar: {len(data.get('results', []))} resultados similares encontrados")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro HTTP na requisição Exa findSimilar: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao decodificar JSON da resposta Exa findSimilar: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao buscar similares Exa: {str(e)}")
            return None

# Instância global
exa_client = ExaClient()

# Função auxiliar para integração com o WebSailor (opcional, mas útil)
def extract_content_with_exa(url: str) -> Optional[str]:
    """
    Função de conveniência para extrair conteúdo de uma única URL usando Exa.
    Retorna o texto principal da página.
    """
    if not exa_client.is_available():
        logger.warning("Exa client não disponível para extração de conteúdo")
        return None

    try:
        # Primeiro, podemos fazer uma busca simples pela URL para obter o ID
        # Ou usar get_contents diretamente se tivermos o ID
        # Para simplificar, vamos assumir que podemos usar get_contents com a URL
        # (Verificar documentação da Exa se isso é suportado diretamente)
        
        # Alternativa: usar search para encontrar a URL e depois get_contents
        search_result = exa_client.search(f"url:{url}", num_results=1)
        if search_result and search_result.get('results'):
            result_id = search_result['results'][0].get('id')
            if result_id:
                content_result = exa_client.get_contents([result_id], text=True, highlights=False, summary=False)
                if content_result and content_result.get('results'):
                    # Retorna o texto do primeiro resultado
                    return content_result['results'][0].get('text', '')
        # Se a busca direta por URL não funcionar, tenta get_contents diretamente (se suportado pela API)
        # content_result = exa_client.get_contents([url], text=True, highlights=False, summary=False)
        # if content_result and content_result.get('results'):
        #     return content_result['results'][0].get('text', '')

    except Exception as e:
        logger.error(f"❌ Erro ao extrair conteúdo com Exa para {url}: {e}")

    logger.warning(f"⚠️ Não foi possível extrair conteúdo com Exa para {url}")
    return None
