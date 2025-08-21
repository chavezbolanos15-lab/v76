#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v4.1 - OpenRouter API Manager (refatorado)
Gerenciador de rotação de APIs do OpenRouter com fallback automático de chaves,
redução dinâmica de max_tokens e fallback de modelos. Interfaces sync/async.
"""

import os
import logging
import time
import random
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenRouterAPIManager:
    """Gerenciador de rotação de APIs do OpenRouter"""

    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.key_cooldowns: Dict[str, datetime] = {}
        self.max_retries = 3
        self.retry_delay = 2
        # Cadeia de fallback de modelos (do mais desejado ao mais barato)
        self.model_fallback_chain: List[str] = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-32b-instruct",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mixtral-8x7b-instruct",
        ]

        # Inicializa estatísticas para cada chave
        for key in self.api_keys:
            self.key_usage_stats[key] = {
                'requests': 0,
                'errors': 0,
                'last_used': None,
                'credits_exhausted': False,
                'rate_limited': False
            }

        logger.info(f"🔄 OpenRouter API Manager inicializado com {len(self.api_keys)} chaves")

    def _load_api_keys(self) -> List[str]:
        """Carrega todas as chaves de API disponíveis"""
        keys: List[str] = []

        # Chave principal
        main_key = os.getenv('OPENROUTER_API_KEY')
        if main_key:
            keys.append(main_key)

        # Chaves adicionais
        for i in range(1, 10):  # Suporte até 10 chaves adicionais
            key = os.getenv(f'OPENROUTER_API_KEY_{i}')
            if key:
                keys.append(key)

        if not keys:
            raise ValueError("Nenhuma chave OpenRouter encontrada nas variáveis de ambiente")

        return keys

    def _get_next_available_key(self) -> str:
        """Obtém a próxima chave disponível na rotação"""
        attempts = 0
        total = len(self.api_keys)
        while attempts < total:
            key = self.api_keys[self.current_key_index]
            stats = self.key_usage_stats[key]

            # Verifica cooldown
            cooldown_until = self.key_cooldowns.get(key)
            if cooldown_until and datetime.now() < cooldown_until:
                logger.debug(f"Chave {key[:10]}... em cooldown até {cooldown_until}")
                self._rotate_to_next_key()
                attempts += 1
                continue
            elif cooldown_until:
                # Cooldown expirou
                del self.key_cooldowns[key]

            # Verifica flags
            if not stats['credits_exhausted'] and not stats['rate_limited']:
                return key

            self._rotate_to_next_key()
            attempts += 1

        # Todas problemáticas: escolhe a com menos erros
        best_key = min(self.api_keys, key=lambda k: self.key_usage_stats[k]['errors'])
        logger.warning(f"⚠️ Todas as chaves têm problemas, usando a com menos erros: {best_key[:10]}...")
        return best_key

    def _rotate_to_next_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    def _handle_api_error(self, key: str, error: Exception):
        """Trata erros de API e atualiza estatísticas"""
        stats = self.key_usage_stats[key]
        stats['errors'] += 1
        error_str = str(error).lower()

        # 402 - créditos insuficientes
        if '402' in error_str or 'payment required' in error_str or 'credit' in error_str:
            stats['credits_exhausted'] = True
            # Cooldown longo até haver intervenção
            self.key_cooldowns[key] = datetime.now() + timedelta(minutes=30)
            logger.warning(f"💳 Chave {key[:10]}... sem créditos (cooldown 30min)")

        # 429 - rate limit
        elif '429' in error_str or 'rate limit' in error_str:
            stats['rate_limited'] = True
            self.key_cooldowns[key] = datetime.now() + timedelta(seconds=60)
            logger.warning(f"⏱️ Chave {key[:10]}... em rate limit (cooldown 60s)")

        # Outros erros
        else:
            self.key_cooldowns[key] = datetime.now() + timedelta(seconds=30)
            logger.warning(f"⚠️ Erro na chave {key[:10]}...: {error}")

    def _create_client(self, api_key: str) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=api_key)

    def _maybe_adjust_request(self, kwargs: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Reduz max_tokens progressivamente e troca modelo conforme tentativas"""
        new_kwargs = dict(kwargs)
        # Reduz max_tokens (exponencial inverso): 1º=orig, 2º=2048, 3º=1024, 4º=512
        max_tokens = new_kwargs.get('max_tokens')
        if max_tokens:
            if attempt == 1:
                new_kwargs['max_tokens'] = min(max_tokens, 2048)
            elif attempt == 2:
                new_kwargs['max_tokens'] = min(max_tokens, 1024)
            elif attempt >= 3:
                new_kwargs['max_tokens'] = min(max_tokens, 512)
        # Troca modelo conforme fallback chain
        model = new_kwargs.get('model')
        if model in self.model_fallback_chain:
            idx = self.model_fallback_chain.index(model)
            if attempt <= 2 and idx + 1 < len(self.model_fallback_chain):
                new_kwargs['model'] = self.model_fallback_chain[idx + 1]
        else:
            # Se modelo não está na chain, força para o primeiro na 2ª tentativa
            if attempt >= 1 and self.model_fallback_chain:
                new_kwargs['model'] = self.model_fallback_chain[0]
        return new_kwargs

    def chat_completion(self, **kwargs) -> Any:
        """Executa chat completion com rotação de chaves (interface síncrona)."""
        last_error: Optional[Exception] = None
        # Até max_retries, com ajuste progressivo de kwargs
        for attempt in range(self.max_retries):
            try:
                api_key = self._get_next_available_key()
                client = self._create_client(api_key)

                # Ajustes de tentativa
                adj_kwargs = self._maybe_adjust_request(kwargs, attempt)

                stats = self.key_usage_stats[api_key]
                stats['requests'] += 1
                stats['last_used'] = datetime.now()

                logger.debug(f"🔑 Usando chave {api_key[:10]}... (tentativa {attempt + 1}) | modelo={adj_kwargs.get('model')} max_tokens={adj_kwargs.get('max_tokens')}")

                response = client.chat.completions.create(**adj_kwargs)

                # Rotaciona para distribuir carga
                self._rotate_to_next_key()
                return response

            except Exception as e:
                last_error = e
                self._handle_api_error(api_key, e)
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"⏳ Aguardando {delay:.1f}s antes da próxima tentativa...")
                    time.sleep(delay)
                continue

        logger.error(f"❌ Todas as tentativas falharam. Último erro: {last_error}")
        raise last_error

    async def chat_completion_async(self, **kwargs) -> Any:
        """Interface assíncrona compatível com asyncio."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat_completion(**kwargs))

    def get_api_status(self) -> Dict[str, Any]:
        status = {
            'total_keys': len(self.api_keys),
            'current_key_index': self.current_key_index,
            'keys_status': []
        }
        for i, key in enumerate(self.api_keys):
            stats = self.key_usage_stats[key]
            key_status = {
                'index': i,
                'key_preview': f"{key[:10]}...{key[-4:]}",
                'requests': stats['requests'],
                'errors': stats['errors'],
                'last_used': stats['last_used'].isoformat() if stats['last_used'] else None,
                'credits_exhausted': stats['credits_exhausted'],
                'rate_limited': stats['rate_limited'],
                'in_cooldown': key in self.key_cooldowns,
                'available': not stats['credits_exhausted'] and not stats['rate_limited'] and key not in self.key_cooldowns
            }
            if key in self.key_cooldowns:
                key_status['cooldown_until'] = self.key_cooldowns[key].isoformat()
            status['keys_status'].append(key_status)

        total_requests = sum(stats['requests'] for stats in self.key_usage_stats.values())
        total_errors = sum(stats['errors'] for stats in self.key_usage_stats.values())
        available_keys = sum(1 for k in status['keys_status'] if k['available'])

        status['summary'] = {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': (total_errors / total_requests * 100) if total_requests > 0 else 0,
            'available_keys': available_keys,
            'health_status': 'healthy' if available_keys > 0 else 'degraded'
        }
        return status

    def reset_key_status(self, key_index: Optional[int] = None):
        if key_index is not None:
            if 0 <= key_index < len(self.api_keys):
                key = self.api_keys[key_index]
                self.key_usage_stats[key].update({'credits_exhausted': False, 'rate_limited': False})
                if key in self.key_cooldowns:
                    del self.key_cooldowns[key]
                logger.info(f"✅ Status da chave {key[:10]}... resetado")
            else:
                logger.error(f"❌ Índice de chave inválido: {key_index}")
        else:
            for key in self.api_keys:
                self.key_usage_stats[key].update({'credits_exhausted': False, 'rate_limited': False})
            self.key_cooldowns.clear()
            logger.info("✅ Status de todas as chaves resetado")

# Instância global
openrouter_manager = OpenRouterAPIManager()


