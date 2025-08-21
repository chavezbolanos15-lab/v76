#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.1 - Analisador de IA para Documentos
Integração dos modelos qwen e predictive_analytics_engine com tratamento de erros aprimorado.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
from .openrouter_api_manager import openrouter_manager

logger = logging.getLogger(__name__)

class DocumentAIAnalyzer:
    """Analisador de IA especializado em documentos"""
    
    def __init__(self):
        self.openrouter_manager = openrouter_manager
        
        # Configurações dos modelos
        self.qwen_model = "qwen/qwen-2.5-72b-instruct"
        self.predictive_model = "anthropic/claude-3.5-sonnet"
        
        # Diretório de uploads
        self.upload_folder = os.path.join(os.getcwd(), 'uploads', 'documents')
        os.makedirs(self.upload_folder, exist_ok=True)
        
    async def analyze_documents(self, session_id: str, extracted_content: List[Dict]) -> Dict:
        """Análise principal dos documentos"""
        try:
            logger.info(f"Iniciando análise de documentos para sessão {session_id}")
            
            individual_analyses = []
            synthesis_result = {}
            insights = {}
            analysis_status = 'completed'
            error_message = None

            # Fase 1: Análise individual com Qwen
            try:
                individual_analyses = await self._analyze_individual_documents(extracted_content)
            except Exception as e:
                logger.error(f"Erro na fase de análise individual: {e}")
                analysis_status = 'partial_error'
                error_message = f"Análise individual falhou: {e}"

            # Fase 2: Síntese e correlação com Predictive Analytics
            if analysis_status == 'completed' or (analysis_status == 'partial_error' and individual_analyses):
                try:
                    synthesis_result = await self._synthesize_and_correlate(individual_analyses)
                    if 'error' in synthesis_result:
                        analysis_status = 'partial_error'
                        error_message = synthesis_result['error']
                except Exception as e:
                    logger.error(f"Erro na fase de síntese: {e}")
                    analysis_status = 'partial_error'
                    error_message = f"Síntese falhou: {e}"

            # Fase 3: Geração de insights e recomendações
            if analysis_status == 'completed' or (analysis_status == 'partial_error' and synthesis_result and 'error' not in synthesis_result):
                try:
                    insights = await self._generate_insights(synthesis_result, extracted_content)
                    if 'error' in insights:
                        analysis_status = 'partial_error'
                        error_message = insights['error']
                except Exception as e:
                    logger.error(f"Erro na fase de insights: {e}")
                    analysis_status = 'partial_error'
                    error_message = f"Geração de insights falhou: {e}"
            
            # Salva resultados
            results = {
                'session_id': session_id,
                'analysis_completed_at': datetime.now().isoformat(),
                'individual_analyses': individual_analyses,
                'synthesis': synthesis_result,
                'insights': insights,
                'document_count': len(extracted_content),
                'analysis_summary': self._create_analysis_summary(individual_analyses, synthesis_result, insights),
                'status': analysis_status,
                'error_message': error_message
            }
            
            self._save_analysis_results(session_id, results)
            self._update_session_status(session_id, analysis_status, 100)
            
            return {
                'success': True,
                'analysis_id': session_id,
                'status': analysis_status,
                'error': error_message
            }
            
        except Exception as e:
            logger.error(f"Erro crítico na análise de documentos: {e}")
            self._update_session_status(session_id, 'error', 0)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _analyze_individual_documents(self, extracted_content: List[Dict]) -> List[Dict]:
        """Análise individual de cada documento com Qwen"""
        analyses = []
        
        for i, doc in enumerate(extracted_content):
            try:
                logger.info(f"Analisando documento {i+1}/{len(extracted_content)}: {doc['filename']}")
                
                prompt = self._create_document_analysis_prompt(doc)
                
                response = await self.openrouter_manager.chat_completion_async(
                    model=self.qwen_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """Você é um especialista em análise de documentos com 20 anos de experiência. 
                            Sua tarefa é analisar profundamente o conteúdo fornecido, extraindo insights valiosos, 
                            padrões, tendências e informações estratégicas. Seja detalhado, preciso e objetivo."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                
                analysis_text = response.choices[0].message.content
                
                structured_analysis = self._structure_analysis(analysis_text, doc)
                
                analyses.append({
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'analysis': structured_analysis,
                    'raw_analysis': analysis_text,
                    'analyzed_at': datetime.now().isoformat(),
                    'model_used': self.qwen_model
                })
                
            except Exception as e:
                logger.error(f"Erro ao analisar {doc['filename']}: {e}")
                analyses.append({
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'analysis': {'error': str(e)},
                    'analyzed_at': datetime.now().isoformat()
                })
        
        return analyses
    
    async def _synthesize_and_correlate(self, individual_analyses: List[Dict]) -> Dict:
        """Síntese e correlação usando Predictive Analytics Engine"""
        try:
            logger.info("Iniciando síntese e correlação com Predictive Analytics")
            
            synthesis_prompt = self._create_synthesis_prompt(individual_analyses)
            
            response = await self.openrouter_manager.chat_completion_async(
                model=self.predictive_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um Predictive Analytics Engine especializado em correlação de dados e síntese estratégica.
                        Sua função é identificar padrões ocultos, correlações entre documentos, tendências emergentes e 
                        fazer previsões baseadas nos dados analisados. Seja analítico, estratégico e preditivo."""
                    },
                    {
                        "role": "user",
                        "content": synthesis_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=6000
            )
            
            synthesis_text = response.choices[0].message.content
            
            return {
                'synthesis_text': synthesis_text,
                'correlations': self._extract_correlations(synthesis_text),
                'patterns': self._extract_patterns(synthesis_text),
                'predictions': self._extract_predictions(synthesis_text),
                'synthesized_at': datetime.now().isoformat(),
                'model_used': self.predictive_model
            }
            
        except Exception as e:
            logger.error(f"Erro na síntese: {e}")
            # Verifica se o erro é 402 e retorna uma mensagem específica
            if "402" in str(e) or "credit" in str(e).lower() or "payment required" in str(e).lower():
                return {'error': "Erro 402: Créditos insuficientes na API do OpenRouter para síntese. Por favor, recarregue seus créditos ou reduza o tamanho da requisição."}
            return {'error': str(e)}
    
    async def _generate_insights(self, synthesis_result: Dict, extracted_content: List[Dict]) -> Dict:
        """Geração de insights finais e recomendações"""
        try:
            logger.info("Gerando insights finais")
            
            insights_prompt = self._create_insights_prompt(synthesis_result, extracted_content)
            
            response = await self.openrouter_manager.chat_completion_async(
                model=self.qwen_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um consultor estratégico sênior especializado em transformar análises em 
                        insights acionáveis. Sua tarefa é criar recomendações práticas, identificar oportunidades e 
                        riscos, e fornecer um roadmap estratégico baseado nos dados analisados."""
                    },
                    {
                        "role": "user",
                        "content": insights_prompt
                    }
                ],
                temperature=0.4,
                max_tokens=5000
            )
            
            insights_text = response.choices[0].message.content
            
            return {
                'insights_text': insights_text,
                'key_insights': self._extract_key_insights(insights_text),
                'recommendations': self._extract_recommendations(insights_text),
                'opportunities': self._extract_opportunities(insights_text),
                'risks': self._extract_risks(insights_text),
                'action_items': self._extract_action_items(insights_text),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de insights: {e}")
            # Verifica se o erro é 402 e retorna uma mensagem específica
            if "402" in str(e) or "credit" in str(e).lower() or "payment required" in str(e).lower():
                return {'error': "Erro 402: Créditos insuficientes na API do OpenRouter para geração de insights. Por favor, recarregue seus créditos ou reduza o tamanho da requisição."}
            return {'error': str(e)}
    
    def _create_document_analysis_prompt(self, doc: Dict) -> str:
        """Cria prompt especializado para análise do documento"""
        base_prompt = f"""
        DOCUMENTO PARA ANÁLISE:
        Arquivo: {doc['filename']}
        Tipo: {doc['file_type']}
        Metadados: {json.dumps(doc.get('metadata', {}), indent=2)}
        
        CONTEÚDO:
        {doc['content'][:8000]}  # Limita para evitar overflow
        
        INSTRUÇÕES DE ANÁLISE:
        1. Identifique o propósito e contexto do documento
        2. Extraia informações-chave, dados importantes e métricas
        3. Identifique padrões, tendências e insights
        4. Avalie a qualidade e confiabilidade das informações
        5. Identifique lacunas ou informações faltantes
        6. Sugira como este documento se relaciona com análise de mercado
        7. Extraia qualquer informação sobre público-alvo, concorrentes ou oportunidades
        
        Forneça uma análise estruturada e detalhada.
        """
        
        # Personaliza baseado no tipo de arquivo
        if doc['filename'].endswith('.json'):
            base_prompt += "\nFOCO ESPECIAL: Analise a estrutura de dados, APIs, configurações ou dados estruturados."
        elif doc['filename'].endswith(('.csv', '.xlsx')):
            base_prompt += "\nFOCO ESPECIAL: Analise os dados quantitativos, tendências numéricas e correlações estatísticas."
        elif doc['filename'].endswith('.md'):
            base_prompt += "\nFOCO ESPECIAL: Analise a documentação, processos descritos e informações técnicas."
        elif doc['filename'].endswith(('.png', '.jpg')):
            base_prompt += "\nFOCO ESPECIAL: Descreva o conteúdo visual e sua relevância para análise de mercado."
        
        return base_prompt
    
    def _create_synthesis_prompt(self, individual_analyses: List[Dict]) -> str:
        """Cria prompt para síntese e correlação"""
        analyses_summary = []
        for analysis in individual_analyses:
            if 'error' not in analysis.get('analysis', {}):
                analyses_summary.append({
                    'filename': analysis['filename'],
                    'type': analysis['file_type'],
                    'key_points': analysis.get('analysis', {}).get('summary', analysis.get('raw_analysis', '')[:1000])
                })
        
        return f"""
        ANÁLISES INDIVIDUAIS PARA SÍNTESE:
        {json.dumps(analyses_summary, indent=2, ensure_ascii=False)}
        
        TAREFA DE SÍNTESE E CORRELAÇÃO:
        1. Identifique correlações entre os documentos analisados
        2. Encontre padrões comuns e divergências
        3. Sintetize insights que emergem da combinação dos documentos
        4. Identifique tendências e previsões baseadas no conjunto de dados
        5. Avalie a consistência e complementaridade das informações
        6. Identifique oportunidades de mercado baseadas na análise conjunta
        7. Faça previsões sobre cenários futuros baseados nos dados
        8. Identifique riscos e desafios potenciais
        
        Forneça uma síntese estratégica e preditiva detalhada.
        """
    
    def _create_insights_prompt(self, synthesis_result: Dict, extracted_content: List[Dict]) -> str:
        """Cria prompt para geração de insights finais"""
        return f"""
        SÍNTESE ESTRATÉGICA:
        {synthesis_result.get('synthesis_text', '')}
        
        DOCUMENTOS ANALISADOS:
        Total: {len(extracted_content)} documentos
        Tipos: {list(set(doc['file_type'] for doc in extracted_content))}
        
        TAREFA DE GERAÇÃO DE INSIGHTS:
        1. Transforme a análise em insights acionáveis
        2. Crie recomendações estratégicas específicas
        3. Identifique oportunidades de negócio concretas
        4. Liste riscos e como mitigá-los
        5. Sugira próximos passos e ações prioritárias
        6. Crie um roadmap estratégico baseado nos achados
        7. Identifique KPIs e métricas para acompanhamento
        8. Sugira como integrar estes insights com análise de mercado
        
        Forneça insights práticos, específicos e acionáveis.
        """
    
    def _structure_analysis(self, analysis_text: str, doc: Dict) -> Dict:
        """Estrutura a análise em formato padronizado"""
        return {
            'summary': analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            'full_analysis': analysis_text,
            'document_type': doc['file_type'],
            'key_findings': self._extract_key_findings(analysis_text),
            'confidence_score': self._calculate_confidence_score(doc, analysis_text)
        }
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extrai principais achados do texto de análise"""
        lines = text.split('\n')
        findings = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['importante', 'chave', 'principal', 'destaque', 'insight']):
                findings.append(line.strip())
        return findings[:5]
    
    def _extract_correlations(self, text: str) -> List[str]:
        """Extrai correlações identificadas"""
        lines = text.split('\n')
        correlations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['correlação', 'relação', 'conexão', 'padrão']):
                correlations.append(line.strip())
        return correlations[:3]
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extrai padrões identificados"""
        lines = text.split('\n')
        patterns = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['padrão', 'tendência', 'comportamento', 'recorrente']):
                patterns.append(line.strip())
        return patterns[:3]
    
    def _extract_predictions(self, text: str) -> List[str]:
        """Extrai previsões feitas"""
        lines = text.split('\n')
        predictions = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['previsão', 'futuro', 'projeção', 'expectativa']):
                predictions.append(line.strip())
        return predictions[:3]
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extrai insights principais"""
        lines = text.split('\n')
        insights = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight', 'descoberta', 'revelação', 'conclusão']):
                insights.append(line.strip())
        return insights[:5]
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extrai recomendações"""
        lines = text.split('\n')
        recommendations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recomend', 'sugest', 'deve', 'deveria']):
                recommendations.append(line.strip())
        return recommendations[:5]
    
    def _extract_opportunities(self, text: str) -> List[str]:
        """Extrai oportunidades identificadas"""
        lines = text.split('\n')
        opportunities = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['oportunidade', 'potencial', 'chance', 'possibilidade']):
                opportunities.append(line.strip())
        return opportunities[:3]
    
    def _extract_risks(self, text: str) -> List[str]:
        """Extrai riscos identificados"""
        lines = text.split('\n')
        risks = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['risco', 'ameaça', 'desafio', 'vulnerabilidade']):
                risks.append(line.strip())
        return risks[:3]
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extrai itens de ação"""
        lines = text.split('\n')
        action_items = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['ação', 'próximo passo', 'implementar', 'executar']):
                action_items.append(line.strip())
        return action_items[:5]

    def _calculate_confidence_score(self, doc: Dict, analysis_text: str) -> float:
        """Calcula um score de confiança baseado na análise"""
        # Exemplo simplificado: baseado no tamanho da análise e tipo de documento
        score = 0.0
        if len(analysis_text) > 1000: # Análise mais longa = mais confiança
            score += 0.3
        if doc['file_type'] in ['pdf', 'docx', 'xlsx']: # Documentos estruturados = mais confiança
            score += 0.2
        if 'error' not in doc.get('analysis', {}): # Sem erros na análise individual
            score += 0.5
        return min(1.0, score) # Limita a 1.0

    def _create_analysis_summary(self, individual_analyses: List[Dict], synthesis_result: Dict, insights: Dict) -> str:
        """Cria um resumo geral da análise"""
        summary_parts = []
        summary_parts.append(f"Análise concluída para {len(individual_analyses)} documentos.")
        
        if synthesis_result and 'synthesis_text' in synthesis_result:
            summary_parts.append("Síntese e correlação realizadas com sucesso.")
            summary_parts.append(f"Principais correlações: {', '.join(synthesis_result.get('correlations', [])[:2])}...")
        elif synthesis_result and 'error' in synthesis_result:
            summary_parts.append(f"Síntese falhou: {synthesis_result['error']}")
        else:
            summary_parts.append("Síntese não foi realizada.")

        if insights and 'insights_text' in insights:
            summary_parts.append("Insights e recomendações gerados.")
            summary_parts.append(f"Principais insights: {', '.join(insights.get('key_insights', [])[:2])}...")
        elif insights and 'error' in insights:
            summary_parts.append(f"Geração de insights falhou: {insights['error']}")
        else:
            summary_parts.append("Geração de insights não foi realizada.")

        return " ".join(summary_parts)

    def _save_analysis_results(self, session_id: str, results: Dict):
        """Salva os resultados da análise em um arquivo JSON"""
        output_dir = os.path.join(os.getcwd(), 'analyses_data', 'results')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"analysis_results_{session_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ Resultados da análise salvos em: {file_path}")

    def _update_session_status(self, session_id: str, status: str, progress: int):
        """Atualiza o status da sessão (simulado)"""
        logger.info(f"Status da sessão {session_id} atualizado: {status} (Progresso: {progress}%) ")

# Instância global
document_ai_analyzer = DocumentAIAnalyzer()


