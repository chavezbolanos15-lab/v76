#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v4.0 - Document Analysis Engine
Motor de análise de documentos com IA especializada (Qwen + Predictive Analytics)
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import mimetypes
import pandas as pd
from PIL import Image
import pytesseract
import docx
import PyPDF2
import openpyxl

# Imports condicionais
try:
    from services.enhanced_ai_manager import enhanced_ai_manager
    HAS_AI_MANAGER = True
except ImportError:
    HAS_AI_MANAGER = False

try:
    from engine.predictive_analytics_engine import PredictiveAnalyticsEngine
    HAS_PREDICTIVE = True
except ImportError:
    HAS_PREDICTIVE = False

from services.auto_save_manager import salvar_etapa, salvar_erro

logger = logging.getLogger(__name__)

class DocumentAnalysisEngine:
    """Motor de análise profunda de documentos com IA especializada"""

    def __init__(self):
        """Inicializa o motor de análise"""
        self.supported_formats = {
            'application/json': self._process_json,
            'text/markdown': self._process_markdown,
            'text/plain': self._process_text,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/msword': self._process_doc,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_xlsx,
            'application/vnd.ms-excel': self._process_xls,
            'text/csv': self._process_csv,
            'image/png': self._process_image,
            'image/jpeg': self._process_image,
            'image/jpg': self._process_image,
            'application/pdf': self._process_pdf
        }

        # Inicializa IA especializada
        self.ai_manager = enhanced_ai_manager if HAS_AI_MANAGER else None
        self.predictive_engine = PredictiveAnalyticsEngine() if HAS_PREDICTIVE else None

        # Configurações de análise
        self.analysis_config = {
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'max_files_per_session': 20,
            'deep_analysis_threshold': 1000,  # caracteres mínimos para análise profunda
            'qwen_model': 'qwen/qwen2.5-vl-32b-instruct:free',
            'analysis_depth': 'ultra_deep'
        }

        logger.info("📚 Document Analysis Engine inicializado com IA especializada")

    async def analyze_uploaded_documents(
        self,
        session_id: str,
        uploaded_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analisa documentos carregados com IA especializada
        
        Args:
            session_id: ID da sessão
            uploaded_files: Lista de arquivos carregados
        """
        logger.info(f"📚 Iniciando análise profunda de {len(uploaded_files)} documentos")

        analysis_results = {
            'session_id': session_id,
            'analysis_started': datetime.now().isoformat(),
            'total_files': len(uploaded_files),
            'processed_files': 0,
            'failed_files': 0,
            'extracted_content': {},
            'ai_analysis': {},
            'predictive_insights': {},
            'unified_knowledge_base': {},
            'document_relationships': {},
            'key_insights': [],
            'expert_synthesis': {}
        }

        try:
            # FASE 1: Extração de Conteúdo
            logger.info("📄 FASE 1: Extraindo conteúdo dos documentos")
            extraction_results = await self._extract_all_documents(uploaded_files, session_id)
            analysis_results['extracted_content'] = extraction_results

            # FASE 2: Análise com Qwen (OpenRouter)
            logger.info("🤖 FASE 2: Análise profunda com Qwen")
            if self.ai_manager:
                qwen_analysis = await self._analyze_with_qwen(extraction_results, session_id)
                analysis_results['ai_analysis'] = qwen_analysis
            else:
                logger.warning("⚠️ AI Manager não disponível")

            # FASE 3: Análise Preditiva
            logger.info("🔮 FASE 3: Análise preditiva especializada")
            if self.predictive_engine:
                predictive_analysis = await self._analyze_with_predictive_engine(
                    extraction_results, analysis_results.get('ai_analysis', {}), session_id
                )
                analysis_results['predictive_insights'] = predictive_analysis
            else:
                logger.warning("⚠️ Predictive Engine não disponível")

            # FASE 4: Síntese Unificada
            logger.info("🧠 FASE 4: Criando base de conhecimento unificada")
            unified_knowledge = await self._create_unified_knowledge_base(
                extraction_results,
                analysis_results.get('ai_analysis', {}),
                analysis_results.get('predictive_insights', {}),
                session_id
            )
            analysis_results['unified_knowledge_base'] = unified_knowledge

            # FASE 5: Análise de Relacionamentos
            logger.info("🔗 FASE 5: Mapeando relacionamentos entre documentos")
            relationships = self._analyze_document_relationships(extraction_results)
            analysis_results['document_relationships'] = relationships

            # FASE 6: Síntese Final de Expertise
            logger.info("🎯 FASE 6: Síntese final de expertise")
            expert_synthesis = await self._create_expert_synthesis(analysis_results, session_id)
            analysis_results['expert_synthesis'] = expert_synthesis

            # Salva análise completa
            salvar_etapa("document_analysis_complete", analysis_results, categoria="document_analysis")

            logger.info(f"✅ Análise de documentos concluída: {analysis_results['processed_files']} processados")
            return analysis_results

        except Exception as e:
            logger.error(f"❌ Erro na análise de documentos: {e}")
            salvar_erro("document_analysis_error", e, contexto={'session_id': session_id})
            raise

    async def _extract_all_documents(
        self,
        uploaded_files: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """Extrai conteúdo de todos os documentos"""
        
        extraction_results = {
            'total_files': len(uploaded_files),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'documents': {},
            'content_summary': {},
            'file_types': {},
            'total_content_length': 0
        }

        for file_info in uploaded_files:
            try:
                file_path = file_info.get('filepath')
                filename = file_info.get('filename')
                mime_type = file_info.get('mime_type')

                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"⚠️ Arquivo não encontrado: {file_path}")
                    extraction_results['failed_extractions'] += 1
                    continue

                logger.info(f"📄 Extraindo: {filename}")

                # Determina processador baseado no tipo MIME
                processor = self.supported_formats.get(mime_type)
                if not processor:
                    # Tenta determinar pelo nome do arquivo
                    processor = self._get_processor_by_extension(filename)

                if processor:
                    content = processor(file_path)
                    
                    if content and len(content.strip()) > 50:
                        doc_id = f"doc_{len(extraction_results['documents']) + 1}"
                        
                        extraction_results['documents'][doc_id] = {
                            'filename': filename,
                            'filepath': file_path,
                            'mime_type': mime_type,
                            'content': content,
                            'content_length': len(content),
                            'extracted_at': datetime.now().isoformat(),
                            'processor_used': processor.__name__
                        }
                        
                        extraction_results['successful_extractions'] += 1
                        extraction_results['total_content_length'] += len(content)
                        
                        # Atualiza estatísticas por tipo
                        file_type = mime_type.split('/')[-1]
                        extraction_results['file_types'][file_type] = extraction_results['file_types'].get(file_type, 0) + 1
                        
                        logger.info(f"✅ {filename}: {len(content)} caracteres extraídos")
                    else:
                        logger.warning(f"⚠️ Conteúdo insuficiente em {filename}")
                        extraction_results['failed_extractions'] += 1
                else:
                    logger.warning(f"⚠️ Formato não suportado: {mime_type}")
                    extraction_results['failed_extractions'] += 1

            except Exception as e:
                logger.error(f"❌ Erro ao processar {filename}: {e}")
                extraction_results['failed_extractions'] += 1

        # Gera resumo do conteúdo
        extraction_results['content_summary'] = self._generate_content_summary(extraction_results['documents'])

        return extraction_results

    async def _analyze_with_qwen(
        self,
        extraction_results: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Análise profunda com Qwen (OpenRouter)"""
        
        if not self.ai_manager:
            return {'error': 'AI Manager não disponível'}

        try:
            # Prepara contexto consolidado dos documentos
            consolidated_content = self._consolidate_document_content(extraction_results['documents'])
            
            # Prompt especializado para análise de documentos
            analysis_prompt = f"""
# VOCÊ É UM ANALISTA ESPECIALISTA EM DOCUMENTOS EMPRESARIAIS

Analise profundamente os documentos fornecidos e torne-se um EXPERT no conteúdo.

## DOCUMENTOS PARA ANÁLISE:
{consolidated_content[:15000]}

## SUA MISSÃO:
1. **ABSORVER TODO O CONHECIMENTO** dos documentos
2. **IDENTIFICAR PADRÕES E INSIGHTS** únicos
3. **MAPEAR OPORTUNIDADES** específicas
4. **EXTRAIR DADOS CRÍTICOS** para análise de mercado
5. **CRIAR SÍNTESE PERSONALIZADA** baseada nos documentos

## RETORNE JSON ESTRUTURADO:

```json
{{
  "document_expertise": {{
    "knowledge_absorbed": "Resumo do conhecimento absorvido",
    "key_concepts_identified": ["Conceito 1", "Conceito 2", "Conceito 3"],
    "unique_insights": ["Insight único 1", "Insight único 2"],
    "data_points_extracted": ["Dado 1", "Dado 2", "Dado 3"],
    "patterns_discovered": ["Padrão 1", "Padrão 2"],
    "opportunities_mapped": ["Oportunidade 1", "Oportunidade 2"]
  }},
  "market_intelligence": {{
    "target_audience_refined": "Público-alvo refinado baseado nos documentos",
    "pain_points_documented": ["Dor específica 1", "Dor específica 2"],
    "value_propositions": ["Proposta 1", "Proposta 2"],
    "competitive_advantages": ["Vantagem 1", "Vantagem 2"],
    "market_positioning": "Posicionamento sugerido baseado nos dados"
  }},
  "strategic_recommendations": {{
    "immediate_actions": ["Ação imediata 1", "Ação imediata 2"],
    "medium_term_strategy": ["Estratégia 1", "Estratégia 2"],
    "long_term_vision": "Visão de longo prazo baseada nos documentos",
    "risk_mitigation": ["Risco 1 e mitigação", "Risco 2 e mitigação"]
  }},
  "personalization_factors": {{
    "unique_business_context": "Contexto único do negócio",
    "specific_challenges": ["Desafio específico 1", "Desafio específico 2"],
    "custom_solutions": ["Solução customizada 1", "Solução customizada 2"],
    "differentiation_opportunities": ["Diferenciação 1", "Diferenciação 2"]
  }},
  "expertise_level": {{
    "document_comprehension": "Alto/Médio/Baixo",
    "knowledge_depth": "Profundo/Moderado/Superficial",
    "analysis_confidence": "95%/80%/60%",
    "ready_for_module_generation": true/false
  }}
}}
```

IMPORTANTE: Seja específico, personalizado e baseado EXCLUSIVAMENTE nos documentos fornecidos.
"""

            # Executa análise com Qwen
            qwen_response = await self.ai_manager.generate_text(analysis_prompt)
            
            # Processa resposta
            qwen_analysis = self._process_qwen_response(qwen_response)
            
            # Salva análise do Qwen
            salvar_etapa("qwen_document_analysis", qwen_analysis, categoria="document_analysis")
            
            return qwen_analysis

        except Exception as e:
            logger.error(f"❌ Erro na análise com Qwen: {e}")
            return {'error': str(e)}

    async def _analyze_with_predictive_engine(
        self,
        extraction_results: Dict[str, Any],
        qwen_analysis: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Análise preditiva especializada dos documentos"""
        
        if not self.predictive_engine:
            return {'error': 'Predictive Engine não disponível'}

        try:
            # Prepara dados para análise preditiva
            predictive_data = {
                'documents_content': extraction_results.get('documents', {}),
                'qwen_insights': qwen_analysis.get('document_expertise', {}),
                'market_intelligence': qwen_analysis.get('market_intelligence', {}),
                'session_id': session_id
            }

            # Executa análise preditiva
            predictive_results = await self.predictive_engine.analyze_documents_predictively(predictive_data)
            
            # Salva análise preditiva
            salvar_etapa("predictive_document_analysis", predictive_results, categoria="document_analysis")
            
            return predictive_results

        except Exception as e:
            logger.error(f"❌ Erro na análise preditiva: {e}")
            return {'error': str(e)}

    async def _create_unified_knowledge_base(
        self,
        extraction_results: Dict[str, Any],
        qwen_analysis: Dict[str, Any],
        predictive_insights: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Cria base de conhecimento unificada"""
        
        try:
            # Consolida todo o conhecimento
            unified_knowledge = {
                'session_id': session_id,
                'knowledge_sources': {
                    'documents_processed': len(extraction_results.get('documents', {})),
                    'total_content_analyzed': extraction_results.get('total_content_length', 0),
                    'ai_analysis_completed': bool(qwen_analysis.get('document_expertise')),
                    'predictive_analysis_completed': bool(predictive_insights.get('predictions'))
                },
                'consolidated_insights': [],
                'expert_knowledge': {},
                'personalized_context': {},
                'analysis_readiness': False
            }

            # Consolida insights únicos
            all_insights = []
            
            # Insights do Qwen
            if qwen_analysis.get('document_expertise', {}).get('unique_insights'):
                all_insights.extend(qwen_analysis['document_expertise']['unique_insights'])
            
            # Insights preditivos
            if predictive_insights.get('key_insights'):
                all_insights.extend(predictive_insights['key_insights'])
            
            # Remove duplicatas e consolida
            unified_knowledge['consolidated_insights'] = list(set(all_insights))

            # Cria conhecimento especializado
            unified_knowledge['expert_knowledge'] = {
                'domain_expertise': qwen_analysis.get('document_expertise', {}).get('knowledge_absorbed', ''),
                'market_intelligence': qwen_analysis.get('market_intelligence', {}),
                'predictive_trends': predictive_insights.get('trend_analysis', {}),
                'strategic_framework': qwen_analysis.get('strategic_recommendations', {}),
                'risk_assessment': predictive_insights.get('risk_analysis', {})
            }

            # Contexto personalizado
            unified_knowledge['personalized_context'] = {
                'business_uniqueness': qwen_analysis.get('personalization_factors', {}).get('unique_business_context', ''),
                'specific_challenges': qwen_analysis.get('personalization_factors', {}).get('specific_challenges', []),
                'custom_solutions': qwen_analysis.get('personalization_factors', {}).get('custom_solutions', []),
                'differentiation_factors': qwen_analysis.get('personalization_factors', {}).get('differentiation_opportunities', [])
            }

            # Verifica se está pronto para geração de módulos
            expertise_level = qwen_analysis.get('expertise_level', {})
            unified_knowledge['analysis_readiness'] = (
                expertise_level.get('ready_for_module_generation', False) and
                len(unified_knowledge['consolidated_insights']) >= 5 and
                bool(unified_knowledge['expert_knowledge']['domain_expertise'])
            )

            # Salva base de conhecimento
            salvar_etapa("unified_knowledge_base", unified_knowledge, categoria="document_analysis")

            return unified_knowledge

        except Exception as e:
            logger.error(f"❌ Erro ao criar base de conhecimento: {e}")
            return {'error': str(e)}

    async def _create_expert_synthesis(
        self,
        analysis_results: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Cria síntese final de expertise para uso nos módulos"""
        
        try:
            # Prepara prompt para síntese final
            synthesis_prompt = f"""
# SÍNTESE FINAL DE EXPERTISE - PREPARAÇÃO PARA MÓDULOS

Baseado na análise completa dos documentos, crie uma síntese de expertise que será usada para gerar módulos personalizados e únicos.

## CONHECIMENTO ABSORVIDO:
{json.dumps(analysis_results.get('unified_knowledge_base', {}), indent=2, ensure_ascii=False)[:10000]}

## ANÁLISE QWEN:
{json.dumps(analysis_results.get('ai_analysis', {}), indent=2, ensure_ascii=False)[:5000]}

## INSIGHTS PREDITIVOS:
{json.dumps(analysis_results.get('predictive_insights', {}), indent=2, ensure_ascii=False)[:5000]}

## CRIE SÍNTESE PARA MÓDULOS PERSONALIZADOS:

```json
{{
  "expert_context_for_modules": {{
    "business_dna": "DNA único do negócio extraído dos documentos",
    "market_positioning_unique": "Posicionamento único baseado nos dados",
    "audience_profile_specific": "Perfil específico do público baseado nos documentos",
    "value_proposition_refined": "Proposta de valor refinada",
    "competitive_differentiation": "Diferenciação competitiva específica"
  }},
  "personalization_directives": {{
    "avoid_generic_content": ["Lista de clichês a evitar"],
    "emphasize_unique_aspects": ["Aspectos únicos a enfatizar"],
    "custom_language_style": "Estilo de linguagem específico",
    "industry_specific_terms": ["Termos específicos da indústria"],
    "brand_voice_guidelines": "Diretrizes de voz da marca"
  }},
  "module_generation_context": {{
    "avatar_customization_data": "Dados para personalizar avatar",
    "drivers_customization_data": "Dados para personalizar drivers mentais",
    "objection_handling_data": "Dados específicos para anti-objeção",
    "proof_points_data": "Pontos de prova específicos",
    "market_analysis_data": "Dados específicos para análise de mercado"
  }},
  "quality_assurance": {{
    "uniqueness_score": "90-100%",
    "personalization_level": "Alto/Médio/Baixo",
    "data_richness": "Rico/Moderado/Limitado",
    "analysis_depth": "Profundo/Moderado/Superficial"
  }}
}}
```

CRÍTICO: Esta síntese será usada para gerar módulos únicos e personalizados. Seja específico e evite generalidades.
"""

            if self.ai_manager:
                synthesis_response = await self.ai_manager.generate_text(synthesis_prompt)
                expert_synthesis = self._process_synthesis_response(synthesis_response)
            else:
                expert_synthesis = self._create_fallback_synthesis(analysis_results)

            # Salva síntese de expertise
            salvar_etapa("expert_synthesis", expert_synthesis, categoria="document_analysis")

            return expert_synthesis

        except Exception as e:
            logger.error(f"❌ Erro na síntese de expertise: {e}")
            return {'error': str(e)}

    def _consolidate_document_content(self, documents: Dict[str, Any]) -> str:
        """Consolida conteúdo de todos os documentos"""
        
        consolidated = "# DOCUMENTOS ANALISADOS\n\n"
        
        for doc_id, doc_data in documents.items():
            consolidated += f"## {doc_data['filename']}\n\n"
            consolidated += f"**Tipo:** {doc_data['mime_type']}\n"
            consolidated += f"**Tamanho:** {doc_data['content_length']} caracteres\n\n"
            consolidated += f"**Conteúdo:**\n{doc_data['content'][:3000]}\n\n"
            consolidated += "---\n\n"

        return consolidated

    def _process_qwen_response(self, response: str) -> Dict[str, Any]:
        """Processa resposta do Qwen"""
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                json_text = response[start:end].strip()
                return json.loads(json_text)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            return {
                'document_expertise': {
                    'knowledge_absorbed': response[:2000],
                    'analysis_method': 'text_extraction'
                }
            }

    def _process_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Processa resposta da síntese"""
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                json_text = response[start:end].strip()
                return json.loads(json_text)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            return {
                'expert_context_for_modules': {
                    'business_dna': response[:1000],
                    'synthesis_method': 'text_extraction'
                }
            }

    # Processadores de arquivo específicos
    def _process_json(self, filepath: str) -> str:
        """Processa arquivo JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao processar JSON: {e}")
            return ""

    def _process_markdown(self, filepath: str) -> str:
        """Processa arquivo Markdown"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Erro ao processar Markdown: {e}")
            return ""

    def _process_text(self, filepath: str) -> str:
        """Processa arquivo de texto"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Erro ao processar texto: {e}")
            return ""

    def _process_docx(self, filepath: str) -> str:
        """Processa arquivo DOCX"""
        try:
            doc = docx.Document(filepath)
            content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content.append(paragraph.text)
            return '\n'.join(content)
        except Exception as e:
            logger.error(f"Erro ao processar DOCX: {e}")
            return ""

    def _process_doc(self, filepath: str) -> str:
        """Processa arquivo DOC (fallback para texto)"""
        return self._process_text(filepath)

    def _process_xlsx(self, filepath: str) -> str:
        """Processa arquivo Excel XLSX"""
        try:
            df = pd.read_excel(filepath, sheet_name=None)
            content = []
            for sheet_name, sheet_df in df.items():
                content.append(f"PLANILHA: {sheet_name}")
                content.append(sheet_df.to_string(index=False))
                content.append("")
            return '\n'.join(content)
        except Exception as e:
            logger.error(f"Erro ao processar XLSX: {e}")
            return ""

    def _process_xls(self, filepath: str) -> str:
        """Processa arquivo Excel XLS"""
        return self._process_xlsx(filepath)

    def _process_csv(self, filepath: str) -> str:
        """Processa arquivo CSV"""
        try:
            df = pd.read_csv(filepath)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Erro ao processar CSV: {e}")
            return ""

    def _process_image(self, filepath: str) -> str:
        """Processa imagem com OCR"""
        try:
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image, lang='por')
            return text.strip()
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {e}")
            return ""

    def _process_pdf(self, filepath: str) -> str:
        """Processa arquivo PDF"""
        try:
            content = []
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        content.append(text)
            return '\n'.join(content)
        except Exception as e:
            logger.error(f"Erro ao processar PDF: {e}")
            return ""

    def _get_processor_by_extension(self, filename: str) -> Optional[callable]:
        """Determina processador pela extensão do arquivo"""
        ext = Path(filename).suffix.lower()
        
        extension_map = {
            '.json': self._process_json,
            '.md': self._process_markdown,
            '.txt': self._process_text,
            '.docx': self._process_docx,
            '.doc': self._process_doc,
            '.xlsx': self._process_xlsx,
            '.xls': self._process_xls,
            '.csv': self._process_csv,
            '.png': self._process_image,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.pdf': self._process_pdf
        }
        
        return extension_map.get(ext)

    def _generate_content_summary(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """Gera resumo do conteúdo extraído"""
        
        summary = {
            'total_documents': len(documents),
            'total_characters': 0,
            'total_words': 0,
            'file_types': {},
            'content_categories': {},
            'key_topics': []
        }

        for doc_data in documents.values():
            content = doc_data['content']
            summary['total_characters'] += len(content)
            summary['total_words'] += len(content.split())
            
            # Categoriza por tipo
            file_type = doc_data['mime_type'].split('/')[-1]
            summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1

        return summary

    def _analyze_document_relationships(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa relacionamentos entre documentos"""
        
        documents = extraction_results.get('documents', {})
        relationships = {
            'content_overlap': {},
            'thematic_connections': {},
            'data_consistency': {},
            'complementary_information': {}
        }

        # Análise básica de sobreposição de conteúdo
        doc_ids = list(documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                content1 = documents[doc_id1]['content'].lower()
                content2 = documents[doc_id2]['content'].lower()
                
                # Calcula palavras em comum
                words1 = set(content1.split())
                words2 = set(content2.split())
                common_words = words1.intersection(words2)
                
                if len(common_words) > 10:  # Threshold para relacionamento
                    relationship_key = f"{doc_id1}_{doc_id2}"
                    relationships['content_overlap'][relationship_key] = {
                        'common_words_count': len(common_words),
                        'similarity_score': len(common_words) / len(words1.union(words2)),
                        'doc1': documents[doc_id1]['filename'],
                        'doc2': documents[doc_id2]['filename']
                    }

        return relationships

    def _create_fallback_synthesis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria síntese de fallback quando IA não está disponível"""
        
        return {
            'expert_context_for_modules': {
                'business_dna': 'Análise baseada em documentos carregados',
                'market_positioning_unique': 'Posicionamento específico do negócio',
                'audience_profile_specific': 'Perfil de público baseado nos dados',
                'value_proposition_refined': 'Proposta de valor refinada',
                'competitive_differentiation': 'Diferenciação competitiva identificada'
            },
            'personalization_directives': {
                'avoid_generic_content': ['Evitar conteúdo genérico'],
                'emphasize_unique_aspects': ['Enfatizar aspectos únicos'],
                'custom_language_style': 'Estilo personalizado',
                'industry_specific_terms': ['Termos específicos da indústria']
            },
            'quality_assurance': {
                'uniqueness_score': '80%',
                'personalization_level': 'Alto',
                'data_richness': 'Rico',
                'analysis_depth': 'Profundo'
            },
            'fallback_mode': True
        }

    def get_analysis_status(self, session_id: str) -> Dict[str, Any]:
        """Retorna status da análise de documentos"""
        
        try:
            # Verifica se existe análise salva
            session_dir = Path(f"analyses_data/{session_id}")
            
            status = {
                'session_id': session_id,
                'analysis_exists': False,
                'documents_processed': 0,
                'analysis_complete': False,
                'ready_for_modules': False
            }

            if session_dir.exists():
                # Verifica arquivos de análise
                if (session_dir / "document_analysis_complete.json").exists():
                    status['analysis_exists'] = True
                    status['analysis_complete'] = True
                    
                    # Carrega dados da análise
                    with open(session_dir / "document_analysis_complete.json", 'r') as f:
                        analysis_data = json.load(f)
                        status['documents_processed'] = analysis_data.get('data', {}).get('processed_files', 0)
                        status['ready_for_modules'] = analysis_data.get('data', {}).get('unified_knowledge_base', {}).get('analysis_readiness', False)

            return status

        except Exception as e:
            logger.error(f"❌ Erro ao verificar status: {e}")
            return {'error': str(e)}

# Instância global
document_analysis_engine = DocumentAnalysisEngine()