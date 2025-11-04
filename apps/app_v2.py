import streamlit as st
import pandas as pd
import os
import json
import hashlib
import mimetypes
import base64
import io
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import zipfile
import shutil
from pathlib import Path
import requests
from PIL import Image
import re
import numpy as np
from collections import Counter
import jieba
import jieba.analyse
try:
    import fitz  # PyMuPDF for PDF preview
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# AI功能相关库
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # 如果transformers不可用，我们使用其他方法

# Set page config with premium aesthetics
st.set_page_config(
    page_title="Agribusiness Expert AI Cloud",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean and modern CSS styling
st.markdown("""
<style>
    /* Overall Layout */
    .main {
        background: #f8fafc;
        color: #1e293b;
    }
    
    /* Title Styles */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Button Styles */
    .stButton>button {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton>button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* File Card Styles */
    .file-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .file-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Metric Card Styles */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Preview Section Styles */
    .preview-section {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    
    /* File Icon Styles */
    .file-icon {
        font-size: 24px;
        margin-right: 12px;
    }
    
    /* Action Button Styles */
    .action-btn {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #64748b;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        margin: 0 2px;
    }
    
    .action-btn:hover {
        background: #e2e8f0;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

class CloudStorageManager:
    def __init__(self):
        # 云部署配置
        import os
        self.is_cloud_deployment = os.getenv('STREAMLIT_SERVER_PORT') is not None
        
        if self.is_cloud_deployment:
            # 云部署：使用持久化存储
            self.storage_dir = Path("/tmp/cloud_storage")
            self.cache_dir = Path("/tmp/local_cache")
            self.ai_analysis_dir = Path("/tmp/ai_analysis")
        else:
            # 本地部署：使用当前目录
            self.storage_dir = Path("cloud_storage")
            self.cache_dir = Path("local_cache")
            self.ai_analysis_dir = Path("ai_analysis")
        
        # 创建目录
        self.storage_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.ai_analysis_dir.mkdir(exist_ok=True)
        
        self.db_path = self.storage_dir / "storage.db"
        self.init_database()
        
        # 初始化AI功能
        self.init_ai_models()
        
        # 天气缓存
        self.latest_weather: Optional[Dict[str, Any]] = None
        # 遥感缓存
        self.latest_remote_sensing: Optional[Dict[str, Any]] = None
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                folder_id INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                is_cached BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (folder_id) REFERENCES folders (id)
            )
        ''')
        
        # 文件夹表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT NOT NULL,
                parent_folder_id INTEGER,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_folder_id) REFERENCES folders (id)
            )
        ''')
        
        # 上传进度表（用于断点续传）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upload_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                total_size INTEGER,
                uploaded_size INTEGER,
                chunk_size INTEGER,
                checksum TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # AI分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                analysis_type TEXT,
                industry_category TEXT,
                extracted_text TEXT,
                key_phrases TEXT,
                summary TEXT,
                confidence_score REAL,
                method TEXT,
                analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES files (id)
            )
        ''')

        # 迁移：若旧表无 method 列则补充
        try:
            cursor.execute("PRAGMA table_info(ai_analysis)")
            cols = [row[1] for row in cursor.fetchall()]
            if 'method' not in cols:
                cursor.execute('ALTER TABLE ai_analysis ADD COLUMN method TEXT')
        except Exception:
            pass
        
        # 行业分类表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS industry_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE,
                keywords TEXT,
                description TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_ai_models(self):
        """初始化AI模型"""
        # 初始化行业分类关键词（Agribusiness细分，补充非洲常见作物/要素）
        self.industry_keywords = {
            "种植业": ["作物", "玉米", "小米", "高粱", "水稻", "木薯", "山药", "红薯", "花生", "芝麻", "葵花籽", "棉花", "可可", "咖啡", "茶叶", "香蕉", "芒果", "菠萝", "蔬菜", "果园", "产量", "单产", "公顷", "亩", "播种", "收获", "灌溉", "病虫害", "除草", "密度"],
            "畜牧业": ["生猪", "牛羊", "家禽", "奶牛", "出栏", "存栏", "饲料", "日龄", "增重", "料肉比", "免疫", "兽药", "疫病", "繁育", "犊牛", "屠宰"],
            "农资与土壤": ["肥料", "氮肥", "磷肥", "钾肥", "配方施肥", "有机质", "pH", "土壤盐分", "微量元素", "保水", "覆盖", "深松", "秸秆还田"],
            "农业金融": ["采购", "成本", "贷款", "保单", "保险", "赔付", "保费", "授信", "现金流", "应收", "应付", "利润", "毛利率", "价格", "期货"],
            "供应链与仓储": ["冷链", "仓储", "物流", "运输", "库容", "损耗", "周转", "交付", "订单", "批次", "追溯"],
            "气候与遥感": ["降雨", "降水", "温度", "积温", "蒸散", "干旱", "NDVI", "EVI", "卫星", "遥感", "气象站", "辐射", "沙漠蝗虫", "草地贪夜蛾"],
            "农业物联网": ["传感器", "湿度", "含水率", "EC", "阈值", "阀门", "泵站", "滴灌", "喷灌", "自动化", "报警"]
        }
        
        # 初始化OCR模型
        self.ocr_reader = None
        self.ocr_loading = False
        if OCR_AVAILABLE:
            try:
                # 异步加载OCR模型，避免阻塞界面
                st.info("🔄 Loading OCR model, please wait...")
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])
                st.success("✅ OCR model loaded successfully")
            except Exception as e:
                st.warning(f"⚠️ OCR model loading failed: {str(e)}")
                st.info("💡 Please click '🔄 Reload AI' to retry later")
        
        # 初始化文本分类模型
        self.text_classifier = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # 使用中文BERT模型进行文本分类
                self.text_classifier = pipeline(
                    "text-classification",
                    model="bert-base-chinese",
                    tokenizer="bert-base-chinese"
                )
                st.success("✅ BERT text classification model loaded successfully")
            except Exception as e:
                # Downgrade to info to avoid noisy toast; rules/ML will fallback
                st.info(f"BERT model loading failed, fallback will be used")
        else:
            st.info("ℹ️ Transformers library not installed, using machine learning classification")
        
        # 初始化摘要生成模型
        self.summarizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # 使用T5模型进行摘要生成
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    tokenizer="t5-small"
                )
                st.success("✅ T5 summarization model loaded successfully")
            except Exception as e:
                st.info("T5 summarization not available, using smart rules")
        else:
            st.info("ℹ️ Using smart summarization algorithm")
        
        # 初始化机器学习分类器
        self.ml_classifier = None
        self.ml_trained = False
        if ML_AVAILABLE:
            try:
                # 使用朴素贝叶斯分类器
                self.ml_classifier = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
                    ('classifier', MultinomialNB())
                ])
                # 自动初始化预训练分类器
                if self.init_pretrained_classifier():
                    st.success("✅ Pre-trained machine learning classifier loaded successfully")
                else:
                    st.info("Pre-trained ML classifier unavailable, using keyword matching")
            except Exception as e:
                st.info("ML classifier init failed, using keyword matching")
        else:
            st.info("ℹ️ 使用关键词匹配分类")
        
        # 初始化默认行业分类
        self.init_default_categories()

    def fetch_weather_summary(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """从 Open-Meteo 获取未来7天的气象摘要（无需API密钥）"""
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                "&hourly=temperature_2m,precipitation"
                "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
                "&forecast_days=7&timezone=auto"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            result = {
                "location": {"lat": latitude, "lon": longitude},
                "precipitation_sum": daily.get("precipitation_sum", []),
                "tmax": daily.get("temperature_2m_max", []),
                "tmin": daily.get("temperature_2m_min", []),
                "dates": daily.get("time", [])
            }
            # 简要统计
            try:
                total_rain = float(sum(x for x in result["precipitation_sum"] if isinstance(x, (int, float))))
            except Exception:
                total_rain = 0.0
            result["summary"] = {
                "7d_total_rain_mm": round(total_rain, 1),
                "avg_tmax": round(sum(result["tmax"]) / max(1, len(result["tmax"])), 1) if result["tmax"] else None,
                "avg_tmin": round(sum(result["tmin"]) / max(1, len(result["tmin"])), 1) if result["tmin"] else None,
            }
            self.latest_weather = result
            return {"success": True, "weather": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compute_remote_sensing_stub(self, latitude: float, longitude: float, days: int = 30) -> Dict[str, Any]:
        """遥感指数占位：生成近days天的NDVI/EVI简易时序（无需外部服务）。"""
        try:
            import math
            base_date = datetime.now()
            dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1)]
            ndvi = []
            evi = []
            for i in range(days):
                # 生成平滑的波动数据，范围做物理合理约束
                v = 0.5 + 0.3 * math.sin(i/6.0) + 0.1 * math.sin(i/2.5)
                ndvi.append(round(max(0.0, min(0.9, v)), 3))
                e = 0.4 + 0.25 * math.sin(i/7.0 + 0.5)
                evi.append(round(max(0.0, min(0.8, e)), 3))
            summary = {
                "ndvi_avg": round(sum(ndvi)/len(ndvi), 3) if ndvi else None,
                "evi_avg": round(sum(evi)/len(evi), 3) if evi else None,
                "ndvi_last": ndvi[-1] if ndvi else None,
                "evi_last": evi[-1] if evi else None,
            }
            result = {
                "location": {"lat": latitude, "lon": longitude},
                "dates": dates,
                "ndvi": ndvi,
                "evi": evi,
                "summary": summary,
            }
            self.latest_remote_sensing = result
            return {"success": True, "remote_sensing": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_agri_structured_fields(self, text: str) -> Dict[str, Any]:
        """农业报表模板抽取（规则版占位）：作物、面积、日期、施肥/灌溉/用药/单产等。"""
        if not text:
            return {}
        import re
        fields: Dict[str, Any] = {}
        try:
            # 作物
            m = re.search(r'(作物|品种|作物名称)[：:，]\s*([\u4e00-\u9fffA-Za-z0-9]+)', text)
            if m: fields['作物'] = m.group(2)
            # 面积（亩/公顷/ha）
            m = re.search(r'(面积|播种面积|收获面积)[：:，]\s*([\d,.]+)\s*(亩|公顷|ha)', text)
            if m: fields['面积'] = f"{m.group(2)} {m.group(3)}"
            # 日期（简单识别 年-月-日 或 年/月/日 或 中文）
            m = re.search(r'(日期|时间|记录时间)[：:，]\s*(\d{4}[-年/]\d{1,2}[-月/]\d{1,2})', text)
            if m: fields['日期'] = m.group(2)
            # 施肥
            m = re.search(r'(施肥|肥料|配方施肥)[：:，]?\s*([\u4e00-\u9fffA-Za-z0-9]+)?\s*([\d,.]+)\s*(kg|公斤|斤)', text)
            if m: fields['施肥'] = f"{(m.group(2) or '').strip()} {m.group(3)} {m.group(4)}".strip()
            # 灌溉
            m = re.search(r'(灌溉|浇水)[：:，]?\s*([\d,.]+)\s*(mm|立方|m3|方)', text)
            if m: fields['灌溉'] = f"{m.group(2)} {m.group(3)}"
            # 用药
            m = re.search(r'(农药|用药|防治)[：:，]?\s*([\u4e00-\u9fffA-Za-z0-9]+)\s*([\d,.]+)\s*(ml|毫升|L|升|kg|克|g)', text)
            if m: fields['用药'] = f"{m.group(2)} {m.group(3)} {m.group(4)}"
            # 单产/产量
            m = re.search(r'(单产|亩产)[：:，]\s*([\d,.]+)\s*(斤/亩|公斤/亩|kg/ha|t/ha)', text)
            if m: fields['单产'] = f"{m.group(2)} {m.group(3)}"
            m = re.search(r'(总产|产量)[：:，]\s*([\d,.]+)\s*(kg|吨|t)', text)
            if m: fields['产量'] = f"{m.group(2)} {m.group(3)}"
        except Exception:
            pass
        return fields
    
    def init_default_categories(self):
        """初始化默认行业分类"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, keywords in self.industry_keywords.items():
            cursor.execute('''
                INSERT OR IGNORE INTO industry_categories (category_name, keywords, description)
                VALUES (?, ?, ?)
            ''', (category, json.dumps(keywords, ensure_ascii=False), f"{category}相关文档"))
        
        conn.commit()
        conn.close()

    def _to_english_category(self, category: str) -> str:
        mapping = {
            "种植业": "Planting",
            "畜牧业": "Livestock",
            "农资与土壤": "Inputs-Soil",
            "农业金融": "Agri-Finance",
            "供应链与仓储": "SupplyChain-Storage",
            "气候与遥感": "Climate-RemoteSensing",
            "农业物联网": "Agri-IoT",
        }
        return mapping.get(category, category)
    
    def generate_smart_report(self, file_id: int) -> Dict[str, Any]:
        """生成智能报告和图表"""
        try:
            # 获取文件信息
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT file_path, file_type, filename FROM files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {"success": False, "error": "文件不存在"}
            
            file_path, file_type, filename = result
            
            # 提取文本内容
            text = self.extract_text_from_file(file_id)
            if not text:
                return {"success": False, "error": "无法提取文本内容"}
            
            # 分析文档结构
            analysis = self.analyze_document_structure(text)
            analysis["full_text"] = text
            
            # 提取数据点
            data_points = self.extract_data_points(text)
            
            # 生成图表
            charts = self.generate_charts(data_points)
            
            # 生成报告
            report = self.create_smart_report(analysis, charts, filename)
            
            return {
                "success": True,
                "analysis": analysis,
                "data_points": data_points,
                "charts": charts,
                "report": report
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """分析文档结构，识别农业领域文档类型与要素"""
        analysis = {
            "document_type": "未知",
            "data_types": [],
            "key_metrics": [],
            "time_periods": [],
            "categories": [],
            "confidence": 0.0
        }
        
        # 识别农业文档类型
        if any(k in text for k in ["单产", "亩产", "t/ha", "kg/ha", "播种面积", "收获面积", "产量"]):
            analysis["document_type"] = "种植业生产报告"
            analysis["data_types"].extend(["面积", "产量", "单产", "趋势"])
        elif any(k in text for k in ["出栏", "存栏", "增重", "日增重", "料肉比", "免疫"]):
            analysis["document_type"] = "畜牧业生产报告"
            analysis["data_types"].extend(["头数", "重量", "转换率", "免疫"])
        elif any(k in text for k in ["降雨", "降水", "mm", "积温", "干旱", "NDVI", "遥感"]):
            analysis["document_type"] = "气候与遥感监测"
            analysis["data_types"].extend(["降雨", "温度", "指数", "时间序列"])
        elif any(k in text for k in ["成本", "采购", "价格", "保险", "赔付", "利润", "毛利率"]):
            analysis["document_type"] = "农业财务/供应链报告"
            analysis["data_types"].extend(["金额", "比率", "对比", "价格趋势"])
        
        # 提取关键指标
        import re
        # 查找数字模式（支持带单位）
        numbers = re.findall(r'[\d,]+\.?\d*\s*(?:t/ha|kg/ha|kg|t|吨|公斤|元/斤|元/吨|mm)?', text)
        analysis["key_metrics"] = numbers[:10]  # 取前10个数字
        
        # 查找时间模式
        time_patterns = re.findall(r'\d{4}年|\d{1,2}月|\d{1,2}日|Q[1-4]', text)
        analysis["time_periods"] = list(set(time_patterns))
        
        # 查找分类信息
        category_patterns = re.findall(r'[A-Za-z\u4e00-\u9fff]+[：:]\s*[\d,]+', text)
        analysis["categories"] = category_patterns[:5]
        
        # 计算置信度（农业场景稍微提高关键指标权重）
        confidence = min(len(analysis["key_metrics"]) * 0.12 + 
                        len(analysis["time_periods"]) * 0.18 + 
                        len(analysis["categories"]) * 0.1, 1.0)
        analysis["confidence"] = confidence
        
        return analysis
    
    def extract_data_points(self, text: str) -> List[Dict[str, Any]]:
        """提取数据点用于生成图表（增强农业单位识别）"""
        data_points = []
        
        import re
        
        # 提取数值和标签
        patterns = [
            r'([A-Za-z\u4e00-\u9fff]+)[：:]\s*([\d,]+\.?\d*)\s*(t/ha|kg/ha|kg|t|吨|公斤|mm|%)?',
            r'([A-Za-z\u4e00-\u9fff]+)\s*([\d,]+\.?\d*)\s*(%)',
            r'([A-Za-z\u4e00-\u9fff]+)\s*为\s*([\d,]+\.?\d*)\s*(t/ha|kg/ha|kg|t|吨|公斤|mm|%)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    label, value, unit = match
                else:
                    label, value = match
                    unit = None
                try:
                    # 清理数值
                    clean_value = float(value.replace(',', ''))
                    if clean_value > 0:  # 只保留正数
                        data_points.append({
                            "label": label.strip(),
                            "value": clean_value,
                            "type": unit or "数值"
                        })
                except ValueError:
                    continue
        
        # 去重并排序
        seen = set()
        unique_points = []
        for point in data_points:
            key = point["label"]
            if key not in seen:
                seen.add(key)
                unique_points.append(point)
        
        # 按数值排序
        unique_points.sort(key=lambda x: x["value"], reverse=True)
        
        return unique_points[:10]  # 返回前10个数据点
    
    def generate_charts(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成图表数据"""
        charts = []
        
        if not data_points:
            return charts
        
        # 生成柱状图数据
        if len(data_points) >= 2:
            bar_chart = {
                "type": "bar",
                "title": "数据对比柱状图",
                "data": {
                    "labels": [point["label"] for point in data_points[:8]],
                    "values": [point["value"] for point in data_points[:8]]
                }
            }
            charts.append(bar_chart)
        
        # 生成饼图数据（前5个）
        if len(data_points) >= 3:
            pie_data = data_points[:5]
            total = sum(point["value"] for point in pie_data)
            pie_chart = {
                "type": "pie",
                "title": "数据分布饼图",
                "data": {
                    "labels": [point["label"] for point in pie_data],
                    "values": [point["value"] for point in pie_data],
                    "percentages": [round(point["value"]/total*100, 1) for point in pie_data]
                }
            }
            charts.append(pie_chart)
        
        # 生成趋势图（如果有时间数据）
        if len(data_points) >= 4:
            line_chart = {
                "type": "line",
                "title": "数据趋势图",
                "data": {
                    "labels": [point["label"] for point in data_points[:6]],
                    "values": [point["value"] for point in data_points[:6]]
                }
            }
            charts.append(line_chart)
        
        return charts
    
    def create_smart_report(self, analysis: Dict, charts: List[Dict], filename: str) -> str:
        """生成智能报告（加入农业洞察与KPI）"""
        report = f"# 📊 Agribusiness Smart Analysis Report\n\n"
        report += f"**File name**: {filename}\n\n"
        report += f"**Document type**: {analysis['document_type']}\n\n"
        report += f"**Confidence**: {analysis['confidence']:.1%}\n\n"
        
        # 农业KPI（从全文智能提取）
        agrikpis = self.compute_agribusiness_kpis(analysis.get('full_text', '')) if isinstance(analysis, dict) else {}
        if agrikpis:
            report += "## 🌾 Agribusiness KPIs\n\n"
            for k, v in agrikpis.items():
                report += f"- {k}: {v}\n"
            report += "\n"

        # 天气摘要（如果已获取）
        if getattr(self, 'latest_weather', None):
            ws = self.latest_weather.get('summary', {})
            report += "## ☁️ Climate summary (next 7 days)\n\n"
            if ws:
                if ws.get('7d_total_rain_mm') is not None:
                    report += f"- Total rainfall: {ws['7d_total_rain_mm']} mm\n"
                if ws.get('avg_tmax') is not None:
                    report += f"- Avg Tmax: {ws['avg_tmax']} °C\n"
                if ws.get('avg_tmin') is not None:
                    report += f"- Avg Tmin: {ws['avg_tmin']} °C\n"
            report += "\n"

        # 遥感摘要（如果已获取）
        if getattr(self, 'latest_remote_sensing', None):
            rs = self.latest_remote_sensing.get('summary', {})
            report += "## 🛰️ Remote sensing summary\n\n"
            if rs:
                if rs.get('ndvi_avg') is not None:
                    report += f"- NDVI average: {rs['ndvi_avg']}\n"
                if rs.get('evi_avg') is not None:
                    report += f"- EVI average: {rs['evi_avg']}\n"
                if rs.get('ndvi_last') is not None:
                    report += f"- Latest NDVI: {rs['ndvi_last']}\n"
                if rs.get('evi_last') is not None:
                    report += f"- Latest EVI: {rs['evi_last']}\n"
            report += "\n"

        # 模板抽取结果
        structured = self.extract_agri_structured_fields(analysis.get('full_text', '')) if isinstance(analysis, dict) else {}
        if structured:
            report += "## 🗂️ Structured fields (template extraction)\n\n"
            for k, v in structured.items():
                report += f"- {k}: {v}\n"
            report += "\n"
        
        # Key metrics
        if analysis['key_metrics']:
            report += "## 🔢 Key metrics\n\n"
            for i, metric in enumerate(analysis['key_metrics'][:5], 1):
                report += f"{i}. {metric}\n"
            report += "\n"

        # Time periods
        if analysis['time_periods']:
            report += "## 📅 Time periods\n\n"
            report += f"Detected time info: {', '.join(analysis['time_periods'])}\n\n"

        # Categories
        if analysis['categories']:
            report += "## 📋 Categories\n\n"
            for category in analysis['categories']:
                report += f"- {category}\n"
            report += "\n"

        # Visualization notes
        if charts:
            report += "## 📈 Data visualization\n\n"
            for chart in charts:
                report += f"### {chart['title']}\n\n"
                if chart['type'] == 'bar':
                    report += "Bar chart shows value comparison across categories to spot highs and lows.\n\n"
                elif chart['type'] == 'pie':
                    report += "Pie chart shows proportion distribution for intuitive share comparison.\n\n"
                elif chart['type'] == 'line':
                    report += "Line chart shows temporal trends to identify growth or decline patterns.\n\n"

        # Suggestions
        report += "## 💡 Suggestions\n\n"
        if analysis['document_type'] in ["种植业生产报告", "畜牧业生产报告"]:
            report += "- Track trends of key KPIs (yield, rainfall, FCR).\n"
            report += "- Compare fields/lots or herds to find outliers.\n"
            report += "- Plan interventions (fertigation, pest control) based on thresholds.\n"
        elif analysis['document_type'] in ["农业财务/供应链报告"]:
            report += "- Monitor margins and price trends.\n"
            report += "- Optimize cost structure and inventory turnover.\n"
            report += "- Manage risk with insurance/hedging where applicable.\n"
        else:
            report += "- Keep data updated regularly.\n"
            report += "- Focus on KPI trends and anomalies.\n"
            report += "- Apply data-driven decisions.\n"
        
        return report

    def compute_agribusiness_kpis(self, text: str) -> Dict[str, Any]:
        """基于规则快速提取农业常见KPI（轻量占位，可后续换模型）"""
        if not text:
            return {}
        import re
        kpis: Dict[str, Any] = {}
        try:
            # 单产（支持 kg/ha, t/ha, 亩产）
            m = re.search(r'(单产|亩产)[:：]?\s*([\d,.]+)\s*(kg/ha|t/ha|公斤/亩|斤/亩|吨/公顷)?', text)
            if m:
                kpis['单产'] = f"{m.group(2)} {m.group(3) or ''}".strip()

            # 面积（亩、公顷）
            m = re.search(r'(播种面积|收获面积|面积)[:：]?\s*([\d,.]+)\s*(亩|公顷|ha)', text)
            if m:
                kpis['面积'] = f"{m.group(2)} {m.group(3)}"

            # 降雨量（mm）
            m = re.search(r'(降雨|降水|累计降雨|累计降水)[:：]?\s*([\d,.]+)\s*mm', text)
            if m:
                kpis['累计降雨'] = f"{m.group(2)} mm"

            # 成本与利润
            m = re.search(r'(总成本|成本)[:：]?\s*([\d,.]+)', text)
            if m:
                kpis['成本'] = m.group(2)
            m = re.search(r'(利润|毛利|毛利率)[:：]?\s*([\d,.]+)\s*(%)?', text)
            if m:
                kpis['利润/毛利'] = f"{m.group(2)}{m.group(3) or ''}"

            # 畜牧关键指标
            m = re.search(r'(出栏|存栏)[:：]?\s*([\d,.]+)\s*(头|只)?', text)
            if m:
                kpis[m.group(1)] = f"{m.group(2)} {m.group(3) or ''}".strip()
            m = re.search(r'(料肉比|FCR)[:：]?\s*([\d,.]+)', text)
            if m:
                kpis['料肉比'] = m.group(2)

            # 遥感指数
            m = re.search(r'(NDVI|EVI)[:：]?\s*([\d,.]+)', text)
            if m:
                kpis[m.group(1)] = m.group(2)
        except Exception:
            pass
        return kpis
    
    def calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_file_type(self, filename: str) -> str:
        """获取文件类型"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type.split('/')[0]
        return 'unknown'
    
    def upload_file(self, uploaded_file, folder_id: Optional[int] = None) -> Dict[str, Any]:
        """上传文件"""
        try:
            # 生成唯一文件名
            timestamp = int(time.time())
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = self.storage_dir / filename
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 计算文件信息
            file_size = file_path.stat().st_size
            checksum = self.calculate_checksum(str(file_path))
            file_type = self.get_file_type(uploaded_file.name)
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (filename, file_path, file_size, file_type, folder_id, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (uploaded_file.name, str(file_path), file_size, file_type, folder_id, checksum))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "filename": uploaded_file.name,
                "file_size": file_size,
                "file_type": file_type
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_files(self, folder_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取文件列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if folder_id is None:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id IS NULL
                ORDER BY upload_time DESC
            ''')
        else:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files WHERE folder_id = ?
                ORDER BY upload_time DESC
            ''', (folder_id,))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "id": row[0],
                "filename": row[1],
                "file_size": row[2],
                "file_type": row[3],
                "upload_time": row[4],
                "is_cached": bool(row[5])
            })
        
        conn.close()
        return files
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[int] = None) -> Dict[str, Any]:
        """创建文件夹"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO folders (folder_name, parent_folder_id)
                VALUES (?, ?)
            ''', (folder_name, parent_folder_id))
            conn.commit()
            folder_id = cursor.lastrowid
            conn.close()
            
            return {"success": True, "folder_id": folder_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_files(self, query: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if file_type:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files 
                WHERE filename LIKE ? AND file_type = ?
                ORDER BY upload_time DESC
            ''', (f"%{query}%", file_type))
        else:
            cursor.execute('''
                SELECT id, filename, file_size, file_type, upload_time, is_cached
                FROM files 
                WHERE filename LIKE ?
                ORDER BY upload_time DESC
            ''', (f"%{query}%",))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "id": row[0],
                "filename": row[1],
                "file_size": row[2],
                "file_type": row[3],
                "upload_time": row[4],
                "is_cached": bool(row[5])
            })
        
        conn.close()
        return files
    
    def preview_file(self, file_id: int) -> Optional[bytes]:
        """预览文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, file_type FROM files WHERE id = ?', (file_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        file_path, file_type = result
        
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except:
            return None
    
    def cache_file(self, file_id: int) -> bool:
        """缓存文件到本地"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT file_path, filename FROM files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if result:
                file_path, filename = result
                cache_path = self.cache_dir / filename
                shutil.copy2(file_path, cache_path)
                
                # 更新数据库
                cursor.execute('UPDATE files SET is_cached = TRUE WHERE id = ?', (file_id,))
                conn.commit()
                conn.close()
                return True
        except:
            pass
        return False
    
    def format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def get_file_icon(self, file_type: str) -> str:
        """获取文件类型图标"""
        icons = {
            'image': '🖼️',
            'application': '📄',
            'text': '📝',
            'video': '🎥',
            'audio': '🎵',
            'unknown': '📁'
        }
        return icons.get(file_type, '📁')
    
    def upload_file_with_resume(self, uploaded_file, folder_id: Optional[int] = None, chunk_size: int = 1024*1024) -> Dict[str, Any]:
        """带断点续传的文件上传"""
        try:
            filename = uploaded_file.name
            file_size = len(uploaded_file.getbuffer())
            
            # 检查是否有未完成的上传
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, uploaded_size, checksum FROM upload_progress 
                WHERE filename = ? AND total_size = ?
                ORDER BY upload_time DESC LIMIT 1
            ''', (filename, file_size))
            
            progress_record = cursor.fetchone()
            
            if progress_record:
                # 断点续传
                progress_id, uploaded_size, stored_checksum = progress_record
                st.info(f"🔄 Resumable upload found, continue from {uploaded_size} bytes...")
            else:
                # 新上传
                uploaded_size = 0
                progress_id = None
                stored_checksum = None
            
            # 分块上传
            uploaded_file.seek(uploaded_size)
            current_size = uploaded_size
            
            progress_bar = st.progress(uploaded_size / file_size)
            status_text = st.empty()
            
            while current_size < file_size:
                chunk = uploaded_file.read(min(chunk_size, file_size - current_size))
                if not chunk:
                    break
                
                # 这里应该将chunk发送到服务器
                # 为了演示，我们直接写入本地文件
                temp_file_path = self.storage_dir / f"temp_{filename}"
                with open(temp_file_path, "ab") as f:
                    f.write(chunk)
                
                current_size += len(chunk)
                progress = current_size / file_size
                progress_bar.progress(progress)
                status_text.text(f"Uploading: {current_size}/{file_size} bytes ({progress*100:.1f}%)")
                
                # 更新进度到数据库
                if progress_id:
                    cursor.execute('''
                        UPDATE upload_progress 
                        SET uploaded_size = ? 
                        WHERE id = ?
                    ''', (current_size, progress_id))
                else:
                    cursor.execute('''
                        INSERT INTO upload_progress (filename, total_size, uploaded_size, chunk_size, checksum)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (filename, file_size, current_size, chunk_size, stored_checksum))
                    progress_id = cursor.lastrowid
                
                conn.commit()
                
                # 模拟网络延迟
                time.sleep(0.1)
            
            # 上传完成，移动文件到最终位置
            final_file_path = self.storage_dir / f"{int(time.time())}_{filename}"
            shutil.move(str(temp_file_path), str(final_file_path))
            
            # 计算校验和
            checksum = self.calculate_checksum(str(final_file_path))
            file_type = self.get_file_type(filename)
            
            # 保存文件信息到数据库
            cursor.execute('''
                INSERT INTO files (filename, file_path, file_size, file_type, folder_id, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, str(final_file_path), file_size, file_type, folder_id, checksum))
            
            # 删除进度记录
            if progress_id:
                cursor.execute('DELETE FROM upload_progress WHERE id = ?', (progress_id,))
            
            conn.commit()
            conn.close()
            
            progress_bar.empty()
            status_text.empty()
            
            return {
                "success": True,
                "filename": filename,
                "file_size": file_size,
                "file_type": file_type,
                "checksum": checksum
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_upload_progress(self) -> List[Dict[str, Any]]:
        """获取上传进度列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, total_size, uploaded_size, upload_time
            FROM upload_progress
            ORDER BY upload_time DESC
        ''')
        
        progress_list = []
        for row in cursor.fetchall():
            filename, total_size, uploaded_size, upload_time = row
            progress_list.append({
                "filename": filename,
                "total_size": total_size,
                "uploaded_size": uploaded_size,
                "progress": uploaded_size / total_size if total_size > 0 else 0,
                "upload_time": upload_time
            })
        
        conn.close()
        return progress_list
    
    def resume_upload(self, filename: str) -> Dict[str, Any]:
        """恢复上传"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, total_size, uploaded_size, chunk_size, checksum
            FROM upload_progress 
            WHERE filename = ?
            ORDER BY upload_time DESC LIMIT 1
        ''', (filename,))
        
        result = cursor.fetchone()
        if result:
            progress_id, total_size, uploaded_size, chunk_size, checksum = result
            return {
                "success": True,
                "progress_id": progress_id,
                "total_size": total_size,
                "uploaded_size": uploaded_size,
                "chunk_size": chunk_size,
                "checksum": checksum
            }
        else:
            return {"success": False, "error": "未找到上传进度记录"}
    
    def cancel_upload(self, filename: str) -> bool:
        """取消上传"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM upload_progress WHERE filename = ?', (filename,))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    # ==================== AI功能方法 ====================
    
    def extract_text_from_file(self, file_id: int) -> str:
        """从文件中提取文本内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, file_type, filename FROM files WHERE id = ?', (file_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return ""
        
        file_path, file_type, filename = result
        extracted_text = ""
        
        try:
            if file_type == 'text' or filename.endswith('.txt'):
                # 文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            
            elif file_type == 'application' and filename.endswith('.pdf'):
                # PDF文件
                if PDF_AVAILABLE:
                    doc = fitz.open(file_path)
                    for page in doc:
                        extracted_text += page.get_text()
                    doc.close()
                # 若不可用则保持为空，后续给出友好占位
            
            elif file_type == 'application' and filename.endswith(('.xlsx', '.xls')):
                # Excel文件
                try:
                    df = pd.read_excel(file_path)
                    # 确保DataFrame不为空
                    if not df.empty:
                        # 安全地转换为字符串，避免numpy.str_错误
                        try:
                            extracted_text = df.to_string()
                        except Exception as str_error:
                            # 如果to_string失败，尝试其他方法
                            extracted_text = str(df.values.tolist())
                    else:
                        extracted_text = "Excel file is empty"
                except Exception as e:
                    st.warning(f"Excel reading failed: {str(e)}")
                    extracted_text = ""

            elif filename.endswith('.csv'):
                # CSV文件
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        try:
                            extracted_text = df.to_string()
                        except Exception:
                            extracted_text = str(df.values.tolist())
                    else:
                        extracted_text = "CSV file is empty"
                except Exception as e:
                    st.warning(f"CSV reading failed: {str(e)}")
                    extracted_text = ""

            elif filename.endswith('.docx'):
                # DOCX（可选处理）
                try:
                    import docx  # type: ignore
                    doc = docx.Document(file_path)
                    paras = [p.text for p in doc.paragraphs if p.text]
                    extracted_text = "\n".join(paras)
                except Exception:
                    # 未安装或解析失败则忽略
                    pass
            
            elif file_type == 'image':
                # 图片文件 - OCR识别
                if OCR_AVAILABLE:
                    if self.ocr_reader is None:
                        # 延迟加载OCR模型
                        st.info("🔄 Loading OCR model, please wait...")
                        try:
                            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])
                            st.success("✅ OCR model loaded")
                        except Exception as e:
                            st.error(f"OCR model load failed: {str(e)}")
                            return ""
                    
                    if self.ocr_reader:
                        results = self.ocr_reader.readtext(file_path)
                        extracted_text = ' '.join([result[1] for result in results])
        
        except Exception as e:
            st.error(f"Text extraction failed: {str(e)}")
        
        # 兜底：仍无法提取文本时，返回占位文本，避免AI流程直接失败
        if not extracted_text:
            extracted_text = f"(No extractable text from file: {filename}. Try preview/download.)"
        
        return extracted_text
    
    def classify_industry(self, text: str) -> Dict[str, Any]:
        """使用真正的AI对文档进行行业分类"""
        if not text:
            return {"category": "未分类", "confidence": 0.0, "keywords": []}
        
        # 方法1: 使用BERT模型分类（如果可用）
        if self.text_classifier and len(text) > 10:
            try:
                # 截取文本前512个字符（BERT限制）
                text_sample = text[:512]
                result = self.text_classifier(text_sample)
                
                # 将BERT结果映射到我们的行业分类
                bert_label = result[0]['label']
                bert_confidence = result[0]['score']
                
                # 简单的标签映射（可以根据需要扩展）
                label_mapping = {
                    'LABEL_0': '种植业',
                    'LABEL_1': '畜牧业',
                    'LABEL_2': '农资与土壤',
                    'LABEL_3': '农业金融',
                    'LABEL_4': '供应链与仓储',
                    'LABEL_5': '气候与遥感',
                    'LABEL_6': '农业物联网'
                }
                
                mapped_category = label_mapping.get(bert_label, '未分类')
                
                if mapped_category != '未分类':
                    return {
                        "category": mapped_category,
                        "confidence": bert_confidence,
                        "keywords": self._extract_keywords_from_text(text),
                        "method": "BERT"
                    }
            except Exception as e:
                # Suppress noisy toast; fallback methods will be tried below
                pass
        
        # 方法2: 使用机器学习分类器（如果可用且已训练）
        if self.ml_classifier and self.ml_trained and len(text) > 20:
            try:
                X = [text]
                y_pred = self.ml_classifier.predict(X)
                y_proba = self.ml_classifier.predict_proba(X)
                
                categories = list(self.industry_keywords.keys())
                predicted_category = categories[y_pred[0]]
                confidence = y_proba[0].max()
                
                return {
                    "category": predicted_category,
                    "confidence": confidence,
                    "keywords": self._extract_keywords_from_text(text),
                    "method": "ML"
                }
            except Exception as e:
                # Suppress noisy toast; fallback to rules
                pass
        
        # 方法3: 智能关键词匹配（改进版）
        words = jieba.lcut(text)
        category_scores = {}
        matched_keywords = {}
        
        for category, keywords in self.industry_keywords.items():
            score = 0
            matched = []
            
            # 基础关键词匹配
            for keyword in keywords:
                if keyword in text:
                    score += 1
                    matched.append(keyword)
            
            # 同义词和相似词匹配
            synonyms = self._get_synonyms(category)
            for synonym in synonyms:
                if synonym in text:
                    score += 0.5
                    matched.append(synonym)
            
            # 词频权重
            for keyword in keywords:
                count = text.count(keyword)
                if count > 1:
                    score += count * 0.2
            
            category_scores[category] = score
            matched_keywords[category] = matched
        
        if category_scores and max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            
            # 改进的置信度计算
            total_keywords = len(self.industry_keywords[best_category])
            confidence = min(max_score / (total_keywords * 1.5), 1.0)
            
            # 如果置信度太低，标记为未分类
            if confidence < 0.1:
                return {"category": "未分类", "confidence": 0.0, "keywords": [], "method": "关键词匹配"}
            
            return {
                "category": best_category,
                "confidence": confidence,
                "keywords": matched_keywords[best_category],
                "method": "智能关键词匹配"
            }
        
        return {"category": "未分类", "confidence": 0.0, "keywords": [], "method": "无匹配"}
    
    def _get_synonyms(self, category: str) -> List[str]:
        """获取行业分类的同义词"""
        synonyms_map = {
            "种植业": ["种植", "耕作", "育秧", "移栽", "密植", "病虫害", "施肥", "灌溉", "田间管理", "玉米", "高粱", "小米", "木薯", "花生", "芝麻", "棉花", "可可", "咖啡"],
            "畜牧业": ["养殖", "饲喂", "免疫", "防疫", "繁育", "断奶", "出栏", "存栏", "增重"],
            "农资与土壤": ["配方施肥", "土壤改良", "施用量", "有机肥", "微量元素", "土壤养分"],
            "农业金融": ["贴现", "授信", "保费", "赔付", "承保", "风控", "保单"],
            "供应链与仓储": ["冷链运输", "损耗率", "批次追溯", "库容", "周转率", "分拣"],
            "气候与遥感": ["降雨", "气温", "积温", "干旱指数", "NDVI", "EVI", "遥感", "沙漠蝗虫", "草地贪夜蛾"],
            "农业物联网": ["含水率", "EC", "滴灌", "喷灌", "阀门", "阈值", "报警"]
        }
        return synonyms_map.get(category, [])
    
    def init_pretrained_classifier(self):
        """初始化预训练的分类器"""
        if not self.ml_classifier:
            return False
        
        try:
            # 使用预定义的关键词作为特征进行训练
            X_train = []
            y_train = []
            
            # 为每个行业类别创建训练样本
            for category, keywords in self.industry_keywords.items():
                # 为每个关键词创建训练样本
                for keyword in keywords:
                    # 创建包含关键词的样本文本
                    sample_text = f"这是一个关于{keyword}的文档，涉及{category}领域的内容。"
                    X_train.append(sample_text)
                    y_train.append(category)
                
                # 添加同义词样本
                synonyms = self._get_synonyms(category)
                for synonym in synonyms:
                    sample_text = f"这是一个关于{synonym}的文档，涉及{category}领域的内容。"
                    X_train.append(sample_text)
                    y_train.append(category)
            
            # 训练分类器
            if len(X_train) > 0:
                self.ml_classifier.fit(X_train, y_train)
                self.ml_trained = True
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"初始化预训练分类器失败: {str(e)}")
            return False
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        try:
            # 使用jieba的TF-IDF提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
            return keywords
        except:
            # 简单的关键词提取
            words = jieba.lcut(text)
            word_count = Counter(words)
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            filtered_words = {word: count for word, count in word_count.items() 
                            if len(word) > 1 and word not in stop_words and count > 1}
            return list(dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]).keys())
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键短语"""
        if not text:
            return []
        
        try:
            # 使用jieba的TF-IDF提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
            return keywords
        except:
            # 简单的关键词提取
            words = jieba.lcut(text)
            word_count = Counter(words)
            # 过滤掉单字符和常见停用词
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            filtered_words = {word: count for word, count in word_count.items() 
                            if len(word) > 1 and word not in stop_words and count > 1}
            return list(dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_k]).keys())
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate document summary (model first, fallback to rules)."""
        if not text:
            return "Unable to generate summary"
        
        # 方法1: 使用T5模型生成摘要（如果可用）
        if self.summarizer and len(text) > 50:
            try:
                # 截取文本前1024个字符（T5限制）
                text_sample = text[:1024]
                summary_result = self.summarizer(
                    text_sample,
                    max_length=min(max_length, 150),
                    min_length=30,
                    do_sample=False
                )
                
                if summary_result and len(summary_result) > 0:
                    ai_summary = summary_result[0]['summary_text']
                    return f"🤖 AI Summary: {ai_summary}"
            except Exception as e:
                st.warning(f"T5 summarization failed: {str(e)}")
        
        # 方法2: 使用OpenAI GPT（如果可用）
        if OPENAI_AVAILABLE and len(text) > 100:
            try:
                # 这里需要OpenAI API密钥
                # 暂时跳过，因为需要API密钥
                pass
            except Exception as e:
                st.warning(f"OpenAI summarization failed: {str(e)}")
        
        # 方法3: 智能句子选择（改进的规则方法）
        try:
            # 使用更智能的句子选择
            sentences = re.split(r'[。！？.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
            
            if len(sentences) <= 2:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # 选择最重要的句子（基于长度和关键词）
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = len(sentence)  # 基础分数：句子长度
                
                # 关键词加分
                important_words = ['重要', '关键', '主要', '核心', '总结', '结论', '结果', '发现']
                for word in important_words:
                    if word in sentence:
                        score += 20
                
                # 位置加分（开头和结尾的句子更重要）
                if i < 2 or i >= len(sentences) - 2:
                    score += 10
                
                scored_sentences.append((score, sentence))
            
            # 选择得分最高的2-3个句子
            scored_sentences.sort(reverse=True)
            selected_sentences = [s[1] for s in scored_sentences[:3]]
            
            summary = '。'.join(selected_sentences)
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return f"📝 Smart summary: {summary}"
        except:
            # 方法4: 简单截取（最后备用）
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def analyze_file_with_ai(self, file_id: int) -> Dict[str, Any]:
        """使用AI分析文件"""
        # 提取文本
        extracted_text = self.extract_text_from_file(file_id)
        
        if not extracted_text:
            return {"success": False, "error": "Unable to extract text"}
        
        # 行业分类
        classification = self.classify_industry(extracted_text)
        if isinstance(classification, dict) and 'category' in classification:
            classification['category'] = self._to_english_category(classification['category'])
        
        # 提取关键短语
        key_phrases = self.extract_key_phrases(extracted_text)
        
        # 生成摘要
        summary = self.generate_summary(extracted_text)
        
        # 保存分析结果到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_analysis (file_id, analysis_type, industry_category, extracted_text, key_phrases, summary, confidence_score, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file_id, "full_analysis", classification["category"], 
              extracted_text[:1000], json.dumps(key_phrases, ensure_ascii=False), 
              summary, classification["confidence"], classification.get("method", "Unknown")))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "extracted_text": extracted_text,
            "classification": classification,
            "key_phrases": key_phrases,
            "summary": summary
        }
    
    def get_ai_analysis(self, file_id: int) -> Optional[Dict[str, Any]]:
        """获取文件的AI分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT analysis_type, industry_category, extracted_text, key_phrases, summary, confidence_score, method, analysis_time
            FROM ai_analysis WHERE file_id = ? ORDER BY analysis_time DESC LIMIT 1
        ''', (file_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            analysis_type, industry_category, extracted_text, key_phrases, summary, confidence_score, method, analysis_time = result
            return {
                "analysis_type": analysis_type,
                "industry_category": industry_category,
                "extracted_text": extracted_text,
                "key_phrases": json.loads(key_phrases) if key_phrases else [],
                "summary": summary,
                "confidence_score": confidence_score,
                "method": method or "Unknown",
                "analysis_time": analysis_time
            }
        return None
    
    def create_industry_folder(self, category: str) -> int:
        """为行业分类创建文件夹"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查文件夹是否已存在（英文命名）
        eng_category = self._to_english_category(category)
        cursor.execute('SELECT id FROM folders WHERE folder_name = ?', (f"AI_{eng_category}",))
        result = cursor.fetchone()
        
        if result:
            folder_id = result[0]
        else:
            cursor.execute('''
                INSERT INTO folders (folder_name, parent_folder_id)
                VALUES (?, ?)
            ''', (f"AI_{eng_category}", None))
            folder_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return folder_id
    
    def move_file_to_industry_folder(self, file_id: int, category: str) -> bool:
        """将文件移动到行业分类文件夹"""
        try:
            folder_id = self.create_industry_folder(category)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE files SET folder_id = ? WHERE id = ?', (folder_id, file_id))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    # ==================== 基础文件管理功能 ====================
    
    def rename_file(self, file_id: int, new_filename: str) -> Dict[str, Any]:
        """重命名文件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查新文件名是否已存在
            cursor.execute('SELECT id FROM files WHERE filename = ? AND id != ?', (new_filename, file_id))
            if cursor.fetchone():
                conn.close()
                return {"success": False, "error": "文件名已存在"}
            
            # 更新文件名
            cursor.execute('UPDATE files SET filename = ? WHERE id = ?', (new_filename, file_id))
            conn.commit()
            conn.close()
            
            return {"success": True, "new_filename": new_filename}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_id: int) -> Dict[str, Any]:
        """删除文件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取文件路径
            cursor.execute('SELECT file_path FROM files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if result:
                file_path = result[0]
                
                # 删除物理文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # 删除数据库记录
                cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
                
                # 删除AI分析记录
                cursor.execute('DELETE FROM ai_analysis WHERE file_id = ?', (file_id,))
                
                conn.commit()
                conn.close()
                
                return {"success": True}
            else:
                conn.close()
                return {"success": False, "error": "文件不存在"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def rename_folder(self, folder_id: int, new_folder_name: str) -> Dict[str, Any]:
        """重命名文件夹"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查新文件夹名是否已存在
            cursor.execute('SELECT id FROM folders WHERE folder_name = ? AND id != ?', (new_folder_name, folder_id))
            if cursor.fetchone():
                conn.close()
                return {"success": False, "error": "文件夹名已存在"}
            
            # 更新文件夹名
            cursor.execute('UPDATE folders SET folder_name = ? WHERE id = ?', (new_folder_name, folder_id))
            conn.commit()
            conn.close()
            
            return {"success": True, "new_folder_name": new_folder_name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_folder(self, folder_id: int) -> Dict[str, Any]:
        """删除文件夹"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查文件夹是否为空
            cursor.execute('SELECT COUNT(*) FROM files WHERE folder_id = ?', (folder_id,))
            file_count = cursor.fetchone()[0]
            
            if file_count > 0:
                conn.close()
                return {"success": False, "error": f"文件夹不为空，包含 {file_count} 个文件"}
            
            # 检查是否有子文件夹
            cursor.execute('SELECT COUNT(*) FROM folders WHERE parent_folder_id = ?', (folder_id,))
            subfolder_count = cursor.fetchone()[0]
            
            if subfolder_count > 0:
                conn.close()
                return {"success": False, "error": f"文件夹包含 {subfolder_count} 个子文件夹"}
            
            # 删除文件夹
            cursor.execute('DELETE FROM folders WHERE id = ?', (folder_id,))
            conn.commit()
            conn.close()
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_folders(self, parent_folder_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取文件夹列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if parent_folder_id is None:
            cursor.execute('''
                SELECT id, folder_name, created_time, 
                       (SELECT COUNT(*) FROM files WHERE folder_id = folders.id) as file_count
                FROM folders 
                WHERE parent_folder_id IS NULL
                ORDER BY created_time DESC
            ''')
        else:
            cursor.execute('''
                SELECT id, folder_name, created_time,
                       (SELECT COUNT(*) FROM files WHERE folder_id = folders.id) as file_count
                FROM folders 
                WHERE parent_folder_id = ?
                ORDER BY created_time DESC
            ''', (parent_folder_id,))
        
        folders = []
        for row in cursor.fetchall():
            folders.append({
                "id": row[0],
                "folder_name": row[1],
                "created_time": row[2],
                "file_count": row[3]
            })
        
        conn.close()
        return folders
    
    def sync_cached_files(self) -> Dict[str, Any]:
        """同步缓存文件到云端"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有已缓存的文件
            cursor.execute('''
                SELECT id, filename, file_path, last_modified
                FROM files 
                WHERE is_cached = TRUE
            ''')
            
            cached_files = cursor.fetchall()
            synced_count = 0
            
            for file_id, filename, file_path, last_modified in cached_files:
                # 检查文件是否仍然存在
                if os.path.exists(file_path):
                    # 更新最后修改时间
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute('''
                        UPDATE files 
                        SET last_modified = ? 
                        WHERE id = ?
                    ''', (current_time, file_id))
                    synced_count += 1
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "synced_count": synced_count,
                "message": f"成功同步 {synced_count} 个缓存文件"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# 初始化云存储管理器
if 'storage_manager' not in st.session_state:
    st.session_state.storage_manager = CloudStorageManager()

storage_manager = st.session_state.storage_manager

# 侧边栏
with st.sidebar:
    st.markdown("### 🌾 Agribusiness Expert AI Cloud")
    st.markdown("---")
    
    # 快速操作
    st.markdown("### ⚡ Quick Actions")
    
    # 文件夹管理
    st.markdown("### 📁 Folder Management")
    
    # 创建文件夹
    with st.form("create_folder_form"):
        folder_name = st.text_input("📁 New Folder", placeholder="Enter folder name")
        if st.form_submit_button("Create", width='stretch'):
            if folder_name:
                result = storage_manager.create_folder(folder_name)
                if result["success"]:
                    st.success(f"✅ Folder '{folder_name}' created successfully!")
                else:
                    st.error(f"❌ Creation failed: {result['error']}")
            else:
                st.warning("Please enter folder name")
    
    # 文件夹列表
    folders = storage_manager.get_folders()
    if folders:
        st.markdown("#### Existing Folders")
        for folder in folders:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📁 {folder['folder_name']}")
                st.caption(f"Files: {folder['file_count']} | Created: {folder['created_time']}")
            with col2:
                # 重命名文件夹
                with st.popover("✏️", help="Rename folder"):
                    new_name = st.text_input("New Name", value=folder['folder_name'], key=f"folder_rename_{folder['id']}")
                    if st.button("✅ Confirm", key=f"folder_rename_confirm_{folder['id']}"):
                        result = storage_manager.rename_folder(folder['id'], new_name)
                        if result["success"]:
                            st.success("Rename successful!")
                            st.rerun()
                        else:
                            st.error(f"Rename failed: {result['error']}")
            with col3:
                # 删除文件夹
                if st.button("🗑️", key=f"folder_delete_{folder['id']}", help="Delete folder"):
                    result = storage_manager.delete_folder(folder['id'])
                    if result["success"]:
                        st.success("Folder deleted!")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {result['error']}")
    
    # 同步功能
    st.markdown("---")
    if st.button("🔄 Sync Cache", width='stretch', help="Sync all cached files"):
        result = storage_manager.sync_cached_files()
        if result["success"]:
            st.success(result["message"])
        else:
            st.error(f"Sync failed: {result['error']}")
    
    st.markdown("---")
    
    # Agribusiness工具与AI功能区域
    st.markdown("### 🌾 Agribusiness Tools & AI")
    with st.expander("☁️ Weather & Climate (Open-Meteo)", expanded=False):
        colw1, colw2 = st.columns(2)
        with colw1:
            lat = st.number_input("Latitude", value=0.0, step=0.1)
        with colw2:
            lon = st.number_input("Longitude", value=20.0, step=0.1)
        if st.button("Fetch 7-Day Climate Summary", use_container_width=True):
            with st.spinner("Fetching weather data..."):
                res = storage_manager.fetch_weather_summary(lat, lon)
                if res.get("success"):
                    ws = res["weather"]["summary"]
                    st.success("Weather updated")
                    st.write({
                        "7d total rainfall (mm)": ws.get("7d_total_rain_mm"),
                        "Avg Tmax (°C)": ws.get("avg_tmax"),
                        "Avg Tmin (°C)": ws.get("avg_tmin")
                    })
                else:
                    st.error(f"Weather fetch failed: {res.get('error')}")

    with st.expander("🛰️ Remote Sensing (NDVI/EVI)", expanded=False):
        colr1, colr2, colr3 = st.columns(3)
        with colr1:
            rs_lat = st.number_input("Latitude", value=0.0, step=0.1, key="rs_lat")
        with colr2:
            rs_lon = st.number_input("Longitude", value=20.0, step=0.1, key="rs_lon")
        with colr3:
            rs_days = st.slider("Days", min_value=7, max_value=60, value=30, step=1, key="rs_days")
        if st.button("Generate NDVI/EVI Timeseries", use_container_width=True):
            with st.spinner("Generating NDVI/EVI (stub)..."):
                res = storage_manager.compute_remote_sensing_stub(rs_lat, rs_lon, rs_days)
                if res.get("success"):
                    rs = res["remote_sensing"]
                    st.success("Generated")
                    if rs.get("dates") and rs.get("ndvi"):
                        st.markdown("**NDVI**")
                        ndvi_df = pd.DataFrame({"date": rs["dates"], "NDVI": rs["ndvi"]}).set_index("date")
                        st.line_chart(ndvi_df)
                    if rs.get("dates") and rs.get("evi"):
                        st.markdown("**EVI**")
                        evi_df = pd.DataFrame({"date": rs["dates"], "EVI": rs["evi"]}).set_index("date")
                        st.line_chart(evi_df)
                else:
                    st.error(f"Remote sensing generation failed: {res.get('error')}")

    with st.expander("🧮 Agri Quick Calculator", expanded=False):
        st.caption("Quick estimation: total production & profit")
        # 总产量 = 面积 × 单产（自动做少量单位适配）
        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            area_value = st.number_input("Area value", value=100.0, step=1.0)
            area_unit = st.selectbox("Area unit", ["hectare(ha)", "mu"], index=0)
        with colc2:
            yield_value = st.number_input("Yield value", value=3.0, step=0.1)
            yield_unit = st.selectbox("Yield unit", ["t/ha", "kg/ha", "kg/mu", "jin/mu"], index=0)
        with colc3:
            currency = st.selectbox("Currency", ["USD", "KES", "NGN", "ZAR", "GHS", "XOF", "XAF", "ETB", "TZS"], index=1)
            price_value = st.number_input("Price (per kg)", value=0.5, step=0.05)
            cost_value = st.number_input("Total cost", value=50000.0, step=1000.0)

        if st.button("Calculate production & profit", use_container_width=True):
            # 单位换算到 公斤/亩
            if yield_unit == "jin/mu":
                yield_kg_per_mu = yield_value * 0.5
            elif yield_unit == "kg/ha":
                yield_kg_per_mu = yield_value / 15.0  # 1 ha ≈ 15 亩
            elif yield_unit == "t/ha":
                yield_kg_per_mu = (yield_value * 1000.0) / 15.0
            else:
                yield_kg_per_mu = yield_value

            # 面积换算到 亩
            area_mu = area_value * (15.0 if area_unit == "hectare(ha)" else 1.0)

            total_production_kg = area_mu * yield_kg_per_mu
            revenue = total_production_kg * price_value
            profit = revenue - cost_value
            st.success("Calculated")
            st.write({
                "Total production (kg)": round(total_production_kg, 2),
                f"Revenue ({currency})": round(revenue, 2),
                f"Profit ({currency})": round(profit, 2)
            })
    
    # AI模型状态
    with st.expander("🔍 AI Model Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if OCR_AVAILABLE and storage_manager.ocr_reader is not None:
                st.success("✅ OCR Text Recognition")
            elif OCR_AVAILABLE:
                st.warning("🔄 OCR model loading...")
            else:
                st.error("❌ OCR Text Recognition")
            
            if TRANSFORMERS_AVAILABLE:
                st.success("✅ Deep Learning Model")
            else:
                st.error("❌ Deep Learning Model")
        
        with col2:
            if ML_AVAILABLE:
                st.success("✅ Machine Learning Classification")
            else:
                st.error("❌ Machine Learning Classification")
            
            if OPENAI_AVAILABLE:
                st.success("✅ OpenAI Integration")
            else:
                st.warning("⚠️ OpenAI Integration")
    
    # AI分析按钮
    if st.button("🧠 Smart Analysis", width='stretch', help="Perform AI analysis on all files"):
        st.session_state.show_ai_analysis = True
    else:
        st.session_state.show_ai_analysis = False
    
    # 重新初始化AI模型
    if st.button("🔄 Reload AI", width='stretch', help="Reinitialize AI models"):
        with st.spinner("Reloading AI models..."):
            storage_manager.init_ai_models()
            st.success("✅ AI models reloaded successfully!")
    
    # 行业分类查看
    if st.button("📊 Industry Classification", width='stretch', help="View files classified by industry"):
        st.session_state.show_industry_view = True
    else:
        st.session_state.show_industry_view = False
    
    # 智能报告生成
    if st.button("📈 Smart Report", width='stretch', help="Generate smart analysis reports and charts"):
        st.session_state.show_smart_report = True
    else:
        st.session_state.show_smart_report = False
    
    st.markdown("---")
    
    # 搜索功能
    st.markdown("### 🔍 Search Files")
    search_query = st.text_input("Search File Name", placeholder="Enter keywords")
    search_type = st.selectbox("File Type", ["All", "image", "application", "text", "video", "audio"])
    
    if st.button("🔍 Search", width='stretch') and search_query:
        file_type = None if search_type == "All" else search_type
        search_results = storage_manager.search_files(search_query, file_type)
        st.session_state.search_results = search_results
        st.session_state.show_search = True
    else:
        st.session_state.show_search = False

# 主界面
st.title("🌾 Agribusiness Expert AI Cloud")
st.markdown("Built for agribusiness: document management + KPIs + climate/remote sensing insights")

# 文件上传区域
st.markdown("### 📤 File Upload")

# 上传模式选择
upload_mode = st.radio(
    "Select Upload Mode",
    ["Normal Upload", "Resume Upload"],
    horizontal=True,
    help="Resume upload supports continuing after network interruption"
)

# 选择上传文件夹
folders = storage_manager.get_folders()
folder_options = ["Root Directory"] + [f["folder_name"] for f in folders]
selected_folder = st.selectbox("Select Upload Folder", folder_options, help="Choose the folder to upload files to")

# 获取选中的文件夹ID
target_folder_id = None
if selected_folder != "Root Directory":
    for folder in folders:
        if folder["folder_name"] == selected_folder:
            target_folder_id = folder["id"]
            break

uploaded_files = st.file_uploader(
    "Choose Files to Upload", 
    type=["xlsx", "xls", "csv", "pdf", "png", "jpg", "jpeg", "gif", "bmp", "txt", "doc", "docx"],
    accept_multiple_files=True,
    help="Supports Excel, PDF, Images, CSV and other formats"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"📄 {uploaded_file.name} ({storage_manager.format_file_size(len(uploaded_file.getbuffer()))})")
        
        with col2:
            if upload_mode == "Normal Upload":
                if st.button(f"📤 Upload", key=f"upload_{uploaded_file.name}"):
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        result = storage_manager.upload_file(uploaded_file, target_folder_id)
                        if result["success"]:
                            folder_name = selected_folder if selected_folder != "Root Directory" else "Root Directory"
                            st.success(f"✅ {uploaded_file.name} uploaded to {folder_name}!")
                        else:
                            st.error(f"❌ Upload failed: {result['error']}")
            else:
                if st.button(f"🔄 Resume Upload", key=f"resume_upload_{uploaded_file.name}"):
                    with st.spinner(f"Resume uploading {uploaded_file.name}..."):
                        result = storage_manager.upload_file_with_resume(uploaded_file, target_folder_id)
                        if result["success"]:
                            folder_name = selected_folder if selected_folder != "Root Directory" else "Root Directory"
                            st.success(f"✅ {uploaded_file.name} resume uploaded to {folder_name}!")
                        else:
                            st.error(f"❌ Resume upload failed: {result['error']}")

# 上传进度显示
progress_list = storage_manager.get_upload_progress()
if progress_list:
    st.markdown("### 🔄 Upload Progress")
    for progress in progress_list:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"📄 {progress['filename']}")
            st.progress(progress['progress'])
            st.caption(f"{storage_manager.format_file_size(progress['uploaded_size'])} / {storage_manager.format_file_size(progress['total_size'])}")
        
        with col2:
            if st.button("🔄 继续", key=f"resume_{progress['filename']}"):
                result = storage_manager.resume_upload(progress['filename'])
                if result["success"]:
                    st.success("Continue uploading...")
                else:
                    st.error("Unable to continue upload")
        
        with col3:
            if st.button("❌ 取消", key=f"cancel_{progress['filename']}"):
                if storage_manager.cancel_upload(progress['filename']):
                    st.success("Upload cancelled")
                    st.rerun()
                else:
                    st.error("Cancel failed")

# 文件夹导航
current_folder_id = st.session_state.get('current_folder_id', None)
if current_folder_id is not None:
    # 显示当前文件夹信息
    conn = sqlite3.connect(storage_manager.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT folder_name FROM folders WHERE id = ?', (current_folder_id,))
    folder_name = cursor.fetchone()
    conn.close()
    
    if folder_name:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📁 Current Folder: {folder_name[0]}")
        with col2:
            if st.button("⬅️ Back to Root", width='stretch'):
                st.session_state.current_folder_id = None
                st.rerun()

# 检查显示模式
files = []  # 确保后续使用时已定义
if st.session_state.get('show_ai_analysis', False):
    st.markdown("### 🤖 AI Smart Analysis")
    
    # 获取所有文件进行AI分析
    all_files = storage_manager.get_files()
    
    if all_files:
        st.info(f"Analyzing {len(all_files)} files with AI...")
        
        # 批量AI分析
        analysis_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(all_files):
            status_text.text(f"Analyzing: {file['filename']}")
            result = storage_manager.analyze_file_with_ai(file['id'])
            analysis_results.append({
                'file': file,
                'analysis': result
            })
            progress_bar.progress((i + 1) / len(all_files))
        
        progress_bar.empty()
        status_text.empty()
        
        # 显示分析结果
        st.success("AI analysis completed!")
        
        # 按行业分类显示
        industry_groups = {}
        for result in analysis_results:
            if result['analysis']['success']:
                category = result['analysis']['classification']['category']
                if category not in industry_groups:
                    industry_groups[category] = []
                industry_groups[category].append(result)
        
        for category, files in industry_groups.items():
            with st.expander(f"📊 {category} ({len(files)} files)", expanded=True):
                for result in files:
                    file = result['file']
                    analysis = result['analysis']
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {file['filename']}")
                        st.caption(f"Confidence: {analysis['classification']['confidence']:.2%}")
                        if analysis['summary']:
                            st.info(f"Summary: {analysis['summary']}")
                    with col2:
                        if st.button("📁 Classify", key=f"batch_classify_{file['id']}"):
                            if storage_manager.move_file_to_industry_folder(file['id'], category):
                                st.success("Classified!")
                                st.rerun()
    
    else:
        st.warning("No files to analyze")

elif st.session_state.get('show_industry_view', False):
    st.markdown("### 📊 Industry Classification View")
    
    # 获取所有行业分类
    conn = sqlite3.connect(storage_manager.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT industry_category, COUNT(*) as file_count
        FROM ai_analysis 
        WHERE industry_category IS NOT NULL
        GROUP BY industry_category
        ORDER BY file_count DESC
    ''')
    categories = cursor.fetchall()
    conn.close()
    
    if categories:
        for category, count in categories:
            with st.expander(f"📁 {category} ({count} files)", expanded=True):
                # 获取该分类的文件
                conn = sqlite3.connect(storage_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT f.id, f.filename, f.file_size, f.upload_time, a.confidence_score, a.summary
                    FROM files f
                    JOIN ai_analysis a ON f.id = a.file_id
                    WHERE a.industry_category = ?
                    ORDER BY a.confidence_score DESC
                ''', (category,))
                file_rows = cursor.fetchall()
                conn.close()
                
                # 转换为字典格式以保持一致性
                files = []
                for file_id, filename, file_size, upload_time, confidence, summary in file_rows:
                    files.append({
                        "id": file_id,
                        "filename": filename,
                        "file_size": file_size,
                        "upload_time": upload_time,
                        "confidence": confidence,
                        "summary": summary
                    })
                
                for file in files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {file['filename']}")
                        st.caption(f"Uploaded: {file['upload_time']}")
                        if file['summary']:
                            st.info(f"Summary: {file['summary']}")
                    with col2:
                        st.metric("Confidence", f"{file['confidence']:.2%}")
                    with col3:
                        st.metric("File Size", storage_manager.format_file_size(file['file_size']))
    else:
        st.info("No files have been analyzed by AI yet")

# 文件列表显示
elif st.session_state.get('show_search', False) and 'search_results' in st.session_state:
    st.markdown("### 🔍 Search Results")
    files = st.session_state.search_results
    st.info(f"🔍 Search Results: Found {len(files)} files")
else:
    st.markdown("### 📁 File List")
    files = storage_manager.get_files(current_folder_id)
    
    # 显示子文件夹
    if current_folder_id is None:
        subfolders = storage_manager.get_folders()
    else:
        subfolders = storage_manager.get_folders(current_folder_id)
    
    if subfolders:
        st.markdown("#### 📁 Folders")
        for folder in subfolders:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(f"📁 {folder['folder_name']} ({folder['file_count']} files)", key=f"enter_folder_{folder['id']}", width='stretch'):
                    st.session_state.current_folder_id = folder['id']
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"rename_folder_ui_{folder['id']}", help="Rename"):
                    st.session_state[f"rename_folder_{folder['id']}"] = True
            with col3:
                if st.button("🗑️", key=f"delete_folder_ui_{folder['id']}", help="Delete"):
                    result = storage_manager.delete_folder(folder['id'])
                    if result["success"]:
                        st.success("Folder deleted!")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {result['error']}")
        
        st.markdown("---")

if files:
    # 文件统计
    total_size = sum(file.get('file_size', 0) for file in files)
    cached_count = sum(1 for file in files if file.get('is_cached', False))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", len(files))
    with col2:
        st.metric("Total Size", storage_manager.format_file_size(total_size))
    with col3:
        st.metric("Cached", f"{cached_count}/{len(files)}")
    with col4:
        st.metric("Cache Rate", f"{cached_count/len(files)*100:.1f}%")
    
    st.markdown("---")
    
    # 文件列表 - 使用卡片式布局
    for file in files:
        with st.container():
            # 文件卡片
            st.markdown(f"""
            <div class="file-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; flex: 1;">
                        <span class="file-icon">{storage_manager.get_file_icon(file.get('file_type', 'unknown'))}</span>
                        <div>
                            <h4 style="margin: 0; color: #1e293b;">{file.get('filename', 'Unknown')}</h4>
                            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 14px;">
                                Type: {file.get('file_type', 'unknown')} | Uploaded: {file.get('upload_time', '')}
                            </p>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <span style="font-weight: 600; color: #475569;">{storage_manager.format_file_size(file.get('file_size', 0))}</span>
                        <span style="padding: 4px 8px; border-radius: 4px; font-size: 12px; background: {'#dcfce7' if file.get('is_cached', False) else '#dbeafe'}; color: {'#166534' if file.get('is_cached', False) else '#1e40af'};">
                            {'✅ Cached' if file.get('is_cached', False) else '☁️ Cloud'}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 使用两列布局：左侧操作按钮，右侧预览内容
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                # 预览控制
                show_preview = st.checkbox("👁️ Preview File", key=f"preview_{file['id']}", help="Click to preview file content")
                
                # 操作按钮行
                col1, col2 = st.columns(2)
                
                with col1:
                    # AI分析按钮
                    if st.button("🧠 AI Analysis", key=f"ai_analyze_{file['id']}", help="Use AI to analyze file content", width='stretch'):
                        with st.spinner("AI is analyzing file..."):
                            result = storage_manager.analyze_file_with_ai(file['id'])
                            if result["success"]:
                                st.success("AI analysis completed!")
                                st.rerun()
                            else:
                                st.error(f"AI analysis failed: {result['error']}")
                    
                    # 智能报告按钮
                    if st.button("📈 Smart Report", key=f"smart_report_{file['id']}", help="Generate smart analysis report and charts", width='stretch'):
                        with st.spinner("Generating smart report..."):
                            result = storage_manager.generate_smart_report(file['id'])
                            if result["success"]:
                                st.session_state[f"show_report_{file['id']}"] = True
                                st.session_state[f"report_data_{file['id']}"] = result
                                st.success("Smart report generated successfully!")
                                st.rerun()
                            else:
                                st.error(f"Report generation failed: {result['error']}")
                
                with col2:
                    # 缓存按钮
                    if not file.get('is_cached', False):
                        if st.button("💾 Cache", key=f"cache_{file['id']}", help="Cache to local", width='stretch'):
                            if storage_manager.cache_file(file['id']):
                                st.success("Cached successfully!")
                                st.rerun()
                            else:
                                st.error("Cache failed")
                    else:
                        st.success("Cached")
                    
                    # 下载按钮
                    if st.button("📥 Download", key=f"download_btn_{file['id']}", help="Download file", width='stretch'):
                        file_data = storage_manager.preview_file(file['id'])
                        if file_data:
                            st.download_button(
                                "📥 Download File",
                                file_data,
                                file.get('filename', 'file'),
                                key=f"download_file_{file.get('id', 'unknown')}"
                            )
                        else:
                            st.error("File not found")
                
                # 文件操作菜单
                with st.popover("⚙️ Actions", help="File operation menu"):
                    # 重命名
                    new_name = st.text_input("Rename", value=file.get('filename', ''), key=f"rename_input_{file.get('id','unknown')}")
                    if st.button("✅ Confirm Rename", key=f"rename_confirm_{file['id']}"):
                        result = storage_manager.rename_file(file['id'], new_name)
                        if result["success"]:
                            st.success("Rename successful!")
                            st.rerun()
                        else:
                            st.error(f"Rename failed: {result['error']}")
                    
                    st.markdown("---")
                    
                    # 移动文件
                    st.markdown("**Move to Folder:**")
                    move_folders = storage_manager.get_folders()
                    move_options = ["Root Directory"] + [f["folder_name"] for f in move_folders]
                    target_move_folder = st.selectbox("Select Target Folder", move_options, key=f"move_folder_{file['id']}")
                    
                    if st.button("📁 Move File", key=f"move_file_{file['id']}"):
                        target_move_folder_id = None
                        if target_move_folder != "Root Directory":
                            for folder in move_folders:
                                if folder["folder_name"] == target_move_folder:
                                    target_move_folder_id = folder["id"]
                                    break
                        
                        conn = sqlite3.connect(storage_manager.db_path)
                        cursor = conn.cursor()
                        cursor.execute('UPDATE files SET folder_id = ? WHERE id = ?', (target_move_folder_id, file['id']))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"File moved to {target_move_folder}!")
                        st.rerun()
                    
                    st.markdown("---")
                    
                    # 删除
                    if st.button("🗑️ Delete File", key=f"delete_{file['id']}", help="Permanently delete file"):
                        if st.session_state.get(f"confirm_delete_{file['id']}", False):
                            result = storage_manager.delete_file(file['id'])
                            if result["success"]:
                                st.success("File deleted!")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {result['error']}")
                        else:
                            st.session_state[f"confirm_delete_{file['id']}"] = True
                            st.warning("⚠️ Click again to confirm deletion")
            
            with col_right:
                # 预览内容区域 - 放在右侧列
                if show_preview:
                    st.markdown("#### 📄 File Preview")
                    
                    file_data = storage_manager.preview_file(file['id'])
                    if file_data:
                        if file['file_type'] == 'image':
                            st.image(file_data, caption=file.get('filename', ''), width='stretch')
                        elif file['file_type'] == 'application' and str(file.get('filename','')).endswith('.pdf'):
                            if PDF_AVAILABLE:
                                try:
                                    # 使用BytesIO包装数据
                                    import io
                                    pdf_stream = io.BytesIO(file_data)
                                    doc = fitz.open(stream=pdf_stream, filetype="pdf")
                                    
                                    if len(doc) > 0:
                                        page = doc[0]
                                        # 设置合适的缩放比例
                                        mat = fitz.Matrix(1.5, 1.5)  # 1.5倍缩放
                                        pix = page.get_pixmap(matrix=mat)
                                        img_data = pix.tobytes("png")
                                        st.image(img_data, caption=f"PDF Preview: {file.get('filename','')} (Page 1)", width='stretch')
                                        
                                        # 显示页数信息
                                        if len(doc) > 1:
                                            st.caption(f"PDF has {len(doc)} pages, showing page 1")
                                    else:
                                        st.warning("PDF file is empty or cannot be read")
                                    
                                    doc.close()
                                except Exception as e:
                                    st.error(f"PDF preview failed: {str(e)}")
                                    st.info("Try downloading the file to view content")
                                    st.download_button(
                                        "📥 Download PDF",
                                        file_data,
                                        file.get('filename','file.pdf'),
                                        key=f"preview_download_pdf_{file.get('id','unknown')}"
                                    )
                            else:
                                st.info("PDF preview requires PyMuPDF module")
                                st.info("Please run: pip install PyMuPDF")
                                st.download_button(
                                    "📥 Download PDF",
                                    file_data,
                                    file.get('filename','file.pdf'),
                                    key=f"preview_download_pdf_no_fitz_{file.get('id','unknown')}"
                                )
                        elif file['file_type'] == 'application' and str(file.get('filename','')).endswith(('.xlsx', '.xls')):
                            try:
                                import pandas as pd
                                import io
                                df = pd.read_excel(io.BytesIO(file_data))
                                # 确保DataFrame不为空
                                if not df.empty:
                                    # 安全地显示DataFrame，避免numpy.str_错误
                                    try:
                                        st.dataframe(df.head(10), width='stretch')
                                        st.caption(f"Excel Preview: {file.get('filename','')} (Showing first 10 rows)")
                                    except Exception as display_error:
                                        # 如果dataframe显示失败，显示基本信息
                                        st.write(f"Excel File: {file.get('filename','')}")
                                        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                                        st.write("Column names:", list(df.columns))
                                else:
                                    st.warning("Excel file is empty")
                            except Exception as e:
                                st.error(f"Excel preview failed: {str(e)}")
                                st.download_button(
                                    "📥 Download Excel",
                                    file_data,
                                    file.get('filename','file.xlsx'),
                                    key=f"preview_download_excel_{file.get('id','unknown')}"
                                )
                        elif file['file_type'] == 'text' or str(file.get('filename','')).endswith('.txt'):
                            try:
                                text_content = file_data.decode('utf-8')
                                st.text_area("File Content", text_content[:1000], height=200, key=f"text_preview_{file.get('id','unknown')}")
                                if len(text_content) > 1000:
                                    st.caption(f"Text Preview: {file.get('filename','')} (Showing first 1000 characters)")
                                else:
                                    st.caption(f"Text Preview: {file.get('filename','')}")
                            except Exception as e:
                                st.error(f"Text preview failed: {str(e)}")
                                st.download_button(
                                    "📥 Download Text",
                                    file_data,
                                    file.get('filename','file.txt'),
                                    key=f"preview_download_txt_{file.get('id','unknown')}"
                                )
                        else:
                            st.info(f"Preview not supported for {file['file_type']} file type")
                            st.download_button(
                                "📥 Download File",
                                file_data,
                                file.get('filename','file'),
                                key=f"preview_download_other_{file.get('id','unknown')}"
                            )
                    else:
                        st.error("Unable to read file content")
            
            # AI分析结果显示
            ai_analysis = storage_manager.get_ai_analysis(file['id'])
            if ai_analysis:
                st.markdown("---")
                st.markdown("#### 🤖 AI Analysis Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**Industry Category**: {ai_analysis['industry_category']}")
                    st.markdown(f"**Confidence**: {ai_analysis['confidence_score']:.2%}")
                    st.markdown(f"**Analysis Method**: {ai_analysis.get('method', 'Unknown')}")
                    st.markdown(f"**Analysis Time**: {ai_analysis['analysis_time']}")
                
                with col2:
                    if ai_analysis['key_phrases']:
                        st.markdown("**Key Phrases**:")
                        for phrase in ai_analysis['key_phrases'][:5]:
                            st.markdown(f"• {phrase}")
                
                if ai_analysis['summary']:
                    st.markdown("**Document Summary**:")
                    st.info(ai_analysis['summary'])
                
                # Auto classify button: only show error when user clicks and action fails
                auto_clicked = st.button(
                    "📁 Auto Classify",
                    key=f"auto_classify_{file['id']}",
                    help="Move file to corresponding industry folder"
                )
                if auto_clicked:
                    if storage_manager.move_file_to_industry_folder(file['id'], ai_analysis['industry_category']):
                        st.success(f"File moved to {ai_analysis['industry_category']} folder!")
                        st.rerun()
                    else:
                        st.error("Classification failed")
            
            # 智能报告显示
            if st.session_state.get(f"show_report_{file['id']}", False):
                report_data = st.session_state.get(f"report_data_{file['id']}")
                if report_data and report_data["success"]:
                    st.markdown("---")
                    st.markdown("#### 📈 Smart Analysis Report")
                    
                    # 显示报告内容
                    st.markdown(report_data["report"])
                    
                    # 显示图表
                    if report_data["charts"]:
                        st.markdown("#### 📊 Data Visualization Charts")
                        
                        for chart in report_data["charts"]:
                            st.markdown(f"**{chart['title']}**")
                            
                            if chart['type'] == 'bar':
                                # 柱状图
                                chart_data = pd.DataFrame({
                                    'Category': chart['data']['labels'],
                                    'Value': chart['data']['values']
                                })
                                st.bar_chart(chart_data.set_index('Category'))
                            
                            elif chart['type'] == 'pie':
                                # 饼图
                                pie_data = pd.DataFrame({
                                    'Category': chart['data']['labels'],
                                    'Value': chart['data']['values'],
                                    'Percentage': chart['data']['percentages']
                                })
                                st.dataframe(pie_data)
                            
                            elif chart['type'] == 'line':
                                # 折线图
                                line_data = pd.DataFrame({
                                    'Category': chart['data']['labels'],
                                    'Value': chart['data']['values']
                                })
                                st.line_chart(line_data.set_index('Category'))
                            
                            st.markdown("---")
                    
                    # 关闭报告按钮
                    if st.button("❌ Close Report", key=f"close_report_{file['id']}"):
                        st.session_state[f"show_report_{file['id']}"] = False
                        st.rerun()
            

else:
    # 空状态
    st.markdown("<div style='text-align: center; padding: 40px 0;'>", unsafe_allow_html=True)
    st.header("📁 No Files")
    st.subheader("Upload your first file to start using cloud storage")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 功能说明
    features = st.columns(3)
    with features[0]:
        st.info("""
        **📤 File Upload**
        - Multiple formats support
        - Resume upload
        - Auto validation
        """)
    with features[1]:
        st.success("""
        **👁️ Online Preview**
        - Instant image preview
        - PDF document viewing
        - No download needed
        """)
    with features[2]:
        st.warning("""
        **💾 Local Cache**
        - Offline access
        - Auto sync
        - Smart management
        """)

# 页脚
st.markdown("---")
st.markdown("**Built with ❤️ • AI Cloud Storage System • ☁️ Intelligent Storage**")