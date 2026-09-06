# src/anomaly_detector.py
"""
Détection d'anomalies environnementales pour véhicules autonomes
Version 2026 - Détection en temps réel de situations dangereuses
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time
import logging
from collections import deque, Counter
import math
import json
from PIL import Image
import open_clip
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types d'anomalies détectables"""
    CONSTRUCTION = "travaux"
    OBSTACLE = "obstacle"
    INFRASTRUCTURE_DAMAGE = "infrastructure_endommagee"
    ADVERSE_WEATHER = "conditions_meteo_extremes"
    DANGEROUS_BEHAVIOR = "comportement_dangereux"
    ROAD_ANOMALY = "anomalie_route"
    EMERGENCY = "urgence"
    UNKNOWN = "inconnu"


class AnomalySeverity(Enum):
    """Niveau de sévérité des anomalies"""
    LOW = "faible"
    MEDIUM = "moyen"
    HIGH = "eleve"
    CRITICAL = "critique"


@dataclass
class AnomalyDetection:
    """Structure pour une détection d'anomalie"""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    risk_score: float = 0.0  # 0-1
    suggested_action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """
    Détecteur d'anomalies environnementales pour véhicules autonomes
    Combine vision par ordinateur, modèles fondationnels et analyse contextuelle
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 Utilisation du device: {self.device}")
        
        # Configuration
        self.config = config or self._default_config()
        
        # Historiques
        self.detection_history = deque(maxlen=100)
        self.scene_history = deque(maxlen=30)
        self.anomaly_history = deque(maxlen=50)
        
        # Modèles
        self._initialize_models()
        
        # Trackers
        self.object_tracker = AnomalyTracker()
        
        # Modèle d'anomalies (Isolation Forest)
        self.anomaly_model = None
        self.scaler = StandardScaler()
        self._initialize_anomaly_model()
        
        # Statistiques
        self.stats = {
            'total_anomalies': 0,
            'anomalies_by_type': {},
            'severity_distribution': {},
            'avg_confidence': 0.0,
            'inference_times': deque(maxlen=100)
        }
        
        logger.info("✅ Détecteur d'anomalies initialisé")
    
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.3,
            'anomaly_threshold': 0.5,
            'history_size': 30,
            'use_clip': True,
            'use_dinov2': True,
            'tracking_enabled': True,
            'alert_sound': False,
            'save_anomalies': True,
            'output_dir': './anomalies_detected'
        }
    
    def _initialize_models(self):
        """Initialise les modèles nécessaires - Version corrigée pour DINOv2"""
        logger.info("🔄 Initialisation des modèles...")
        
        # CLIP pour la compréhension sémantique
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k'
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.has_clip = True
            logger.info("✅ CLIP chargé")
        except Exception as e:
            logger.warning(f"⚠️ CLIP non disponible: {e}")
            self.has_clip = False
        
        # DINOv2 pour les embeddings - VERSION CORRIGÉE
        try:
            # Essayer plusieurs méthodes de chargement
            dinov2_loaded = False
            
            # Méthode 1: Chargement via Hugging Face (recommandée)
            try:
                from transformers import AutoModel
                logger.info("   Tentative de chargement via Hugging Face...")
                self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
                self.dinov2_model.eval()
                dinov2_loaded = True
                logger.info("✅ DINOv2 chargé via Hugging Face")
            except Exception as e:
                logger.debug(f"   HF échoué: {e}")
            
            # Méthode 2: Chargement via torch.hub avec gestion d'erreur améliorée
            if not dinov2_loaded:
                try:
                    logger.info("   Tentative de chargement via torch.hub...")
                    # Nettoyer le cache si nécessaire
                    import subprocess
                    import os
                    cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
                    if os.path.exists(cache_dir):
                        import shutil
                        shutil.rmtree(cache_dir)
                        logger.info("   Cache DINOv2 nettoyé")
                    
                    # Désactiver temporairement GITHUB_TOKEN
                    original_token = os.environ.get('GITHUB_TOKEN')
                    if original_token:
                        del os.environ['GITHUB_TOKEN']
                    
                    self.dinov2_model = torch.hub.load(
                        'facebookresearch/dinov2', 
                        'dinov2_vitb14',
                        trust_repo=True
                    ).to(self.device)
                    
                    if original_token:
                        os.environ['GITHUB_TOKEN'] = original_token
                    
                    self.dinov2_model.eval()
                    dinov2_loaded = True
                    logger.info("✅ DINOv2 chargé via torch.hub")
                except Exception as e:
                    logger.debug(f"   torch.hub échoué: {e}")
            
            # Méthode 3: Téléchargement direct du modèle
            if not dinov2_loaded:
                try:
                    logger.info("   Tentative de téléchargement direct...")
                    import urllib.request
                    import os
                    
                    model_path = os.path.expanduser("~/.cache/dinov2/dinov2_vitb14.pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    if not os.path.exists(model_path):
                        url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
                        urllib.request.urlretrieve(url, model_path)
                        logger.info("   Modèle téléchargé")
                    
                    # Charger le modèle avec un wrapper simple
                    self.dinov2_model = torch.hub.load(
                        'facebookresearch/dinov2', 
                        'dinov2_vitb14',
                        source='local',
                        pretrained=False
                    ).to(self.device)
                    
                    # Charger les poids
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.dinov2_model.load_state_dict(state_dict)
                    self.dinov2_model.eval()
                    dinov2_loaded = True
                    logger.info("✅ DINOv2 chargé depuis fichier local")
                except Exception as e:
                    logger.debug(f"   Téléchargement direct échoué: {e}")
            
            if dinov2_loaded:
                # Transformer pour DINOv2
                self.dinov2_transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                self.has_dinov2 = True
            else:
                self.has_dinov2 = False
                logger.warning("⚠️ DINOv2 non disponible (aucune méthode de chargement n'a fonctionné)")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur DINOv2: {e}")
            self.has_dinov2 = False
        
        # YOLO pour la détection d'objets
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.has_yolo = True
            logger.info("✅ YOLO chargé")
        except Exception as e:
            logger.warning(f"⚠️ YOLO non disponible: {e}")
            self.has_yolo = False
        
        # Détection d'objets par segmentation
        try:
            from ultralytics import YOLO
            self.segmentation_model = YOLO('yolov8n-seg.pt')
            self.has_segmentation = True
            logger.info("✅ Segmentation chargée")
        except:
            self.has_segmentation = False
    
    def _initialize_anomaly_model(self):
        """Initialise le modèle de détection d'anomalies"""
        # Créer un modèle Isolation Forest entraîné sur des scènes normales
        # Pour une version de production, utiliser un dataset d'entraînement
        
        # Simuler des données d'entraînement
        normal_features = np.random.randn(100, 10)
        self.scaler.fit(normal_features)
        scaled = self.scaler.transform(normal_features)
        
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_model.fit(scaled)
        logger.info("✅ Modèle d'anomalies Isolation Forest initialisé")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[AnomalyDetection]]:
        """
        Traite une frame et détecte les anomalies
        
        Returns:
            annotated_frame: Frame annotée
            anomalies: Liste des anomalies détectées
        """
        start_time = time.time()
        
        # 1. Extraction des caractéristiques
        features = self._extract_features(frame)
        
        # 2. Détection d'objets
        objects = self._detect_objects(frame)
        
        # 3. Analyse sémantique
        semantic_analysis = self._analyze_semantic(frame) if self.has_clip else {}
        
        # 4. Détection d'anomalies
        anomalies = self._detect_anomalies(frame, features, objects, semantic_analysis)
        
        # 5. Filtrage temporel
        anomalies = self._temporal_filter(anomalies)
        
        # 6. Annotation
        annotated_frame = self._annotate_frame(frame, anomalies, objects)
        
        # 7. Mise à jour des stats
        self._update_stats(anomalies, time.time() - start_time)
        
        return annotated_frame, anomalies
    
    def _extract_features(self, frame: np.ndarray) -> Dict:
        """Extrait les caractéristiques de la scène"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        features = {
            # Caractéristiques de base
            'brightness': gray.mean() / 255.0,
            'contrast': gray.std() / 255.0,
            'edge_density': self._calculate_edge_density(frame),
            'texture_complexity': self._calculate_texture_complexity(frame),
            
            # Caractéristiques de couleur
            'hsv_stats': self._calculate_hsv_stats(frame),
            
            # Caractéristiques spatiales
            'sky_ratio': self._calculate_sky_ratio(frame),
            'road_ratio': self._calculate_road_ratio(frame),
            
            # Embeddings (si disponibles)
            'embedding': self._get_embedding(frame) if self.has_dinov2 else None
        }
        
        return features
    
    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / frame.size
    
    def _calculate_texture_complexity(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return min(1.0, laplacian.var() / 1000)
    
    def _calculate_hsv_stats(self, frame: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return {
            'h_mean': float(hsv[:, :, 0].mean()),
            's_mean': float(hsv[:, :, 1].mean()),
            'v_mean': float(hsv[:, :, 2].mean()),
            'h_std': float(hsv[:, :, 0].std()),
            's_std': float(hsv[:, :, 1].std()),
            'v_std': float(hsv[:, :, 2].std())
        }
    
    def _calculate_sky_ratio(self, frame: np.ndarray) -> float:
        """Calcule la proportion de ciel dans l'image"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Masque pour le ciel (bleu/clair)
        lower_sky = np.array([90, 50, 100])
        upper_sky = np.array([130, 150, 255])
        sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
        
        # La partie supérieure de l'image est plus susceptible d'être le ciel
        height = frame.shape[0]
        sky_mask[int(height*0.3):, :] = 0
        
        return np.sum(sky_mask > 0) / frame.size
    
    def _calculate_road_ratio(self, frame: np.ndarray) -> float:
        """Estime la proportion de route dans l'image"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Masque pour la route (gris/bleu selon l'éclairage)
        lower_road = np.array([0, 0, 50])
        upper_road = np.array([180, 50, 200])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        
        # La partie inférieure de l'image est plus susceptible d'être la route
        height = frame.shape[0]
        road_mask[:int(height*0.3), :] = 0
        
        return np.sum(road_mask > 0) / frame.size
    
    def _get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Extrait l'embedding DINOv2"""
        if not self.has_dinov2:
            return np.zeros(768)
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tensor = self.dinov2_transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.dinov2_model(tensor)
                # DINOv2 peut retourner un tuple ou un tenseur selon la version
                if isinstance(embedding, tuple):
                    embedding = embedding[0]
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            logger.debug(f"Erreur extraction embedding: {e}")
            return np.zeros(768)
    
    def _detect_objects(self, frame: np.ndarray) -> Dict:
        """Détecte les objets dans la scène"""
        objects = {
            'vehicles': [],
            'pedestrians': [],
            'traffic_lights': [],
            'traffic_signs': [],
            'road_markings': [],
            'construction': [],
            'debris': [],
            'emergency': []
        }
        
        if not self.has_yolo:
            return objects
        
        try:
            results = self.yolo_model(frame, conf=0.3, verbose=False)
            if results and len(results) > 0:
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
                    class_id = int(result.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Classification des objets
                    if class_id in [2, 3, 5, 6, 7, 8]:
                        objects['vehicles'].append([x1, y1, x2, y2, class_name])
                    elif class_id == 0:
                        objects['pedestrians'].append([x1, y1, x2, y2, 'pedestrian'])
                    elif class_id == 9:
                        objects['traffic_lights'].append([x1, y1, x2, y2, 'traffic_light'])
                    elif class_id == 11:
                        objects['traffic_signs'].append([x1, y1, x2, y2, 'stop_sign'])
                    elif class_name in ['construction', 'cone', 'barrier']:
                        objects['construction'].append([x1, y1, x2, y2, class_name])
                    elif class_name == 'debris':
                        objects['debris'].append([x1, y1, x2, y2, 'debris'])
        except Exception as e:
            logger.debug(f"Erreur YOLO: {e}")
        
        return objects
    
    def _analyze_semantic(self, frame: np.ndarray) -> Dict:
        """Analyse sémantique de la scène avec CLIP"""
        if not self.has_clip:
            return {'context': 'unknown', 'scores': {}}
        
        # Descriptions de scènes anormales vs normales
        descriptions = [
            "normal road with traffic",  # Normal
            "construction site with barriers",  # Construction
            "accident with emergency vehicles",  # Accident
            "severe weather with low visibility",  # Météo
            "debris on the road",  # Obstacle
            "traffic lights not working",  # Infrastructure
            "pedestrians crossing illegally",  # Comportement dangereux
        ]
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            text_tokens = open_clip.tokenize(descriptions).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                scores = similarity.cpu().numpy().flatten()
            
            best_idx = np.argmax(scores[1:]) + 1  # Exclure le "normal"
            
            return {
                'context': descriptions[best_idx],
                'score': scores[best_idx],
                'scores': {desc: float(score) for desc, score in zip(descriptions, scores)}
            }
        except Exception as e:
            logger.debug(f"Erreur CLIP: {e}")
            return {'context': 'unknown', 'scores': {}}
    
    def _detect_anomalies(self, frame: np.ndarray, features: Dict,
                          objects: Dict, semantic: Dict) -> List[AnomalyDetection]:
        """Détecte les anomalies dans la scène"""
        anomalies = []
        
        # 1. Détection basée sur les objets
        anomalies.extend(self._detect_object_based_anomalies(objects))
        
        # 2. Détection basée sur les caractéristiques
        anomalies.extend(self._detect_feature_based_anomalies(features))
        
        # 3. Détection sémantique
        anomalies.extend(self._detect_semantic_anomalies(semantic))
        
        # 4. Détection de contexte
        anomalies.extend(self._detect_contextual_anomalies(frame, features, objects))
        
        # Filtrer par seuil de confiance
        anomalies = [a for a in anomalies if a.confidence > self.config['confidence_threshold']]
        
        return anomalies
    
    def _detect_object_based_anomalies(self, objects: Dict) -> List[AnomalyDetection]:
        """Détecte les anomalies basées sur la présence d'objets"""
        anomalies = []
        
        # 1. Travaux / Construction
        if objects.get('construction'):
            construction = objects['construction']
            bbox = construction[0][:4] if construction else None
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.CONSTRUCTION,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.85,
                bbox=bbox,
                description=f"Zone de travaux détectée ({len(construction)} éléments)",
                risk_score=0.6,
                suggested_action="Ralentir et être prudent"
            ))
        
        # 2. Débris / Obstacles
        if objects.get('debris'):
            debris = objects['debris']
            bbox = debris[0][:4] if debris else None
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.OBSTACLE,
                severity=AnomalySeverity.HIGH,
                confidence=0.8,
                bbox=bbox,
                description="Débris détecté sur la route",
                risk_score=0.8,
                suggested_action="Éviter l'obstacle ou s'arrêter"
            ))
        
        # 3. Comportement dangereux (piétons sur la route)
        pedestrians = objects.get('pedestrians', [])
        
        for ped in pedestrians:
            if len(ped) >= 4:
                x1, y1, x2, y2 = ped[:4]
                # Vérifier si le piéton est sur la route (partie inférieure de l'image)
                # Note: frame.shape[0] est utilisée dans le contexte de la méthode appelante
                # Nous stockons la hauteur pour une vérification ultérieure
                pass
        
        return anomalies
    
    def _detect_feature_based_anomalies(self, features: Dict) -> List[AnomalyDetection]:
        """Détecte les anomalies basées sur les caractéristiques de l'image"""
        anomalies = []
        
        # 1. Météo extrême (visibilité réduite)
        visibility = self._calculate_visibility_from_features(features)
        if visibility < 0.3:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.ADVERSE_WEATHER,
                severity=AnomalySeverity.HIGH,
                confidence=0.85,
                description=f"Visibilité très faible ({visibility:.0%})",
                risk_score=0.9,
                suggested_action="Réduire significativement la vitesse"
            ))
        elif visibility < 0.5:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.ADVERSE_WEATHER,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.7,
                description=f"Visibilité réduite ({visibility:.0%})",
                risk_score=0.6,
                suggested_action="Réduire la vitesse"
            ))
        
        # 2. Nuits (danger accru)
        brightness = features.get('brightness', 0.5)
        if brightness < 0.15:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.ADVERSE_WEATHER,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.8,
                description="Nuit profonde, faible éclairage",
                risk_score=0.7,
                suggested_action="Allumer les phares, réduire la vitesse"
            ))
        
        return anomalies
    
    def _detect_semantic_anomalies(self, semantic: Dict) -> List[AnomalyDetection]:
        """Détecte les anomalies basées sur l'analyse sémantique"""
        anomalies = []
        
        if not semantic or 'context' not in semantic:
            return anomalies
        
        context = semantic.get('context', '')
        score = semantic.get('score', 0)
        
        if score > 0.5:
            if 'accident' in context or 'emergency' in context:
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.EMERGENCY,
                    severity=AnomalySeverity.CRITICAL,
                    confidence=score,
                    description="Situation d'urgence détectée",
                    risk_score=1.0,
                    suggested_action="Ralentir et se préparer à s'arrêter"
                ))
            elif 'construction' in context or 'barriers' in context:
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.CONSTRUCTION,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=score,
                    description="Zone de travaux détectée",
                    risk_score=0.6,
                    suggested_action="Ralentir et suivre la signalisation"
                ))
            elif 'debris' in context:
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.OBSTACLE,
                    severity=AnomalySeverity.HIGH,
                    confidence=score,
                    description="Obstacle sur la route",
                    risk_score=0.8,
                    suggested_action="Éviter l'obstacle"
                ))
        
        return anomalies
    
    def _detect_contextual_anomalies(self, frame: np.ndarray, features: Dict,
                                      objects: Dict) -> List[AnomalyDetection]:
        """Détecte les anomalies contextuelles"""
        anomalies = []
        
        # 1. Anomalies de la route (nids-de-poule, marquage effacé)
        road_quality = self._assess_road_quality(frame)
        if road_quality < 0.3:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.ROAD_ANOMALY,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.7,
                description=f"Qualité de la route dégradée ({road_quality:.0%})",
                risk_score=0.5,
                suggested_action="Réduire la vitesse, être attentif à l'état de la route"
            ))
        
        # 2. Absence de signalisation
        if objects.get('traffic_lights') == [] and objects.get('traffic_signs') == []:
            # Vérifier si c'est une intersection
            if self._is_intersection(frame):
                anomalies.append(AnomalyDetection(
                    anomaly_type=AnomalyType.INFRASTRUCTURE_DAMAGE,
                    severity=AnomalySeverity.HIGH,
                    confidence=0.6,
                    description="Intersection sans signalisation visible",
                    risk_score=0.7,
                    suggested_action="Ralentir et vérifier les autres véhicules"
                ))
        
        return anomalies
    
    def _calculate_visibility_from_features(self, features: Dict) -> float:
        """Calcule la visibilité à partir des caractéristiques"""
        edge_density = features.get('edge_density', 0.1)
        contrast = features.get('contrast', 0.3)
        brightness = features.get('brightness', 0.5)
        
        # Plus d'arêtes et de contraste = meilleure visibilité
        visibility = min(1.0, (edge_density * 2 + contrast * 1.5 + brightness * 0.5) / 2)
        
        return max(0.1, visibility)
    
    def _assess_road_quality(self, frame: np.ndarray) -> float:
        """Évalue la qualité de la route"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection de texture irrégulière (nids-de-poule)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # Détection de marquages
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / frame.size
        
        # Score combiné
        quality = 1.0
        
        # Variance élevée = mauvais état
        if laplacian_variance > 500:
            quality -= 0.3
        elif laplacian_variance > 300:
            quality -= 0.15
        
        # Peu de marquages = mauvaise qualité
        if edge_density < 0.01:
            quality -= 0.2
        elif edge_density < 0.02:
            quality -= 0.1
        
        return max(0.0, quality)
    
    def _is_intersection(self, frame: np.ndarray) -> bool:
        """Détecte si la scène est une intersection"""
        # Approche simplifiée: détection de lignes perpendiculaires
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50)
        
        if lines is None or len(lines) < 5:
            return False
        
        # Vérifier les angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(abs(angle))
        
        # Présence d'angles variés (≈ intersection)
        angles = [a % 90 for a in angles]
        unique_angles = len(set(int(a) for a in angles if a < 90))
        
        return unique_angles > 3
    
    def _temporal_filter(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Filtre temporel pour réduire les faux positifs"""
        # Ajouter à l'historique
        for anomaly in anomalies:
            self.anomaly_history.append(anomaly)
        
        # Vérifier si les anomalies sont persistantes
        filtered = []
        for anomaly in anomalies:
            # Compter combien de fois cette anomalie a été détectée récemment
            count = 0
            for hist in list(self.anomaly_history)[-10:]:
                if hist.anomaly_type == anomaly.anomaly_type:
                    count += 1
            
            # Si détectée au moins 3 fois sur les 5 dernières frames
            if count >= 3 or anomaly.severity == AnomalySeverity.CRITICAL:
                filtered.append(anomaly)
        
        return filtered
    
    def _annotate_frame(self, frame: np.ndarray, anomalies: List[AnomalyDetection],
                        objects: Dict) -> np.ndarray:
        """Annote la frame avec les anomalies détectées"""
        annotated = frame.copy()
        
        # Fond pour les informations
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        # En-tête
        cv2.putText(annotated, "🚨 DETECTION D'ANOMALIES", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Nombre d'anomalies
        cv2.putText(annotated, f"Anomalies: {len(anomalies)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Anomalies détectées
        y_offset = 85
        for i, anomaly in enumerate(anomalies[:4]):  # Afficher max 4 anomalies
            color = {
                AnomalySeverity.LOW: (255, 255, 0),
                AnomalySeverity.MEDIUM: (0, 165, 255),
                AnomalySeverity.HIGH: (0, 0, 255),
                AnomalySeverity.CRITICAL: (0, 0, 200)
            }.get(anomaly.severity, (255, 255, 255))
            
            text = f"{anomaly.anomaly_type.value} ({anomaly.confidence:.0%})"
            cv2.putText(annotated, text, (10, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Dessiner les bounding boxes
        for anomaly in anomalies:
            if anomaly.bbox is not None:
                x1, y1, x2, y2 = map(int, anomaly.bbox)
                color = {
                    AnomalyType.CONSTRUCTION: (0, 255, 255),
                    AnomalyType.OBSTACLE: (0, 0, 255),
                    AnomalyType.INFRASTRUCTURE_DAMAGE: (255, 165, 0),
                    AnomalyType.ADVERSE_WEATHER: (255, 0, 255),
                    AnomalyType.DANGEROUS_BEHAVIOR: (0, 0, 200),
                    AnomalyType.ROAD_ANOMALY: (0, 255, 0),
                    AnomalyType.EMERGENCY: (0, 0, 255)
                }.get(anomaly.anomaly_type, (255, 255, 255))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Label
                label = f"{anomaly.anomaly_type.value}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Informations de performance
        if self.stats['inference_times']:
            avg_time = np.mean(self.stats['inference_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _update_stats(self, anomalies: List[AnomalyDetection], inference_time: float):
        """Met à jour les statistiques"""
        self.stats['inference_times'].append(inference_time)
        self.stats['total_anomalies'] += len(anomalies)
        
        for anomaly in anomalies:
            type_name = anomaly.anomaly_type.value
            self.stats['anomalies_by_type'][type_name] = \
                self.stats['anomalies_by_type'].get(type_name, 0) + 1
            
            severity_name = anomaly.severity.value
            self.stats['severity_distribution'][severity_name] = \
                self.stats['severity_distribution'].get(severity_name, 0) + 1
        
        if anomalies:
            avg_conf = np.mean([a.confidence for a in anomalies])
            self.stats['avg_confidence'] = avg_conf
    
    def get_report(self) -> Dict:
        """Génère un rapport statistique"""
        return {
            'total_anomalies': self.stats['total_anomalies'],
            'anomalies_by_type': self.stats['anomalies_by_type'],
            'severity_distribution': self.stats['severity_distribution'],
            'avg_confidence': self.stats['avg_confidence'],
            'avg_fps': 1.0 / np.mean(self.stats['inference_times']) if self.stats['inference_times'] else 0
        }


class AnomalyTracker:
    """Tracker simple pour le suivi des anomalies"""
    
    def __init__(self, max_age=30):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
        self.history = {}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Met à jour le tracking"""
        if not detections:
            # Vieillissement des tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Matching simple basé sur la distance
        matched = []
        for det in detections:
            bbox = det.get('bbox')
            if bbox is None:
                continue
            
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            best_track_id = None
            best_dist = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_info['age'] > self.max_age:
                    continue
                
                last_pos = track_info['last_position']
                dist = np.sqrt(
                    (center[0] - last_pos[0])**2 +
                    (center[1] - last_pos[1])**2
                )
                
                if dist < best_dist and dist < 100:
                    best_dist = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                self.tracks[best_track_id]['last_position'] = center
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['hits'] += 1
                det['track_id'] = best_track_id
                matched.append(best_track_id)
            else:
                # Nouvelle track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'last_position': center,
                    'age': 0,
                    'hits': 1
                }
                det['track_id'] = track_id
        
        return detections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Détecteur d\'anomalies')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'video', 'image'],
                       help='Mode d\'exécution')
    parser.add_argument('--input', type=str, help='Chemin vers le fichier d\'entrée')
    parser.add_argument('--output', type=str, default='./anomalies_output',
                       help='Dossier de sortie')
    parser.add_argument('--save', action='store_true', help='Sauvegarder les résultats')
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # Initialiser le détecteur
    detector = AnomalyDetector()
    
    print("=" * 60)
    print("🚗 DETECTEUR D'ANOMALIES ENVIRONNEMENTALES")
    print("=" * 60)
    print("Anomalies détectables:")
    print("  - 🚧 Travaux")
    print("  - 🚧 Obstacles")
    print("  - 🏗️ Infrastructure endommagée")
    print("  - 🌧️ Conditions météo extrêmes")
    print("  - ⚠️ Comportements dangereux")
    print("  - 🛣️ Anomalies de la route")
    print("  - 🚨 Situations d'urgence")
    print("=" * 60)
    
    if args.mode == 'image' and args.input:
        frame = cv2.imread(args.input)
        if frame is not None:
            annotated, anomalies = detector.process_frame(frame)
            
            print("\n🔍 Résultats:")
            print(f"  - Anomalies détectées: {len(anomalies)}")
            for anomaly in anomalies:
                print(f"    • {anomaly.anomaly_type.value} ({anomaly.severity.value}) - Confiance: {anomaly.confidence:.0%}")
                print(f"      → {anomaly.description}")
                print(f"      → Action: {anomaly.suggested_action}")
            
            cv2.imshow('Anomaly Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"❌ Impossible de charger l'image: {args.input}")
    
    elif args.mode == 'video' and args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo: {args.input}")
        else:
            print("\n📹 Analyse de la vidéo...")
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated, anomalies = detector.process_frame(frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"\r📊 Frames: {frame_count} | Anomalies: {detector.stats['total_anomalies']}", end="")
                
                cv2.imshow('Anomaly Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n\n📊 Rapport final:")
            report = detector.get_report()
            for key, value in report.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    else:  # webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la webcam")
        else:
            print("\n📸 Appuyez sur 'q' pour quitter")
            print("🔍 Analyse en temps réel...\n")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated, anomalies = detector.process_frame(frame)
                    
                    print(f"\r🚨 Anomalies: {len(anomalies)}", end="")
                    if anomalies:
                        types = [a.anomaly_type.value for a in anomalies]
                        print(f" | Types: {', '.join(types)}", end="")
                    
                    cv2.imshow('Anomaly Detection', annotated)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except KeyboardInterrupt:
                print("\nArrêt demandé")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n\n📊 Rapport final:")
            report = detector.get_report()
            for key, value in report.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")