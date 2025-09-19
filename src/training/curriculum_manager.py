#!/usr/bin/env python3
"""
段階的学習管理クラス
Curriculum Learning Manager for Bittle Walking
"""

import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class CurriculumLearningManager:
    """段階的学習管理クラス"""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        初期化
        
        Args:
            config_path: 段階的学習設定ファイルのパス
            output_dir: 出力ディレクトリ
        """
        self.config_path = Path(config_path)
        self.base_output_dir = Path(output_dir)
        
        # 学習実行ごとのディレクトリを作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.learning_run_dir = self.base_output_dir / f"learning_run_{timestamp}"
        self.learning_run_dir.mkdir(parents=True, exist_ok=True)
        
        # 各段階のディレクトリを作成
        self.stage_dirs = {}
        self._create_stage_directories()
        
        # 現在の出力ディレクトリを第1段階に設定
        self.output_dir = self.stage_dirs[1]
        
        # 設定読み込み
        self._load_config()
        
        # 段階管理
        self.current_stage = 1
        self.stage_start_time = time.time()
        self.total_timesteps = 0
        self.stage_timesteps = 0
        
        # ログ設定
        self._setup_logging()
        
        # 段階統計
        self.stage_stats = {
            'episode_lengths': [],
            'rewards': [],
            'forward_distances': [],
            'success_count': 0,
            'total_episodes': 0
        }
    
    def _create_stage_directories(self):
        """各段階のディレクトリを作成"""
        stage_names = {
            1: "balance",
            2: "short_walk", 
            3: "medium_walk",
            4: "long_walk"
        }
        
        for stage_num, stage_name in stage_names.items():
            stage_dir = self.learning_run_dir / f"stage_{stage_num}_{stage_name}"
            stage_dir.mkdir(exist_ok=True)
            self.stage_dirs[stage_num] = stage_dir
        
    def _load_config(self):
        """設定ファイルを読み込み"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.curriculum_config = self.config['curriculum']
        self.transfer_config = self.config['transfer_learning']
        self.monitoring_config = self.config['monitoring']
        
    def _setup_logging(self):
        """ログ設定"""
        log_file = self.learning_run_dir / "curriculum_learning.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_current_stage_config(self) -> Dict[str, Any]:
        """現在の段階の設定を取得"""
        stage_key = f"stage_{self.current_stage}_{self._get_stage_name()}"
        return self.curriculum_config['stages'][stage_key]
        
    def _get_stage_name(self) -> str:
        """段階名を取得"""
        stage_names = {
            1: "balance",
            2: "short_walk", 
            3: "medium_walk",
            4: "long_walk"
        }
        return stage_names[self.current_stage]
        
    def get_stage_name_display(self) -> str:
        """表示用段階名を取得"""
        stage_config = self.get_current_stage_config()
        return stage_config['name']
        
    def create_stage_config(self, stage_config: Dict[str, Any]) -> Path:
        """段階用の設定ファイルを作成"""
        # ベース設定を読み込み
        base_config_path = Path(__file__).parent.parent.parent / "config" / "training_config_12h_improved.yaml"
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # 段階設定を適用
        if 'environment' in stage_config:
            # 環境設定の更新
            if 'corridor' in stage_config['environment']:
                if 'environment' not in base_config:
                    base_config['environment'] = {}
                base_config['environment']['corridor'] = stage_config['environment']['corridor']
            if 'episode' in stage_config['environment']:
                if 'training' not in base_config:
                    base_config['training'] = {}
                base_config['training']['episode'] = stage_config['environment']['episode']
                
        if 'hyperparameters' in stage_config:
            # ハイパーパラメータの更新
            base_config['algorithm']['hyperparameters'].update(stage_config['hyperparameters'])
            
        if 'reward' in stage_config:
            # 報酬設定の更新
            base_config['environment']['reward'] = stage_config['reward']
            
        if 'video_recording' in stage_config:
            # 動画録画設定の更新
            self.logger.info(f"段階 {self.current_stage} の動画録画設定を適用: {stage_config['video_recording']}")
            base_config['video_recording'] = stage_config['video_recording']
            
        # 段階用設定ファイルを保存
        stage_config_path = self.output_dir / f"stage_{self.current_stage}_config.yaml"
        with open(stage_config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
            
        return stage_config_path
        
    def get_previous_model_path(self) -> Optional[Path]:
        """前段階のモデルパスを取得"""
        if self.current_stage <= 1:
            return None
            
        previous_model_path = self.output_dir / f"stage_{self.current_stage-1}_model.pth"
        if previous_model_path.exists():
            return previous_model_path
        return None
        
    def check_stage_completion(self) -> bool:
        """段階完了条件をチェック"""
        stage_key = f"stage_{self.current_stage}_complete"
        criteria = self.transfer_config['stage_criteria']
        
        if stage_key not in criteria:
            self.logger.warning(f"段階 {self.current_stage} の完了条件が定義されていません")
            return False
            
        completion_criteria = criteria[stage_key]
        
        # 統計の計算
        if len(self.stage_stats['episode_lengths']) < 50:
            self.logger.info(f"統計データ不足: {len(self.stage_stats['episode_lengths'])}/50 エピソード")
            return False
            
        # 最近の統計を計算
        recent_lengths = self.stage_stats['episode_lengths'][-100:]
        recent_rewards = self.stage_stats['rewards'][-100:]
        recent_distances = self.stage_stats['forward_distances'][-100:]
        
        avg_length = sum(recent_lengths) / len(recent_lengths)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_distance = sum(recent_distances) / len(recent_distances)
        success_rate = self.stage_stats['success_count'] / max(self.stage_stats['total_episodes'], 1)
        
        self.logger.info(f"段階 {self.current_stage} 統計:")
        self.logger.info(f"  平均エピソード長: {avg_length:.2f}s")
        self.logger.info(f"  平均報酬: {avg_reward:.2f}")
        self.logger.info(f"  平均前進距離: {avg_distance:.2f}m")
        self.logger.info(f"  成功率: {success_rate:.2%}")
        
        # 条件チェック
        if 'min_episode_length' in completion_criteria:
            if avg_length < completion_criteria['min_episode_length']:
                self.logger.info(f"エピソード長不足: {avg_length:.2f} < {completion_criteria['min_episode_length']}")
                return False
                
        if 'min_stability_reward' in completion_criteria:
            if avg_reward < completion_criteria['min_stability_reward']:
                self.logger.info(f"安定性報酬不足: {avg_reward:.2f} < {completion_criteria['min_stability_reward']}")
                return False
                
        if 'min_forward_distance' in completion_criteria:
            if avg_distance < completion_criteria['min_forward_distance']:
                self.logger.info(f"前進距離不足: {avg_distance:.2f} < {completion_criteria['min_forward_distance']}")
                return False
                
        if 'success_rate' in completion_criteria:
            if success_rate < completion_criteria['success_rate']:
                self.logger.info(f"成功率不足: {success_rate:.2%} < {completion_criteria['success_rate']:.2%}")
                return False
                
        self.logger.info(f"段階 {self.current_stage} 完了条件を満たしました！")
        return True
        
    def transition_to_next_stage(self) -> bool:
        """次の段階に移行"""
        if self.current_stage >= 4:
            self.logger.info("全段階完了！")
            return False
            
        self.current_stage += 1
        self.stage_start_time = time.time()
        self.stage_timesteps = 0
        
        # 現在の出力ディレクトリを次の段階に更新
        self.output_dir = self.stage_dirs[self.current_stage]
        
        # 段階統計をリセット
        self.stage_stats = {
            'episode_lengths': [],
            'rewards': [],
            'forward_distances': [],
            'success_count': 0,
            'total_episodes': 0
        }
        
        self.logger.info(f"段階 {self.current_stage} に移行: {self.get_stage_name_display()} -> {self.output_dir}")
        return True
        
    def update_episode_stats(self, episode_length: float, reward: float, forward_distance: float, success: bool):
        """エピソード統計を更新"""
        self.stage_stats['episode_lengths'].append(episode_length)
        self.stage_stats['rewards'].append(reward)
        self.stage_stats['forward_distances'].append(forward_distance)
        self.stage_stats['total_episodes'] += 1
        
        if success:
            self.stage_stats['success_count'] += 1
            
    def get_stage_progress(self) -> Dict[str, Any]:
        """段階の進捗情報を取得"""
        stage_config = self.get_current_stage_config()
        elapsed_time = time.time() - self.stage_start_time
        
        return {
            'current_stage': self.current_stage,
            'stage_name': self.get_stage_name_display(),
            'elapsed_time': elapsed_time,
            'stage_timesteps': self.stage_timesteps,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.stage_stats['total_episodes'],
            'success_rate': self.stage_stats['success_count'] / max(self.stage_stats['total_episodes'], 1),
            'avg_episode_length': sum(self.stage_stats['episode_lengths'][-100:]) / max(len(self.stage_stats['episode_lengths'][-100:]), 1),
            'avg_reward': sum(self.stage_stats['rewards'][-100:]) / max(len(self.stage_stats['rewards'][-100:]), 1),
            'avg_forward_distance': sum(self.stage_stats['forward_distances'][-100:]) / max(len(self.stage_stats['forward_distances'][-100:]), 1)
        }
        
    def should_check_completion(self) -> bool:
        """完了条件をチェックすべきかどうか"""
        check_interval = self.monitoring_config['stage_progress']['check_interval']
        return self.stage_timesteps % check_interval == 0
        
    def is_final_stage(self) -> bool:
        """最終段階かどうか"""
        return self.current_stage >= 4
        
    def save_stage_summary(self):
        """段階の要約を保存"""
        summary = {
            'stage': self.current_stage,
            'stage_name': self.get_stage_name_display(),
            'completion_time': datetime.now().isoformat(),
            'total_timesteps': self.stage_timesteps,
            'total_episodes': self.stage_stats['total_episodes'],
            'success_rate': self.stage_stats['success_count'] / max(self.stage_stats['total_episodes'], 1),
            'avg_episode_length': sum(self.stage_stats['episode_lengths']) / max(len(self.stage_stats['episode_lengths']), 1),
            'avg_reward': sum(self.stage_stats['rewards']) / max(len(self.stage_stats['rewards']), 1),
            'avg_forward_distance': sum(self.stage_stats['forward_distances']) / max(len(self.stage_stats['forward_distances']), 1)
        }
        
        summary_path = self.output_dir / f"stage_{self.current_stage}_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
            
        self.logger.info(f"段階 {self.current_stage} の要約を保存しました: {summary_path}")
