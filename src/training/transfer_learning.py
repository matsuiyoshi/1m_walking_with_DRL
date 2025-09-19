#!/usr/bin/env python3
"""
転移学習制御クラス
Transfer Learning Controller for Curriculum Learning
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


class TransferLearningController:
    """転移学習制御クラス"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        初期化
        
        Args:
            config: 転移学習設定
            output_dir: 出力ディレクトリ
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 転移学習設定
        self.enabled = config.get('enabled', True)
        self.load_previous_model = config.get('stage_transition', {}).get('load_previous_model', True)
        self.lr_adjustment = config.get('stage_transition', {}).get('lr_adjustment', {})
        self.freeze_layers = config.get('stage_transition', {}).get('freeze_layers', {})
        
    def load_previous_stage_model(self, model_path: Path, agent) -> bool:
        """
        前段階のモデルを読み込み
        
        Args:
            model_path: 前段階のモデルファイルパス
            agent: PPOエージェント
            
        Returns:
            読み込み成功かどうか
        """
        if not self.enabled or not self.load_previous_model:
            self.logger.info("転移学習が無効または前段階モデルの読み込みが無効")
            return False
            
        if not model_path.exists():
            self.logger.warning(f"前段階モデルが見つかりません: {model_path}")
            return False
            
        try:
            # モデルを読み込み
            checkpoint = torch.load(model_path, map_location=agent.device)
            
            # エージェントの状態を復元
            if 'actor_state_dict' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.logger.info("Actorの重みを読み込みました")
                
            if 'critic_state_dict' in checkpoint:
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.logger.info("Criticの重みを読み込みました")
                
            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Optimizerの状態を読み込みました")
                
            if 'scheduler_state_dict' in checkpoint and agent.scheduler is not None:
                agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Schedulerの状態を読み込みました")
                
            # 学習率を調整
            self._adjust_learning_rate(agent)
            
            # 層の凍結
            self._freeze_layers(agent)
            
            self.logger.info(f"前段階モデルを正常に読み込みました: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"前段階モデルの読み込みに失敗: {e}")
            return False
            
    def _adjust_learning_rate(self, agent):
        """学習率を調整"""
        if not self.lr_adjustment:
            return
            
        factor = self.lr_adjustment.get('factor', 0.5)
        original_lr = agent.hyperparams['learning_rate']
        new_lr = original_lr * factor
        
        # オプティマイザーの学習率を更新
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        self.logger.info(f"学習率を調整しました: {original_lr} -> {new_lr}")
        
        # スケジューラーがあれば更新
        if agent.scheduler is not None:
            if hasattr(agent.scheduler, 'base_lrs'):
                agent.scheduler.base_lrs = [new_lr]
            elif hasattr(agent.scheduler, 'lr_lambda'):
                # カスタムスケジューラーの場合
                pass
                
    def _freeze_layers(self, agent):
        """指定された層を凍結"""
        if not self.freeze_layers.get('enabled', False):
            return
            
        freeze_layers = self.freeze_layers.get('freeze_layers', [])
        if not freeze_layers:
            return
            
        # Actorの層を凍結
        for name, param in agent.actor.named_parameters():
            for layer_name in freeze_layers:
                if layer_name in name:
                    param.requires_grad = False
                    self.logger.info(f"Actor層を凍結: {name}")
                    
        # Criticの層を凍結
        for name, param in agent.critic.named_parameters():
            for layer_name in freeze_layers:
                if layer_name in name:
                    param.requires_grad = False
                    self.logger.info(f"Critic層を凍結: {name}")
                    
    def save_stage_model(self, agent, stage: int, additional_info: Optional[Dict[str, Any]] = None):
        """
        段階のモデルを保存
        
        Args:
            agent: PPOエージェント
            stage: 段階番号
            additional_info: 追加情報
        """
        if not self.enabled:
            return
            
        try:
            # チェックポイントを作成
            checkpoint = {
                'stage': stage,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'hyperparams': agent.hyperparams
            }
            
            # スケジューラーの状態も保存
            if agent.scheduler is not None:
                checkpoint['scheduler_state_dict'] = agent.scheduler.state_dict()
                
            # 追加情報を保存
            if additional_info:
                checkpoint.update(additional_info)
                
            # モデルファイルを保存
            model_path = self.output_dir / f"stage_{stage}_model.pth"
            torch.save(checkpoint, model_path)
            
            self.logger.info(f"段階 {stage} のモデルを保存しました: {model_path}")
            
        except Exception as e:
            self.logger.error(f"モデルの保存に失敗: {e}")
            
    def get_transfer_learning_info(self) -> Dict[str, Any]:
        """転移学習の情報を取得"""
        return {
            'enabled': self.enabled,
            'load_previous_model': self.load_previous_model,
            'lr_adjustment': self.lr_adjustment,
            'freeze_layers': self.freeze_layers
        }
        
    def create_transfer_summary(self, stage: int, success: bool, improvement: float) -> Dict[str, Any]:
        """
        転移学習の要約を作成
        
        Args:
            stage: 段階番号
            success: 転移成功かどうか
            improvement: 改善度
            
        Returns:
            転移学習要約
        """
        return {
            'stage': stage,
            'transfer_success': success,
            'improvement': improvement,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
