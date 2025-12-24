"""
chat.py - VLM API Interface for Billiards Strategy
支持多种VLM API（OpenAI GPT-4V, Claude, 等）
"""

import base64
import io
import json
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple
import re


class VLMChat:
    """VLM对话接口"""
    
    def __init__(self, provider='qwen', model='qwen3-vl-flash', api_key=None, base_url=None):
        """
        初始化VLM客户端
        
        Args:
            provider: 'openai', 'claude', 'qwen' (阿里云Qwen)
            model: 模型名称
            api_key: API密钥（如果不提供，从环境变量读取）
            base_url: API基础URL（用于兼容OpenAI的服务，如Qwen）
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
        
        # 获取API密钥
        if api_key is None:
            if provider == 'openai' or provider == 'qwen':
                api_key = os.getenv('OPENAI_API_KEY')
            elif provider == 'claude':
                api_key = os.getenv('ANTHROPIC_API_KEY')

        
        self.api_key = api_key
        
        # 初始化客户端
        if provider in ['openai', 'qwen']:
            try:
                from openai import OpenAI
                # Qwen使用OpenAI兼容的API
                if provider == 'qwen':
                    if base_url is None:
                        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    print(f"[VLMChat] Initialized Qwen client: {base_url}")
                    print(f"[VLMChat] Model: {model}")
                else:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    print(f"[VLMChat] Initialized OpenAI client with model {model}")
            except ImportError:
                print("[VLMChat] Warning: openai package not installed. Install with: pip install openai")
                self.client = None
        elif provider == 'claude':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                print(f"[VLMChat] Initialized Claude client with model {model}")
            except ImportError:
                print("[VLMChat] Warning: anthropic package not installed. Install with: pip install anthropic")
                self.client = None
        else:
            print(f"[VLMChat] Provider {provider} not supported yet")
            self.client = None
    
    def encode_image(self, image: Image.Image) -> str:
        """将PIL Image编码为base64字符串"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_strategy_from_image(self, image: Image.Image, 
                               my_remaining: int, 
                               enemy_remaining: int,
                               my_targets: List[str],
                               game_phase: str = 'mid',
                               return_raw_response: bool = False) -> Dict:
        """
        从图片获取VLM的战略建议
        
        Args:
            image: 台球对局图片
            my_remaining: 我方剩余球数
            enemy_remaining: 对方剩余球数
            my_targets: 我方目标球列表
            game_phase: 游戏阶段 'early', 'mid', 'end'
            
        Returns:
            Dict包含:
                - strategy: 'aggressive', 'conservative', 'defensive', 'positional'
                - target_priority: List[str] 目标球优先级
                - risk_tolerance: float 0-1
                - reasoning: str 推理过程
                - key_considerations: List[str] 关键考虑
        """
        
        # 构建提示词
        prompt = self._build_strategy_prompt(my_remaining, enemy_remaining, 
                                            my_targets, game_phase)
        
        # 调用VLM
        if self.provider in ['openai', 'qwen']:
            response = self._call_openai(image, prompt)
        elif self.provider == 'claude':
            response = self._call_claude(image, prompt)
        else:
            # 降级到规则基策略
            fallback = self._fallback_strategy(my_remaining, enemy_remaining, my_targets)
            if return_raw_response:
                return fallback, prompt, "Fallback strategy (no VLM call)"
            return fallback
        
        # 解析响应
        strategy = self._parse_strategy_response(response)
        
        # 如果需要返回原始响应
        if return_raw_response:
            return strategy, prompt, response
        
        return strategy
    
    def _build_strategy_prompt(self, my_remaining: int, enemy_remaining: int,
                               my_targets: List[str], game_phase: str) -> str:
        """构建战略提示词"""
        
        # 判断局势
        if my_remaining < enemy_remaining:
            situation = "leading"
        elif my_remaining == enemy_remaining:
            situation = "even"
        else:
            situation = "behind"
        
        prompt = f"""You are an expert billiards player analyzing this game situation.

**Current Situation:**
- My remaining balls: {my_remaining} (targets: {', '.join(my_targets)})
- Opponent's remaining balls: {enemy_remaining}
- Game phase: {game_phase}
- Position: {situation}

**Task:**
Analyze the image and provide strategic guidance for the next shot. Consider:
1. Ball positions and accessibility
2. Pocket opportunities
3. Position for next shot
4. Risk vs reward
5. Whether to play aggressive or conservative

**Respond in JSON format:**
{{
    "strategy": "<aggressive|conservative|defensive|positional>",
    "target_priority": ["<ball_id>", "<ball_id>", ...],
    "risk_tolerance": <0.0-1.0>,
    "reasoning": "<your detailed analysis>",
    "key_considerations": ["<point1>", "<point2>", ...]
}}

**Strategy definitions:**
- aggressive: Go for difficult shots, prioritize pocketing balls
- conservative: Play safe, controlled shots with high success rate
- defensive: Safety play, deny opponent opportunities
- positional: Focus on setting up next shot, even if no ball is pocketed

Provide ONLY the JSON, no additional text."""
        
        return prompt
    
    def _call_openai(self, image: Image.Image, prompt: str) -> str:
        """调用OpenAI GPT-4V"""
        if self.client is None:
            return ""
        
        try:
            # 编码图片
            base64_image = self.encode_image(image)
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[VLMChat] OpenAI API error: {e}")
            return ""
    
    def _call_claude(self, image: Image.Image, prompt: str) -> str:
        """调用Claude Vision"""
        if self.client is None:
            return ""
        
        try:
            # 编码图片
            base64_image = self.encode_image(image)
            
            # 调用API
            response = self.client.messages.create(
                model=self.model if 'claude' in self.model else "claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"[VLMChat] Claude API error: {e}")
            return ""
    
    def _parse_strategy_response(self, response: str) -> Dict:
        """解析VLM的JSON响应"""
        try:
            # 提取JSON（有时VLM会在JSON前后添加文本）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                strategy = json.loads(json_str)
                
                # 验证必需字段
                required_fields = ['strategy', 'target_priority', 'risk_tolerance']
                for field in required_fields:
                    if field not in strategy:
                        print(f"[VLMChat] Warning: Missing field '{field}' in VLM response")
                        strategy[field] = self._get_default_value(field)
                
                # 验证strategy值
                valid_strategies = ['aggressive', 'conservative', 'defensive', 'positional']
                if strategy['strategy'] not in valid_strategies:
                    print(f"[VLMChat] Invalid strategy '{strategy['strategy']}', using 'conservative'")
                    strategy['strategy'] = 'conservative'
                
                # 验证risk_tolerance范围
                strategy['risk_tolerance'] = max(0.0, min(1.0, float(strategy['risk_tolerance'])))
                
                print(f"[VLMChat] Parsed strategy: {strategy['strategy']}, "
                      f"risk={strategy['risk_tolerance']:.2f}")
                
                return strategy
            else:
                print("[VLMChat] No JSON found in response, using fallback")
                return self._get_default_strategy()
                
        except json.JSONDecodeError as e:
            print(f"[VLMChat] JSON parse error: {e}")
            return self._get_default_strategy()
        except Exception as e:
            print(f"[VLMChat] Unexpected error parsing response: {e}")
            return self._get_default_strategy()
    
    def _get_default_value(self, field: str):
        """获取字段的默认值"""
        defaults = {
            'strategy': 'conservative',
            'target_priority': [],
            'risk_tolerance': 0.5,
            'reasoning': 'Default strategy',
            'key_considerations': []
        }
        return defaults.get(field, None)
    
    def _get_default_strategy(self) -> Dict:
        """获取默认策略"""
        return {
            'strategy': 'conservative',
            'target_priority': [],
            'risk_tolerance': 0.5,
            'reasoning': 'Default fallback strategy',
            'key_considerations': []
        }
    
    def _fallback_strategy(self, my_remaining: int, enemy_remaining: int, 
                          my_targets: List[str]) -> Dict:
        """降级策略（当VLM不可用时）"""
        
        # 基于规则的简单策略
        if my_remaining < enemy_remaining:
            # 领先，保守打法
            strategy = 'conservative'
            risk = 0.3
        elif my_remaining == enemy_remaining:
            # 平局，平衡打法
            strategy = 'balanced'
            risk = 0.5
        else:
            # 落后，激进打法
            strategy = 'aggressive'
            risk = 0.7
        
        return {
            'strategy': strategy,
            'target_priority': my_targets,  # 按顺序
            'risk_tolerance': risk,
            'reasoning': 'Rule-based fallback strategy',
            'key_considerations': ['No VLM available, using heuristics']
        }


def test_vlm_chat():
    """测试VLM对话功能"""
    # 创建测试图片
    from drawer import BilliardsDrawer
    import pooltool as pt
    
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
    }
    
    drawer = BilliardsDrawer()
    img = drawer.draw_table_state(balls, my_targets=['1'], enemy_targets=[])
    
    # 测试VLM（需要API key）
    vlm = VLMChat(provider='qwen')
    strategy = vlm.get_strategy_from_image(
        img, 
        my_remaining=1, 
        enemy_remaining=3,
        my_targets=['1'],
        game_phase='end'
    )
    
    print("VLM Strategy:", json.dumps(strategy, indent=2))


if __name__ == "__main__":
    test_vlm_chat()

