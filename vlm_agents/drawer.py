"""
drawer.py - Billiards Table Visualization for VLM
生成适合VLM理解的台球对局图片
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免tkinter错误

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io


class BilliardsDrawer:
    """绘制台球对局的俯视图"""
    
    def __init__(self):
        # 默认尺寸（但会从table对象动态读取）
        self.table_width = 2.24
        self.table_height = 1.12
        self.ball_radius = 0.028575
        
        # 标准台球颜色
        self.ball_colors = {
            'cue': '#FFFFFF',
            '1': '#FFFF00', '2': '#0000FF', '3': '#FF0000', '4': '#800080',
            '5': '#FFA500', '6': '#008000', '7': '#8B0000', '8': '#000000',
            '9': '#FFFF00', '10': '#0000FF', '11': '#FF0000', '12': '#800080',
            '13': '#FFA500', '14': '#008000', '15': '#8B0000'
        }
    
    def _get_table_dimensions(self, table):
        """从table对象获取实际尺寸"""
        if table is None:
            return self.table_width, self.table_height
        
        # 尝试读取实际尺寸
        width = getattr(table, 'w', self.table_width)
        height = getattr(table, 'l', self.table_height)  # l是长度（Y方向）
        
        return width, height
        
    def get_ball_position(self, ball):
        """获取球的2D位置（兼容不同的ball对象结构）"""
        try:
            # 尝试标准方式
            if hasattr(ball, 'state') and hasattr(ball.state, 'rvw'):
                pos = ball.state.rvw[0][:2]
                return tuple(float(x) for x in pos)
            # 尝试其他可能的属性
            elif hasattr(ball, 'xyz'):
                return (float(ball.xyz[0]), float(ball.xyz[1]))
            elif hasattr(ball, 'pos'):
                return (float(ball.pos[0]), float(ball.pos[1]))
            else:
                print(f"[Drawer] Warning: Unknown ball structure: {type(ball)}, {dir(ball)}")
                return (0, 0)
        except Exception as e:
            print(f"[Drawer] Error getting ball position: {e}")
            return (0, 0)
    
    def draw_table_state(self, balls, my_targets, enemy_targets=None, 
                        title="Billiards Game State", 
                        annotate=True, table=None) -> Image.Image:
        """
        绘制完整的台球对局状态
        
        Args:
            balls: 球的字典 {ball_id: ball_object}
            my_targets: 我方目标球列表
            enemy_targets: 对方目标球列表（可选）
            title: 图片标题
            annotate: 是否添加文字标注
            table: Table对象（用于获取实际尺寸）
            
        Returns:
            PIL Image对象
        """
        # 获取实际台球桌尺寸
        table_width, table_height = self._get_table_dimensions(table)
        
        # 根据实际尺寸计算图形大小（保持比例，竖屏显示）
        # width ≈ 1.0, height ≈ 2.0
        aspect_ratio = table_height / table_width  # 约为 2.0
        fig_height = 16
        fig_width = fig_height / aspect_ratio  # 约为 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # 设置坐标轴范围（使用实际尺寸）
        margin = 0.1
        ax.set_xlim(-margin, table_width + margin)
        ax.set_ylim(-margin, table_height + margin)
        ax.set_aspect('equal')
        
        # 设置背景色
        ax.set_facecolor('#0B6623')  # 台球桌绿色
        fig.patch.set_facecolor('#2C1810')  # 深棕色背景
        
        # 绘制球台边框（使用实际尺寸）
        table_rect = patches.Rectangle(
            (0, 0), table_width, table_height,
            linewidth=4, edgecolor='#8B4513', facecolor='#0B6623', zorder=1
        )
        ax.add_patch(table_rect)
        
        # 绘制袋口（6个标准袋口位置）
        # 竖屏布局：短边(x轴)2个洞，长边(y轴)3个洞
        # 4个角袋 + 2个中袋（在长边y轴的中点）
        pocket_positions = [
            (0, 0), (table_width, 0),  # 底边两个角袋
            (0, table_height/2), (table_width, table_height/2),  # 长边中袋（左右）
            (0, table_height), (table_width, table_height)  # 顶边两个角袋
        ]
        pocket_radius = 0.06
        for px, py in pocket_positions:
            pocket = plt.Circle((px, py), pocket_radius, 
                              facecolor='#1a1a1a', edgecolor='#8B4513', 
                              linewidth=2, zorder=10)
            ax.add_patch(pocket)
        
        # 统计球的信息
        my_remaining_balls = []
        enemy_remaining_balls = []
        
        # 调试：记录球的数量和位置
        active_balls = []
        
        # 绘制球
        for ball_id, ball in balls.items():
            # 检查球是否已进袋
            is_pocketed = False
            if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                is_pocketed = (ball.state.s == 4)
            
            if is_pocketed:  # 已进袋，跳过
                continue
            
            pos = self.get_ball_position(ball)
            active_balls.append((ball_id, pos))
            
            color = self.ball_colors.get(ball_id, '#CCCCCC')
            
            # 判断球的归属
            is_my_target = ball_id in my_targets
            is_enemy_target = enemy_targets and ball_id in enemy_targets
            is_cue = ball_id == 'cue'
            is_eight = ball_id == '8'
            
            if is_my_target:
                my_remaining_balls.append(ball_id)
            elif is_enemy_target:
                enemy_remaining_balls.append(ball_id)
            
            # 绘制球体（确保球的大小和颜色正确）
            ball_radius_display = 0.04  # 显示半径（适合2.24x1.12的球桌）
            
            if is_cue:
                # 白球特殊标记（红色边框）
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='red', 
                                        linewidth=3, zorder=100)
            elif is_my_target:
                # 我方目标球（绿色边框）
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='lime', 
                                        linewidth=2.5, zorder=90)
            elif is_enemy_target:
                # 对方目标球（橙色边框）
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='orange', 
                                        linewidth=2.5, zorder=90)
            elif is_eight:
                # 8号球（紫色边框）
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='purple', 
                                        linewidth=2.5, zorder=90)
            else:
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='white', 
                                        linewidth=1.5, zorder=80)
            
            ax.add_patch(ball_circle)
            
            # 添加球号（增大字号以便清晰显示）
            if ball_id != 'cue':
                text_color = 'white' if ball_id == '8' else 'black'
                ax.text(pos[0], pos[1], ball_id, 
                       fontsize=14, fontweight='bold',
                       ha='center', va='center', 
                       color=text_color, zorder=200)
        
        # 调试输出
        if len(active_balls) > 0:
            print(f"[Drawer] Drawing {len(active_balls)} balls:")
            for bid, pos in active_balls[:5]:  # 只打印前5个
                print(f"  Ball {bid}: pos=({pos[0]:.3f}, {pos[1]:.3f})")
        
        # 添加图例和信息（调整位置避免遮挡）
        if annotate:
            info_text = f"My Targets ({len(my_remaining_balls)}): {', '.join(my_remaining_balls) if my_remaining_balls else 'None'}\n"
            if enemy_targets:
                info_text += f"Enemy Targets ({len(enemy_remaining_balls)}): {', '.join(enemy_remaining_balls) if enemy_remaining_balls else 'None'}"
            
            # 将信息放在球桌下方（增大字号）
            ax.text(table_width/2, -0.08, info_text,
                   fontsize=15, ha='center', color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, pad=0.6))
            
            # 图例放在右上角外侧（避免与球桌重叠）
            legend_elements = [
                patches.Patch(facecolor='white', edgecolor='red', linewidth=3, label='Cue Ball'),
                patches.Patch(facecolor='gray', edgecolor='lime', linewidth=3, label='My Targets'),
                patches.Patch(facecolor='gray', edgecolor='orange', linewidth=3, label='Enemy Targets'),
                patches.Patch(facecolor='black', edgecolor='purple', linewidth=3, label='8-Ball')
            ]
            ax.legend(handles=legend_elements, 
                     loc='upper left',
                     bbox_to_anchor=(1.02, 1),  # 放在右侧外部
                     fontsize=13, 
                     framealpha=0.95,
                     handlelength=2,  # 图例标记长度
                     handleheight=1.5)  # 图例标记高度
        
        ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
        ax.axis('off')  # 关闭坐标轴
        
        plt.tight_layout()
        
        # 转换为PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image = Image.open(buf).copy()  # 复制以避免buffer问题
        plt.close(fig)
        
        return image
    
    def draw_with_suggested_shot(self, balls, my_targets, suggested_target=None, 
                                suggested_direction=None, enemy_targets=None, table=None) -> Image.Image:
        """
        绘制带有建议shot的对局图
        
        Args:
            suggested_target: 建议击打的目标球ID
            suggested_direction: 建议的方向（phi角度）
            table: Table对象（用于获取实际尺寸）
        """
        # 获取实际台球桌尺寸
        table_width, table_height = self._get_table_dimensions(table)
        
        # 根据实际尺寸计算图形大小（竖屏显示）
        aspect_ratio = table_height / table_width
        fig_height = 16
        fig_width = fig_height / aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # 设置坐标轴范围（使用实际尺寸）
        margin = 0.1
        ax.set_xlim(-margin, table_width + margin)
        ax.set_ylim(-margin, table_height + margin)
        ax.set_aspect('equal')
        
        # 背景色
        ax.set_facecolor('#0B6623')
        fig.patch.set_facecolor('#2C1810')
        
        # 球台边框（使用实际尺寸）
        table_rect = patches.Rectangle(
            (0, 0), table_width, table_height,
            linewidth=4, edgecolor='#8B4513', facecolor='#0B6623', zorder=1
        )
        ax.add_patch(table_rect)
        
        # 袋口（竖屏布局：短边2个洞，长边3个洞）
        pocket_positions = [
            (0, 0), (table_width, 0),  # 底边两角
            (0, table_height/2), (table_width, table_height/2),  # 长边中袋
            (0, table_height), (table_width, table_height)  # 顶边两角
        ]
        pocket_radius = 0.06
        for px, py in pocket_positions:
            pocket = plt.Circle((px, py), pocket_radius, 
                              facecolor='#1a1a1a', edgecolor='#8B4513', 
                              linewidth=2, zorder=10)
            ax.add_patch(pocket)
        
        cue_pos = None
        target_pos = None
        ball_radius_display = 0.04
        
        # 绘制球
        for ball_id, ball in balls.items():
            # 检查球是否已进袋
            is_pocketed = False
            if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                is_pocketed = (ball.state.s == 4)
            
            if is_pocketed:
                continue
            
            pos = self.get_ball_position(ball)
            color = self.ball_colors.get(ball_id, '#CCCCCC')
            
            if ball_id == 'cue':
                cue_pos = pos
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='red', 
                                        linewidth=3, zorder=100)
            elif ball_id == suggested_target:
                target_pos = pos
                # 高亮建议目标
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='yellow', 
                                        linewidth=4, zorder=100)
            elif ball_id in my_targets:
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='lime', 
                                        linewidth=2.5, zorder=90)
            elif enemy_targets and ball_id in enemy_targets:
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='orange', 
                                        linewidth=2.5, zorder=90)
            else:
                ball_circle = plt.Circle(pos, ball_radius_display, 
                                        facecolor=color, edgecolor='white', 
                                        linewidth=1.5, zorder=80)
            
            ax.add_patch(ball_circle)
            
            if ball_id != 'cue':
                text_color = 'white' if ball_id == '8' else 'black'
                ax.text(pos[0], pos[1], ball_id, 
                       fontsize=14, fontweight='bold',
                       ha='center', va='center', color=text_color, zorder=200)
        
        # 绘制建议shot的箭头
        if cue_pos is not None and suggested_direction is not None:
            phi_rad = np.radians(suggested_direction)
            arrow_length = 0.25
            dx = arrow_length * np.cos(phi_rad)
            dy = arrow_length * np.sin(phi_rad)
            
            ax.arrow(cue_pos[0], cue_pos[1], dx, dy,
                    head_width=0.06, head_length=0.05, 
                    fc='cyan', ec='cyan', linewidth=3, 
                    zorder=150, alpha=0.9)
            
            ax.text(cue_pos[0] + dx * 1.4, cue_pos[1] + dy * 1.4,
                   f'Suggested\nφ={suggested_direction:.1f}°',
                   fontsize=11, color='cyan', fontweight='bold',
                   ha='center', zorder=200,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # 如果有目标球，绘制连线
        if cue_pos is not None and target_pos is not None:
            ax.plot([cue_pos[0], target_pos[0]], 
                   [cue_pos[1], target_pos[1]],
                   'y--', linewidth=2.5, alpha=0.7, zorder=50)
        
        ax.set_title('Suggested Shot', fontsize=16, fontweight='bold', 
                    color='white', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image = Image.open(buf).copy()
        plt.close(fig)
        
        return image


def test_drawer():
    """测试绘图功能"""
    import pooltool as pt
    
    # 创建简单测试场景
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
    }
    
    drawer = BilliardsDrawer()
    img = drawer.draw_table_state(balls, my_targets=['1'], 
                                  enemy_targets=['9'],
                                  title="Test Game State")

    img.save("/home/yuhc/data/AI_project/AI3603-Billiards/vlm_agents/test_billiards.png")
    print("Test image saved to /test_billiards.png")


if __name__ == "__main__":
    test_drawer()

