"""
Godot TCP Client - 与 Godot 游戏引擎通信的 Python 端
连接 Godot 的 NetworkManager (TCP Server:11008)
发送动作指令，接收游戏状态
"""

import socket
import json
import random
import time
from typing import Optional


class GodotClient:
    """TCP 客户端，连接 Godot 游戏引擎"""
    
    # 6个离散动作：上移/下移/左移/右移/攻击/待机
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_ATTACK = 4
    ACTION_IDLE = 5
    NUM_ACTIONS = 6
    
    def __init__(self, host: str = "127.0.0.1", port: int = 11008):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self._recv_buffer = ""
    
    def connect(self, timeout: float = 30.0) -> bool:
        """连接到 Godot TCP Server"""
        print(f"[GodotClient] 尝试连接 {self.host}:{self.port} ...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect((self.host, self.port))
                self.connected = True
                self.sock.settimeout(0.1)  # 非阻塞超时
                print(f"[GodotClient] 已连接到 Godot ({self.host}:{self.port})")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                if self.sock:
                    self.sock.close()
                time.sleep(1.0)
                continue
        
        print(f"[GodotClient] 连接超时 ({timeout}s)")
        return False
    
    def disconnect(self):
        """断开连接"""
        if self.sock:
            self.sock.close()
            self.sock = None
        self.connected = False
        print("[GodotClient] 已断开连接")
    
    def send_actions(self, actions: list[int]) -> bool:
        """发送动作指令给 Godot
        
        Args:
            actions: 长度为4的列表，每个元素为0-5的整数
            
        Returns:
            是否发送成功
        """
        if not self.connected or not self.sock:
            return False
        
        msg = {"type": "action", "actions": actions}
        return self._send_json(msg)
    
    def send_reset(self) -> bool:
        """发送重置指令"""
        if not self.connected or not self.sock:
            return False
        
        msg = {"type": "reset"}
        return self._send_json(msg)
    
    def send_ping(self) -> bool:
        """发送心跳"""
        if not self.connected or not self.sock:
            return False
        
        msg = {"type": "ping"}
        return self._send_json(msg)
    
    def receive_state(self) -> Optional[dict]:
        """接收游戏状态（非阻塞）
        
        Returns:
            游戏状态字典，或 None（如果没有新数据）
        """
        if not self.connected or not self.sock:
            return None
        
        # 尝试读取数据
        try:
            data = self.sock.recv(8192)
            if not data:
                self.connected = False
                return None
            self._recv_buffer += data.decode("utf-8")
        except socket.timeout:
            return None
        except ConnectionResetError:
            self.connected = False
            return None
        
        # 解析完整的 JSON 消息（以换行符分隔）
        messages = []
        while "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
            except json.JSONDecodeError:
                print(f"[GodotClient] JSON 解析失败: {line[:100]}")
        
        # 返回最后一条 state 消息
        state_msg = None
        for msg in messages:
            if msg.get("type") == "state":
                state_msg = msg
            elif msg.get("type") == "reset_ack":
                print("[GodotClient] 收到 reset 确认")
            elif msg.get("type") == "pong":
                pass  # 心跳回复，忽略
        
        return state_msg
    
    def _send_json(self, data: dict) -> bool:
        """发送 JSON 数据"""
        try:
            text = json.dumps(data, ensure_ascii=False) + "\n"
            self.sock.sendall(text.encode("utf-8"))
            return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.connected = False
            return False


def random_agent_step(num_players: int = 4) -> list[int]:
    """随机策略：每个智能体随机选择一个动作"""
    return [random.randint(0, GodotClient.NUM_ACTIONS - 1) for _ in range(num_players)]


def main():
    """主函数：连接 Godot，循环发送随机动作"""
    client = GodotClient(host="127.0.0.1", port=11008)
    
    if not client.connect(timeout=30.0):
        print("[Main] 无法连接 Godot，退出")
        return
    
    print("[Main] 开始随机策略循环 (Ctrl+C 退出)")
    print("[Main] 动作映射: 0=上移, 1=下移, 2=左移, 3=右移, 4=攻击, 5=待机")
    
    step_count = 0
    
    try:
        while client.connected:
            # 1. 接收游戏状态
            state = client.receive_state()
            
            if state is not None:
                step_count += 1
                
                # 打印状态信息
                players = state.get("players", [])
                if step_count % 30 == 1:  # 每30步打印一次
                    print(f"\n[Step {step_count}] 游戏状态:")
                    for p in players:
                        status = "存活" if p.get("alive", False) else "死亡"
                        print(f"  Player {p.get('id', '?')} ({status}): "
                              f"pos=({p.get('x', 0):.0f}, {p.get('y', 0):.0f}), "
                              f"hp={p.get('hp', 0):.0f}/{p.get('max_hp', 100):.0f}")
                
                # 2. 生成随机动作
                actions = random_agent_step(num_players=len(players))
                
                if step_count % 30 == 1:
                    action_names = ["上移", "下移", "左移", "右移", "攻击", "待机"]
                    action_str = ", ".join(
                        f"P{i}:{action_names[a]}" for i, a in enumerate(actions)
                    )
                    print(f"  动作: {action_str}")
                
                # 3. 发送动作
                client.send_actions(actions)
            else:
                # 没有收到状态数据时短暂等待
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        print(f"\n[Main] 用户中断，共执行 {step_count} 步")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
