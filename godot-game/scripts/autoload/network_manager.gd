extends Node

## 网络通信单例（Autoload）
## 负责 Godot ↔ Python 的 TCP 通信
## - Godot 侧作为 TCP Server 监听端口 11008
## - 每帧收集4个玩家状态发送给 Python
## - 接收 Python 端的动作指令并分发到各玩家

signal client_connected
signal client_disconnected
signal actions_received(actions: Array)

const DEFAULT_PORT: int = 11008
const RECONNECT_CHECK_INTERVAL: float = 1.0

var _server: TCPServer
var _client: StreamPeerTCP
var _is_client_connected: bool = false
var _reconnect_timer: float = 0.0

## 待发送的游戏状态缓冲区
var _pending_state: Dictionary = {}
## 是否已收到本帧的动作
var _actions_ready: bool = false
var _cached_actions: Array = []

# ===== 生命周期 =====

func _ready() -> void:
	_start_server()

func _process(delta: float) -> void:
	_poll_connections(delta)
	_read_incoming_data()

# ===== 服务端管理 =====

func _start_server() -> void:
	_server = TCPServer.new()
	var err = _server.listen(DEFAULT_PORT)
	if err != OK:
		push_error("[NetworkManager] TCP Server 启动失败，端口 %d，错误码: %d" % [DEFAULT_PORT, err])
		return
	print("[NetworkManager] TCP Server 已启动，监听端口 %d" % DEFAULT_PORT)

func _poll_connections(delta: float) -> void:
	# 检查新连接
	if _server.is_connection_available():
		var new_client = _server.take_connection()
		if new_client != null:
			_client = new_client
			_is_client_connected = true
			print("[NetworkManager] Python 客户端已连接")
			client_connected.emit()
	
	# 检查现有连接状态
	if _is_client_connected and _client != null:
		_client.poll()
		var status = _client.get_status()
		if status == StreamPeerTCP.STATUS_NONE or status == StreamPeerTCP.STATUS_ERROR:
			print("[NetworkManager] Python 客户端断开连接")
			_is_client_connected = false
			_client = null
			client_disconnected.emit()

# ===== 数据读取 =====

func _read_incoming_data() -> void:
	if not _is_client_connected or _client == null:
		return
	
	var available = _client.get_available_bytes()
	if available <= 0:
		return
	
	var data = _client.get_data(available)
	if data[0] != OK:
		return
	
	var raw_bytes: PackedByteArray = data[1]
	var text = raw_bytes.get_string_from_utf8()
	
	# 处理可能的多条消息（按换行分割）
	var messages = text.split("\n", false)
	for msg in messages:
		_handle_message(msg.strip_edges())

func _handle_message(text: String) -> void:
	if text.is_empty():
		return
	
	var json = JSON.new()
	var err = json.parse(text)
	if err != OK:
		push_warning("[NetworkManager] JSON 解析失败: %s" % text)
		return
	
	var data = json.data
	if not data is Dictionary:
		push_warning("[NetworkManager] 非字典类型消息: %s" % text)
		return
	
	var msg_type = data.get("type", "")
	match msg_type:
		"action":
			var actions = data.get("actions", [])
			if actions.size() > 0:
				_cached_actions = actions
				_actions_ready = true
				actions_received.emit(actions)
		"reset":
			_handle_reset()
		"ping":
			send_message({"type": "pong"})
		_:
			push_warning("[NetworkManager] 未知消息类型: %s" % msg_type)

# ===== 数据发送 =====

func send_message(data: Dictionary) -> void:
	if not _is_client_connected or _client == null:
		return
	
	var json_text = JSON.stringify(data) + "\n"
	var err = _client.put_data(json_text.to_utf8_buffer())
	if err != OK:
		push_warning("[NetworkManager] 发送数据失败，错误码: %d" % err)

## 收集所有玩家状态并发送给 Python
func send_game_state(players: Array[Player]) -> void:
	var player_states: Array = []
	for p in players:
		player_states.append(p.get_state())
	
	var state_data = {
		"type": "state",
		"players": player_states,
		"observation": [],  # 观测数据暂为空
	}
	
	send_message(state_data)

## 发送重置确认
func send_reset_ack() -> void:
	send_message({"type": "reset_ack"})

# ===== 动作查询 =====

## 检查是否有缓存的动作，并取出（消费式）
func get_cached_actions() -> Array:
	if _actions_ready:
		_actions_ready = false
		return _cached_actions
	return []

## 是否有客户端连接
func is_client_connected() -> bool:
	return _is_client_connected

# ===== 重置 =====

func _handle_reset() -> void:
	print("[NetworkManager] 收到 reset 指令")
	# 通知 PlayScene 执行重置
	EventBus.game_reset_requested.emit()
	send_reset_ack()

func _exit_tree() -> void:
	if _client != null:
		_client.disconnect_from_host()
	if _server != null:
		_server.stop()
