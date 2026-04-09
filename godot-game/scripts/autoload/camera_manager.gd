extends Node

## 相机管理器（Autoload 单例）
## 管理主相机 + 4个玩家相机的切换
## 当前激活相机决定了 UI（血条/技能栏）的显示策略

signal camera_switched(camera_id: int)

## -1 = 主相机, 0-3 = 对应玩家相机
var current_camera_id: int = -1

## 主相机引用
var main_camera: Camera2D = null
## 玩家相机引用数组
var player_cameras: Array[Camera2D] = []
## 玩家引用数组（由 PlayScene 设置）
var players: Array = []

## 主相机配置
const MAIN_CAMERA_ZOOM := Vector2(0.79, 0.79)
const MAIN_CAMERA_POSITION := Vector2(78, 16)

## 玩家相机配置
const PLAYER_CAMERA_ZOOM := Vector2(2.2, 2.2)
const PLAYER_CAMERA_SMOOTH_SPEED := 8.0

func _process(delta: float) -> void:
	_update_player_cameras(delta)

## 初始化相机系统（由 PlayScene 在 _ready 中调用）
func setup(scene_root: Node) -> void:
	_find_or_create_cameras(scene_root)
	switch_to_main()

## 查找或创建相机节点
func _find_or_create_cameras(scene_root: Node) -> void:
	# 查找已有的主相机（原 Camera2D 节点）
	main_camera = scene_root.get_node_or_null("CameraMain")
	if main_camera == null:
		main_camera = scene_root.get_node_or_null("Camera2D")
	if main_camera != null:
		main_camera.name = "CameraMain"
		main_camera.zoom = MAIN_CAMERA_ZOOM
		main_camera.position = MAIN_CAMERA_POSITION
		main_camera.drag_horizontal_enabled = true
		main_camera.drag_vertical_enabled = true
	
	# 创建4个玩家相机
	for i in range(4):
		var cam_name = "CameraPlayer%d" % i
		var cam = scene_root.get_node_or_null(cam_name)
		if cam == null:
			cam = Camera2D.new()
			cam.name = cam_name
			scene_root.add_child(cam)
			cam.set_owner(scene_root)
		cam.zoom = PLAYER_CAMERA_ZOOM
		cam.position_smoothing_enabled = true
		cam.position_smoothing_speed = PLAYER_CAMERA_SMOOTH_SPEED
		cam.enabled = false
		player_cameras.append(cam)

## 每帧更新玩家相机位置，跟随对应玩家
func _update_player_cameras(delta: float) -> void:
	for i in range(min(player_cameras.size(), players.size())):
		if players[i] != null and not players[i].is_dead:
			player_cameras[i].global_position = players[i].global_position

## 切换到主相机
func switch_to_main() -> void:
	current_camera_id = -1
	if main_camera != null:
		main_camera.enabled = true
	for cam in player_cameras:
		if cam != null:
			cam.enabled = false
	camera_switched.emit(-1)
	print("[CameraManager] 切换到主相机")

## 切换到指定玩家的相机
func switch_to_player(player_id: int) -> void:
	if player_id < 0 or player_id >= player_cameras.size():
		push_warning("[CameraManager] 无效的 player_id: %d" % player_id)
		return
	
	current_camera_id = player_id
	
	if main_camera != null:
		main_camera.enabled = false
	
	for i in range(player_cameras.size()):
		if player_cameras[i] != null:
			player_cameras[i].enabled = (i == player_id)
	
	camera_switched.emit(player_id)
	print("[CameraManager] 切换到玩家 %d 相机" % player_id)

## 根据按钮索引切换相机（0=主相机, 1-4=玩家1-4）
func switch_by_index(index: int) -> void:
	if index == 0:
		switch_to_main()
	else:
		switch_to_player(index - 1)

## 获取当前相机ID（-1=主相机，0-3=玩家相机）
func get_current_camera_id() -> int:
	return current_camera_id

## 是否处于主相机视角
func is_main_camera() -> bool:
	return current_camera_id == -1

## 是否应该显示指定玩家的UI（血条+技能栏）
func should_show_player_ui(player_id: int) -> bool:
	# 主相机下隐藏所有UI
	if current_camera_id == -1:
		return false
	# 玩家相机下只显示对应玩家的UI
	return current_camera_id == player_id
