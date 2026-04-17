extends Node
class_name PlayScene
"""
 游戏主场景控制器
 职责：
 - 初始化玩家、相机系统、UI
 - 处理游戏重置、暂停等全局状态
 - 提供地图数据（竞技场边界、巡逻区域、装饰物坐标）
"""

@export var screen_transition: ColorRect # 屏幕过渡遮罩
@export var pause_menu: PauseMenu	# 暂停菜单

@onready var canvas_layer:CanvasLayer=$CanvasLayer
@onready var control_node:Control=$CanvasLayer/Control
@onready var _grass_layer:TileMapLayer=$Map/Grass
@onready var _road_layer:TileMapLayer=$Map/Road
@onready var _collision_deco_layer:TileMapLayer=$Map/CollisonDecoration
# 视野传感器（环境级工具，统一管理所有玩家的观测计算）
@onready var vision_sensor: VisionSensor = $VisionSensor
# 所有玩家引用,按 player_id 排序
var players: Array[Player] = []
var enemies:Array[Enemy]=[]
# 奖励球管理器
var reward_ball_manager: RewardBallManager = null
# 奖励管理器
var reward_manager: RewardManager = null
# 相机切换按钮组
var camera_buttons: Array[Button] = []
# 玩家专属血条数组与技能栏，与各自的player对应
var player_health_bars: Array[TextureProgressBar] = []
var player_spell_bars: Array[SpellBar] = []
# 视野圆可视化
var vision_circles: Array[VisionCircle] = []
var vision_circles_visible: bool = false  # 视野圆是否可见（默认隐藏）

# 地图数据
var arena_bounds: Rect2 = Rect2()#Grass层表示的的竞技场区域
var patrol_rect: Rect2 = Rect2()#Road层表示的巡逻区域
var collision_decoration_positions: Array[Vector2] = []  ## CollisionDecoration 各 tile 的世界坐标
var arena_length:float#正方形竞技场，只记录边长

var is_resetting:bool=false #四个玩家都执行重置时只重置一次
func _ready() -> void:
	#加载地图数据
	_init_map_data()
	# 连接全局事件
	EventBus.game_paused.connect(_handle_pause)

	# 延迟一帧，确保所有子节点已就绪
	await get_tree().process_frame
	
	_collect_enemis()
	_collect_players()          
	_setup_camera_system()      
	_setup_camera_switch_ui()   
	_setup_player_uis()         
	_setup_reward_ball_manager()
	_setup_reward_manager()
	_setup_vision_circles()         

# 初始化地图数据：竞技场边界、巡逻区域、碰撞装饰物坐标
func _init_map_data() -> void:
	# Grass 层（竞技场）
	if _grass_layer:
		arena_bounds = MathUtils._tilemap_to_world_rect(_grass_layer)
		arena_length=arena_bounds.size[0]
	else:
		print("get tile_layer error")
	
	# Road 层（巡逻区域）
	if _road_layer != null:
		patrol_rect =MathUtils._tilemap_to_world_rect(_road_layer)
	
	# CollisionDecoration
	if _collision_deco_layer != null:
		collision_decoration_positions =MathUtils._tilemap_to_world_positions(_collision_deco_layer)
	
	#print("[PlayScene] 地图数据初始化完成")
	#print("  竞技场边界: %s" % arena_bounds)
	#print("  巡逻区域: %s" % patrol_rect)
	#print("  障碍物数量: %d" % collision_decoration_positions.size())

# 收集场景中所有 Player 节点
func _collect_players() -> void:
	players.clear()
	for node in get_tree().get_nodes_in_group("player"):
		if node is Player:
			players.append(node)
	
	# 按 player_id 排序，确保动作数组顺序一致
	players.sort_custom(func(a, b): return a.player_id < b.player_id)
	
	#print("[PlayScene] 找到 %d 个玩家" % players.size())
	#for p in players:
		#print("  Player %d (%s) at %s" % [p.player_id, p.skin_color, p.position])

func _collect_enemis()->void:
	for child in get_children():
		if child is Enemy:
			enemies.append(child)
	#print(enemies)
# 初始化相机系统，将玩家引用传给 CameraManager
func _setup_camera_system() -> void:
	CameraManager.players.clear()
	CameraManager.players=players.duplicate()
	CameraManager.setup(self)

# 创建相机切换按钮组
func _setup_camera_switch_ui() -> void:
	if canvas_layer == null:
		push_error("[PlayScene] 未找到 CanvasLayer")
		return
	# 获取 ESC 暂停按钮的位置，相机切换按钮放在它下方
	var pause_button = canvas_layer.get_node_or_null("PauseButton") as TextureButton
	var start_x := 929.0
	var start_y := 155.0
	if pause_button != null:
		start_x = pause_button.offset_left
		start_y = pause_button.offset_bottom
	
	# 创建垂直按钮容器
	var panel = VBoxContainer.new()
	panel.name = "CameraSwitchPanel"
	panel.offset_left = start_x
	panel.offset_top = start_y - 250
	panel.offset_right = start_x + 150.0
	panel.offset_bottom = start_y
	panel.add_theme_constant_override("separation", 5)
	canvas_layer.add_child(panel)
	panel.set_owner(self)
	
	# 按钮配置,默认4个玩家，硬编码，以后看情况改
	print(players[0].skin_color)
	var button_configs := [
		["主相机", 0],
		["玩家{color}".format({"color":players[0].skin_color}), 1],
		["玩家{color}".format({"color":players[1].skin_color}), 2],
		["玩家{color}".format({"color":players[2].skin_color}), 3],
		["玩家{color}".format({"color":players[3].skin_color}), 4],
	]
	
	camera_buttons.clear()
	for config in button_configs:
		var btn = Button.new()
		btn.name = "CameraBtn_%d" % config[1]
		btn.text = config[0]
		btn.custom_minimum_size = Vector2(150, 40)
		btn.tooltip_text = "切换到%s" % config[0]
		var idx: int = config[1]
		btn.pressed.connect(func(): _on_camera_button_pressed(idx))
		panel.add_child(btn)
		btn.set_owner(self)
		camera_buttons.append(btn)
	
	# 连接相机切换信号，同步按钮高亮状态
	CameraManager.camera_switched.connect(_on_camera_switched)
	_update_button_highlight(-1)  # 初始高亮主相机
	
	# 在面板底部添加"视野提示"切换按钮
	var vision_btn := Button.new()
	vision_btn.name = "VisionToggleButton"
	vision_btn.text = "视野提示"
	vision_btn.custom_minimum_size = Vector2(150, 40)
	vision_btn.tooltip_text = "显示/隐藏玩家视野范围"
	vision_btn.modulate = Color(1.0, 1.0, 1.0)  # 默认白色=已关闭
	vision_btn.pressed.connect(_on_vision_toggle_pressed)
	panel.add_child(vision_btn)
	vision_btn.set_owner(self)

# 动态创建玩家的血条和技能栏
func _setup_player_uis() -> void:
	if canvas_layer == null:
		return
		
	if control_node == null:
		return
	
	# 获取现有蓝玩家血条的纹理资源，用于复制给其他血条
	var blue_health_bar = canvas_layer.get_node_or_null("PlayerHealthBar_Blue")
	
	var player_names = ["Blue", "Black", "Red", "Yellow"]
	# 血条位置
	var health_bar_offset = Vector2(3.74, -64.0)
	
	player_health_bars.clear()
	player_spell_bars.clear()
	
	for i in range(players.size()):
		var p = players[i]
		# 创建血条 
		if i == 0 and blue_health_bar != null:
			player_health_bars.append(blue_health_bar)
		else:
			# 动态创建其他玩家的血条
			var health_bar = TextureProgressBar.new()
			health_bar.name = "PlayerHealthBar_%s" % player_names[i]
			health_bar.anchors_preset = Control.PRESET_BOTTOM_LEFT
			health_bar.anchor_top = 1.0
			health_bar.anchor_bottom = 1.0
			health_bar.offset_left = health_bar_offset.x
			health_bar.offset_top = health_bar_offset.y
			health_bar.offset_right = health_bar_offset.x + 306.0
			health_bar.offset_bottom = health_bar_offset.y + 50.0
			health_bar.grow_vertical = Control.GROW_DIRECTION_BEGIN
			health_bar.value = 100.0
			health_bar.max_value = 100.0
			# 复制蓝血条的纹理
			if blue_health_bar != null:
				health_bar.texture_under = blue_health_bar.texture_under
				health_bar.texture_progress = blue_health_bar.texture_progress
				health_bar.texture_progress_offset = blue_health_bar.texture_progress_offset
			health_bar.script = load("res://scripts/utils/player_health_bar.gd")
			health_bar.player = p
			canvas_layer.add_child(health_bar)
			health_bar.set_owner(self)
			player_health_bars.append(health_bar)
		
		# =创建技能栏
		var spell_bar = SpellBar.new()
		spell_bar.name = "SpellBar_%s" % player_names[i]
		spell_bar.self_modulate = Color(1, 1, 1, 0)  # 初始透明
		# 布局：底部居中，略微偏移避免与血条重叠
		spell_bar.anchor_left = 0.5
		spell_bar.anchor_right = 0.5
		spell_bar.anchor_top = 1.0
		spell_bar.anchor_bottom = 1.0
		spell_bar.offset_left = -63.0
		spell_bar.offset_top = -123.0
		spell_bar.offset_right = 67.0
		spell_bar.offset_bottom = 7.0
		spell_bar.grow_horizontal = Control.GROW_DIRECTION_BOTH
		spell_bar.grow_vertical = Control.GROW_DIRECTION_BEGIN
		spell_bar.bound_player_id = p.player_id
		spell_bar.visible = false  # 初始隐藏，相机切换时决定显示
		
		# 创建内部结构: MarginContainer > HBoxContainer > SpellButton
		var margin = MarginContainer.new()
		margin.name = "MarginContainer"
		margin.layout_mode = 2
		spell_bar.add_child(margin)
		margin.set_owner(self)
		
		var hbox = HBoxContainer.new()
		hbox.name = "HBoxContainer"
		hbox.layout_mode = 2
		hbox.alignment = BoxContainer.ALIGNMENT_CENTER
		margin.add_child(hbox)
		hbox.set_owner(self)
		
		# 加载 SpellButton 场景
		var spell_button_scene = load("res://assets/scenes/texture_button.tscn")
		if spell_button_scene != null:
			var spell_btn = spell_button_scene.instantiate()
			spell_btn.layout_mode = 2
			hbox.add_child(spell_btn)
			spell_btn.set_owner(self)
		
		spell_bar.button_container = hbox
		control_node.add_child(spell_bar)
		spell_bar.set_owner(self)
		
		# 注册玩家技能到技能栏
		for skill_idx in range(p.skill_controller.skills.size()):
			var skill = p.skill_controller.skills[skill_idx]
			spell_bar.register_skill(skill, skill_idx)
		
		# 更新玩家的 spell_bar 引用
		p.player_spell_bar = spell_bar
		player_spell_bars.append(spell_bar)
	
	# 确保蓝色血条的 player 引用正确
	if blue_health_bar != null and players.size() > 0:
		blue_health_bar.player = players[0]
		blue_health_bar.visible = false  # 初始隐藏（主相机模式）
	
	# 隐藏旧的共享 SpellBar（兼容旧场景）
	var old_spell_bar = control_node.get_node_or_null("SpellBar")
	if old_spell_bar != null:
		old_spell_bar.visible = false

# 初始化奖励球管理器
func _setup_reward_ball_manager() -> void:
	reward_ball_manager = RewardBallManager.new()
	reward_ball_manager.name = "RewardBallManager"
	add_child(reward_ball_manager)
	reward_ball_manager.set_owner(self)
	reward_ball_manager.setup(self)

# 初始化奖励管理器
func _setup_reward_manager() -> void:
	reward_manager = RewardManager.new()
	reward_manager.name = "RewardManager"
	add_child(reward_manager)
	reward_manager.set_owner(self)
	reward_manager.setup(self)

# 为每个玩家创建视野范围虚线圆
func _setup_vision_circles() -> void:
	vision_circles.clear()
	# 各玩家对应的视野圆颜色
	var vision_colors := {
		"Blue": Color(0.3, 0.6, 1.0, 1.0),
		"Black": Color(0.7, 0.7, 0.7, 1.0),
		"Red": Color(1.0, 0.3, 0.3, 1.0),
		"Yellow": Color(1.0, 1.0, 0.3, 1.0),
	}
	var radius := vision_sensor.vision_radius if vision_sensor else 250.0
	for p in players:
		var circle := VisionCircle.new()
		circle.name = "VisionCircle_%s" % p.skin_color
		var color: Color = vision_colors.get(p.skin_color, Color(1, 1, 1, 1))
		circle.setup(p, radius, color)
		circle.visible = vision_circles_visible  # 默认隐藏
		p.add_child(circle)
		circle.set_owner(self)
		vision_circles.append(circle)

# 视野提示按钮回调
func _on_vision_toggle_pressed() -> void:
	vision_circles_visible = not vision_circles_visible
	for circle in vision_circles:
		if is_instance_valid(circle):
			circle.visible = vision_circles_visible
	_update_vision_toggle_highlight()

# 更新视野提示按钮的视觉状态
func _update_vision_toggle_highlight() -> void:
	var btn: Button = canvas_layer.get_node_or_null("CameraSwitchPanel/VisionToggleButton") as Button
	if btn == null:
		return
	if vision_circles_visible:
		btn.modulate = Color(0.5, 1.0, 0.7)  # 绿色高亮 = 已开启
	else:
		btn.modulate = Color(1.0, 1.0, 1.0)   # 白色 = 已关闭

# 相机切换按钮回调（由按钮 pressed 信号触发）
func _on_camera_button_pressed(index: int) -> void:
	CameraManager.switch_by_index(index)

# 相机切换处理
func _on_camera_switched(camera_id: int) -> void:
	_update_button_highlight(camera_id)

# 更新按钮高亮状态
func _update_button_highlight(camera_id: int) -> void:
	var active_index := 0 if camera_id == -1 else camera_id + 1
	for i in range(camera_buttons.size()):
		if i == active_index:
			camera_buttons[i].modulate = Color(0.5, 1.0, 0.7)  # 绿色高亮
		else:
			camera_buttons[i].modulate = Color(1.0, 1.0, 1.0)   # 默认白色

# 构建 map_state（边界距离 + 障碍物相对向量）
func _build_map_state(player: Player) -> Array[float]:
	var map_state: Array[float] = []
	var player_pos = player.global_position
	
	# 4个边界距离[0, 1]
	map_state.append((player_pos.x - arena_bounds.position.x) / arena_length)
	map_state.append((arena_bounds.end.x - player_pos.x) / arena_length)
	map_state.append((player_pos.y - arena_bounds.position.y) / arena_length)
	map_state.append((arena_bounds.end.y - player_pos.y) / arena_length)
	
	#障碍物的相对向量 [-1, 1]
	for deco_pos in collision_decoration_positions:
		map_state.append((deco_pos.x - player_pos.x) / arena_length)
		map_state.append((deco_pos.y - player_pos.y) / arena_length)
	
	return map_state

#为指定玩家生成观测数据
func get_obs_for_player(player: Player) -> Dictionary:
	if vision_sensor == null or not is_instance_valid(vision_sensor):
		# 无传感器时返回最小观测
		return {
			"self_state": [0.0, 0.0, 0.0, 0.0],
			"nearby_players": [],
			"nearby_balls": [],
			"nearby_enemies": [],
			"map_state":[]
		}
	# 收集当前活跃的奖励球
	var all_balls: Array[RewardBall] = []
	if reward_ball_manager != null:
		all_balls = reward_ball_manager.reward_balls
	var obs_dict = vision_sensor.scan(
		player,
		players,
		enemies,
		all_balls,
		arena_length,
	)
	
	# 添加地图状态到观测字典
	obs_dict["map_state"] = _build_map_state(player)
	
	return obs_dict
	

func _apply_actions(actions: Array) -> void:# 将动作数组分发到各玩家
	for i in range(min(actions.size(), players.size())):
		var action_value = int(actions[i])
		players[i].set_action(action_value)

# 处理游戏重置请求
func _handle_reset() -> void:
	var time=Time.get_time_string_from_system()
	print("[PlayScene] 执行游戏重置 at ",time)
	for p in players:
		p.current_animation_wrapper = null
		p.is_dead = false
		p.position = p.spawn_position
		p.current_health = p.max_health
		p.pending_action = Player.Action.IDLE
		p.last_damage_source = null  # 清除伤害来源记录
		# 重置所有技能冷却
		for skill in p.skill_controller.cooldowns.keys():
			p.skill_controller.cooldowns[skill] = 0.0
			skill.current_cooldown = 0.0
	
	for enemy in enemies:
		enemy.full_reset()
	
	# 重置奖励球
	if reward_ball_manager:
		reward_ball_manager.reset_all()
	
	# 重置奖励管理器
	if reward_manager:
		reward_manager.reset()

func _reset_player_state(player: Player) -> void:
	player.current_animation_wrapper = null
	player.is_dead = false
	player.position = player.spawn_position
	player.current_health = player.max_health
	player.last_damage_source = null  # 清除伤害来源记录

func _reset_with_transition(player:Player)->void:
	var tween = fade_in()
	await tween.finished
	_reset_player_state(player)
	tween = fade_out()
	await tween.finished
# 处理玩家死亡信号
func _on_player_player_died(player: Player) -> void:
	var tween
	var is_main_camera=CameraManager.current_camera_id==-1
	
	if is_main_camera:
		_reset_player_state(player)
		return
	_reset_with_transition(player)

#淡出
func fade_out() -> Tween:
	var tween = create_tween()
	tween.tween_property(
		screen_transition,
		"color:a",
		0.0,
		0.4
	).set_trans(Tween.TRANS_LINEAR).set_ease(Tween.EASE_OUT)
	return tween

#淡入
func fade_in() -> Tween:
	var tween = create_tween()
	tween.tween_property(
		screen_transition,
		"color:a",
		1.0,
		0.5
	).set_trans(Tween.TRANS_LINEAR).set_ease(Tween.EASE_IN)
	return tween
#暂停
func _on_pause_button_pressed() -> void:
	EventBus.game_paused.emit(true)
	pause_menu.show()
	get_tree().paused = true

#暂停时降低屏幕亮度
func _handle_pause(paused: bool) -> void:
	if paused:
		screen_transition.color = Color(0, 0, 0, 0.5)# 半透明
	else:
		screen_transition.color = Color(0, 0, 0, 0)# 完全透明
