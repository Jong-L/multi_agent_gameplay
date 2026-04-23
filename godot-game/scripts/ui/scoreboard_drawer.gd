extends Control
class_name ScoreboardDrawer

## 分数榜抽屉面板
## 位于屏幕左侧的可隐藏面板，展示各玩家纯奖励累计值（不含塑形奖励）
## 按分数从高到低排序

# 颜色配置
const COLOR_HIGHLIGHT := Color(0.5, 1.0, 0.7)  ## 绿色高亮，按钮
const COLOR_NORMAL := Color(1.0, 1.0, 1.0)      ## 白色默认
const COLOR_PANEL_BG := Color(0.12, 0.12, 0.14, 0.8)  ## 面板背景色（半透明深色）
const COLOR_TITLE := Color(0.9, 0.9, 0.9)       ## 标题颜色
const PLAYER_COLORS := {
	"Blue": Color(0.3, 0.6, 1.0),
	"Black": Color(0.6, 0.6, 0.6),
	"Red": Color(1.0, 0.35, 0.35),
	"Yellow": Color(1.0, 0.9, 0.3),
}  #玩家颜色，用于区分不同玩家

# Toggle 按钮尺寸
@export var toggle_button_width: float = 80.0
@export var toggle_button_height: float = 40.0

# 动画时长（秒）
@export var animation_duration: float = 0.25

var _is_open: bool = false
var _toggle_btn: Button = null
var _panel: PanelContainer = null
var _list_container: VBoxContainer = null
var _player_rows: Dictionary = {}  # {player_id: HBoxContainer}
var _players: Array[Player] = []

func _ready() -> void:
	# 确保自身大小有效
	if size.x < 100:
		size.x = 280
	if size.y < 100:
		size.y = get_viewport().get_visible_rect().size.y if get_viewport() != null else 648
	
	_build_ui()
	EventBus.pure_reward_changed.connect(_on_pure_reward_changed)
	
	# 延迟两帧确保布局计算完成，然后强制隐藏面板
	await get_tree().process_frame
	await get_tree().process_frame
	_force_hide_panel()

func setup(players: Array[Player]) -> void:
	"""由 PlayScene 调用，传入玩家列表初始化分数榜"""
	_players = players.duplicate()
	_rebuild_player_rows()

func _build_ui() -> void:
	# ── Drawer Panel ──
	# 先创建面板，因为按钮位置依赖面板尺寸
	_panel = PanelContainer.new()
	_panel.name = "DrawerPanel"
	_panel.size = Vector2(size.x, size.y)
	_panel.position = Vector2(-size.x, 0)  # 初始隐藏在左侧
	add_child(_panel)
	
	# ── Toggle Button ──
	# 按钮左下角锚定到面板左上角
	_toggle_btn = Button.new()
	_toggle_btn.name = "ToggleButton"
	_toggle_btn.text = "分数榜"
	_toggle_btn.custom_minimum_size = Vector2(toggle_button_width, toggle_button_height)
	_toggle_btn.size = Vector2(toggle_button_width, toggle_button_height)
	# 按钮左下角 = 面板左上角（面板显示时）
	# 按钮 position 是左上角坐标，所以：
	# x = 面板左上角 x = 0
	# y = 面板左上角 y - 按钮高度 = -toggle_button_height
	_toggle_btn.position = Vector2(0, -toggle_button_height)
	_toggle_btn.pressed.connect(_on_toggle_pressed)
	# 默认样式
	_toggle_btn.add_theme_font_size_override("font_size", 14)
	add_child(_toggle_btn)
	
	# 面板背景样式
	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = COLOR_PANEL_BG
	panel_style.border_width_right = 2
	panel_style.border_color = Color(0.3, 0.3, 0.35)
	_panel.add_theme_stylebox_override("panel", panel_style)
	
	# ── 内部布局 ──
	var main_vbox := VBoxContainer.new()
	main_vbox.name = "MainVBox"
	main_vbox.add_theme_constant_override("separation", 6)
	_panel.add_child(main_vbox)
	
	# 顶部内边距
	var top_spacer := Control.new()
	top_spacer.custom_minimum_size = Vector2(0, 8)
	main_vbox.add_child(top_spacer)
	
	# 标题
	var title_label := Label.new()
	title_label.name = "TitleLabel"
	title_label.text = "分数榜"
	title_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title_label.add_theme_font_size_override("font_size", 18)
	title_label.add_theme_color_override("font_color", COLOR_TITLE)
	main_vbox.add_child(title_label)
	
	# 分隔线
	var sep := HSeparator.new()
	sep.modulate = Color(1.044, 1.044, 1.167, 1.0)
	main_vbox.add_child(sep)
	
	# 列表标题行
	var header := _create_header_row()
	main_vbox.add_child(header)
	
	# 玩家列表容器
	_list_container = VBoxContainer.new()
	_list_container.name = "PlayerList"
	_list_container.add_theme_constant_override("separation", 4)
	main_vbox.add_child(_list_container)
	
	# 确保 ToggleButton 在最上层
	_toggle_btn.z_index = 10

func _create_header_row() -> HBoxContainer:
	var row := HBoxContainer.new()
	row.custom_minimum_size = Vector2(0, 28)
	row.add_theme_constant_override("separation", 8)
	
	var rank := Label.new()
	rank.text = "#"
	rank.custom_minimum_size = Vector2(28, 0)
	rank.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	rank.add_theme_font_size_override("font_size", 12)
	rank.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	row.add_child(rank)
	
	var spacer := Control.new()
	spacer.custom_minimum_size = Vector2(20, 0)
	row.add_child(spacer)
	
	var name_label := Label.new()
	name_label.text = "玩家"
	name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	name_label.add_theme_font_size_override("font_size", 12)
	name_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	row.add_child(name_label)
	
	var score := Label.new()
	score.text = "奖励"
	score.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	score.custom_minimum_size = Vector2(70, 0)
	score.add_theme_font_size_override("font_size", 12)
	score.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	row.add_child(score)
	
	return row

func _rebuild_player_rows() -> void:
	# 清除旧行
	for child in _list_container.get_children():
		child.queue_free()
	_player_rows.clear()
	
	for p in _players:
		var row := _create_player_row(p)
		_list_container.add_child(row)
		_player_rows[p.player_id] = row
	
	_update_rankings()

func _create_player_row(player: Player) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.custom_minimum_size = Vector2(0, 34)
	row.add_theme_constant_override("separation", 8)
	
	# 排名
	var rank_label := Label.new()
	rank_label.name = "RankLabel"
	rank_label.text = "-"
	rank_label.custom_minimum_size = Vector2(28, 0)
	rank_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	rank_label.add_theme_font_size_override("font_size", 14)
	row.add_child(rank_label)
	
	# 颜色标识块
	var color_rect := ColorRect.new()
	color_rect.name = "ColorRect"
	color_rect.custom_minimum_size = Vector2(16, 16)
	color_rect.color = _get_player_color(player.skin_color)
	color_rect.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(color_rect)
	
	# 玩家名
	var name_label := Label.new()
	name_label.name = "NameLabel"
	name_label.text = player.skin_color
	name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	name_label.add_theme_font_size_override("font_size", 14)
	name_label.add_theme_color_override("font_color",PLAYER_COLORS.get(player.skin_color,Color.WHITE))
	row.add_child(name_label)
	
	# 分数
	var score_label := Label.new()
	score_label.name = "ScoreLabel"
	score_label.text = "0.00"
	score_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	score_label.custom_minimum_size = Vector2(70, 0)
	score_label.add_theme_font_size_override("font_size", 14)
	score_label.add_theme_color_override("font_color",PLAYER_COLORS.get(player.skin_color,Color.WHITE))
	row.add_child(score_label)
	
	return row

func _get_player_color(color_name: String) -> Color:
	match color_name:
		"Blue": return Color(0.3, 0.6, 1.0)
		"Black": return Color(0.6, 0.6, 0.6)
		"Red": return Color(1.0, 0.35, 0.35)
		"Yellow": return Color(1.0, 0.9, 0.3)
		_: return Color.WHITE

func _on_pure_reward_changed(player_id: int, total_pure_reward: float) -> void:
	var row = _player_rows.get(player_id)
	if row == null:
		return
	var score_label := row.get_node("ScoreLabel") as Label
	if score_label != null:
		score_label.text = "%.2f" % total_pure_reward
	_update_rankings()

func _update_rankings() -> void:
	# 收集所有分数
	var entries: Array[Dictionary] = []
	for pid in _player_rows.keys():
		var row = _player_rows[pid]
		var score_label := row.get_node("ScoreLabel") as Label
		var score := float(score_label.text)
		entries.append({"id": pid, "score": score, "row": row})
	
	# 按分数从高到低排序
	entries.sort_custom(func(a, b): return a.score > b.score)
	
	# 更新排名显示和列表顺序
	for i in range(entries.size()):
		var entry = entries[i]
		var rank_label := entry.row.get_node("RankLabel") as Label
		rank_label.text = str(i + 1)
		# 第一名特殊颜色
		if i == 0:
			rank_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.3))
		else:
			rank_label.remove_theme_color_override("font_color")
		# 重排节点顺序
		_list_container.move_child(entry.row, i)

func _on_toggle_pressed() -> void:
	_is_open = not _is_open
	_animate_drawer()
	_update_toggle_highlight()

func _force_hide_panel() -> void:
	"""强制将面板完全隐藏到屏幕外（用于初始化和保底）"""
	if _panel == null:
		return
	# 使用 get_global_rect().size 获取面板实际渲染尺寸（包含边框和样式）
	var actual_width: float = _panel.get_global_rect().size.x
	_panel.position.x = -actual_width

func _animate_drawer() -> void:
	var actual_width: float = _panel.get_global_rect().size.x
	var target_x := 0.0 if _is_open else -actual_width
	var tween := create_tween()
	tween.tween_property(_panel, "position:x", target_x, animation_duration) \
		.set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)

func _update_toggle_highlight() -> void:
	if _is_open:
		_toggle_btn.modulate = COLOR_HIGHLIGHT
	else:
		_toggle_btn.modulate = COLOR_NORMAL

func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		if _panel != null:
			_panel.size = Vector2(size.x, size.y)
			if not _is_open:
				var actual_width: float = _panel.get_global_rect().size.x
				_panel.position.x = -actual_width
		if _toggle_btn != null:
			# 按钮位置始终锚定到面板左上角（面板显示时的位置）
			_toggle_btn.position = Vector2(0, -toggle_button_height)
