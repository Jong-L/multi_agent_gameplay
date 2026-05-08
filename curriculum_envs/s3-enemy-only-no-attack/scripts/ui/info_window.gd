extends Window
class_name InfoWindow

## 调试信息窗口
## 展示每个玩家的 reward 详细数据（episode 累计 + 上一 circle 增量）
## 生命周期：_ready() 做最小初始化 → PlayScene 调用 setup() → 连信号刷新

@export var game_config: GameConfig = null
@onready var _grid: GridContainer = $GridContainer

## 指标定义：添加新指标只需追加一行
## key 对应 RewardManager.get_player_debug_info() / 信号返回的字段名
const METRICS: Array[Dictionary] = [
	{"key": "total_reward",       "label": "Total Reward"},
	{"key": "prev_circle_total",  "label": "Prev Circle Total"},
	{"key": "total_pure",         "label": "Total Pure"},
	{"key": "prev_circle_pure",   "label": "Prev Circle Pure"},
	{"key": "total_ball",         "label": "Total Ball Shaping"},
	{"key": "prev_circle_ball",   "label": "Prev Circle Ball"},
	{"key": "total_wall",         "label": "Total Wall Shaping"},
	{"key": "prev_circle_wall",   "label": "Prev Circle Wall"},
]

const PLAYER_COLORS := {
	"Blue":   Color(0.30, 0.60, 1.00),
	"Black":  Color(0.60, 0.60, 0.60),
	"Red":    Color(1.00, 0.35, 0.35),
	"Yellow": Color(1.00, 0.90, 0.30),
}

const PANEL_BG_COLOR := Color(0.141, 0.141, 0.161, 0.761)

var play_scene: PlayScene = null

# {player_id: {"title": Label, "rows": {metric_key: Label}}}
var _player_panels: Dictionary = {}
var _initialized: bool = false
var _info_font_size:int=15
var _title_font_size:int=20

func _ready() -> void:
	if game_config == null or not game_config.enable_info_window:
		return

	# 将 self 引用存到 PlayScene，方便后续 setup
	var ps: Node = get_parent()
	if ps is PlayScene:
		ps.info_window = self

	EventBus.reward_circle_completed.connect(_on_circle_completed)

# 由 PlayScene._ready() 在玩家收集完毕后调用
func setup(ps: PlayScene) -> void:
	if _initialized:
		return
	play_scene = ps
	_build_ui()
	_initialized = true
	show()

## ── UI 构建 ──

func _build_ui() -> void:
	_grid.add_theme_constant_override("h_separation", 12)
	_grid.add_theme_constant_override("v_separation", 12)

	for player in play_scene.players:
		var pid: int = player.player_id
		var skin: String = player.skin_color
		_create_player_panel(pid, skin)

func _create_player_panel(pid: int, skin_color: String) -> void:
	var player_color: Color = PLAYER_COLORS.get(skin_color, Color.WHITE)

	var panel_container := PanelContainer.new()
	panel_container.name = "Panel_%s" % skin_color
	panel_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panel_container.size_flags_vertical = Control.SIZE_EXPAND_FILL

	# 面板样式：圆角卡片 + 主题色边框
	var style := StyleBoxFlat.new()
	style.bg_color = PANEL_BG_COLOR
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.border_width_left = 2
	style.border_width_top = 2
	style.border_width_right = 2
	style.border_width_bottom = 2
	style.border_color = player_color * 0.8
	style.content_margin_left = 12
	style.content_margin_top = 12
	style.content_margin_right = 12
	style.content_margin_bottom = 12
	panel_container.add_theme_stylebox_override("panel", style)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel_container.add_child(vbox)

	# 标题
	var title_label := Label.new()
	title_label.name = "Title"
	title_label.text = skin_color
	title_label.add_theme_font_size_override("font_size", _title_font_size)
	title_label.add_theme_color_override("font_color", player_color)
	title_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(title_label)

	# 分隔线
	var sep := HSeparator.new()
	var sep_style := StyleBoxLine.new()
	sep_style.color = player_color * 0.5
	sep_style.thickness = 1
	sep.add_theme_stylebox_override("separator", sep_style)
	vbox.add_child(sep)

	var row_labels: Dictionary = {}
	for metric in METRICS:
		var row := HBoxContainer.new()
		row.size_flags_horizontal = Control.SIZE_EXPAND_FILL

		var name_label := Label.new()
		name_label.text = metric.label
		name_label.add_theme_font_size_override("font_size", _info_font_size)
		name_label.add_theme_color_override("font_color", Color(0.75, 0.75, 0.75))
		row.add_child(name_label)

		var value_label := Label.new()
		value_label.name = metric.key
		value_label.text = "--"
		value_label.add_theme_font_size_override("font_size", _info_font_size)
		value_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		value_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(value_label)

		vbox.add_child(row)
		row_labels[metric.key] = value_label

	_grid.add_child(panel_container)
	_player_panels[pid] = {"title": title_label, "rows": row_labels}

## ── 信号处理 ──

func _on_circle_completed(circle_data: Array) -> void:
	if not _initialized:
		return
	for entry in circle_data:
		var pid: int = entry.get("player_id", -1)
		if not _player_panels.has(pid):
			continue
		_update_panel(pid, entry)

func _update_panel(pid: int, data: Dictionary) -> void:
	var panel: Dictionary = _player_panels[pid]
	var rows: Dictionary = panel.rows

	for metric in METRICS:
		var key: String = metric.key
		var row: Label = rows.get(key)
		if row == null:
			continue

		var value: float = data.get(key, 0.0)
		row.text = "%s%.4f" % [_sign_prefix(value), value]
		row.add_theme_color_override("font_color", _reward_color(value))

## ── 工具 ──

func _sign_prefix(value: float) -> String:
	if value > 0.001:
		return "+"
	return ""

func _reward_color(value: float) -> Color:
	if value > 0.001:
		return Color(1.0, 0.45, 0.45)   # 柔和红
	elif value < -0.001:
		return Color(0.45, 1.0, 0.45)   # 柔和绿
	return Color(0.9, 0.9, 0.9)       # 灰白

func _on_close_requested() -> void:
	hide()
