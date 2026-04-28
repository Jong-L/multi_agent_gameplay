class_name RewardLogger

## 奖励日志记录器 (CSV 格式)
## 缓存写入：奖励事件先存入内存，在 episode 结束或场景正常退出时批量写入 CSV
## 排除塑形奖励 仅记录 add_reward 来源的奖励

var _file: FileAccess = null
var _log_path: String = ""
var _episode_id: int = -1
var _buffer: Array[Dictionary] = []

# 由RewardManager实例化
func _init(log_path: String = "") -> void:
	if log_path.is_empty():
		_log_path = _get_default_log_path()
	else:
		_log_path = log_path
	
	# print("[RewardLogger] 日志路径: ", ProjectSettings.globalize_path(_log_path))

func _get_default_log_path() -> String:
	var t = Time.get_datetime_dict_from_system()
	var time_str = "%02d-%02d_%02d-%02d" % [
		t["month"], t["day"],
		t["hour"], t["minute"]
	]
	# 添加进程ID以区分并行训练的多个环境实例
	var process_id = OS.get_process_id()
	return "D:\\schoolTour\\softwares\\multi-agent-gameplay\\logs\\valid_mask_comparison\\32_rays_average_%s_pid%d.csv" % [time_str, process_id]

func _ensure_file_open() -> void:
	if _file != null:
		return
	
	var dir_path := _log_path.get_base_dir()
	if not DirAccess.dir_exists_absolute(dir_path):
		var err := DirAccess.make_dir_recursive_absolute(dir_path)
		if err != OK:
			push_error("[RewardLogger] 无法创建目录: " + dir_path)
			return
	
	if FileAccess.file_exists(_log_path):
		_file = FileAccess.open(_log_path, FileAccess.READ_WRITE)
		if _file:
			var length := _file.get_length()
			if length > 0:
				_file.seek(length - 1)
				var last_byte := _file.get_8()
				if last_byte != 10:  # 不是 \n，补一个换行避免与旧内容粘连
					_file.seek_end()
					_file.store_8(10)
			_file.seek_end()# 将文件指针移动到文件末尾
	else:
		_file = FileAccess.open(_log_path, FileAccess.WRITE)
	
	if _file == null:
		push_error("[RewardLogger] 无法打开文件: " + _log_path + ", 错误码: " + str(FileAccess.get_open_error()))

func start_episode() -> void:
	_episode_id += 1
	#print("[RewardLogger] Episode %d 开始" % _episode_id)

func end_episode() -> void:
	_flush()
	#print("[RewardLogger] Episode %d 结束，已写入 CSV" % _episode_id)

## 外部调用：缓存一条奖励事件（game_time 截断到小数点后两位）
func log_reward(player_id: int, source: String, value: float, game_time: float) -> void:
	if source=="run":
		return
	if _episode_id < 0:
		_episode_id = 0
	
	_buffer.append({
		"episode_id": _episode_id,
		"player_id": player_id,
		"source": source,
		"value": value,
		"game_time": game_time
	})

## 强制将缓存数据写入磁盘（正常关闭、场景切换时调用）
func flush() -> void:
	_flush()

func _flush() -> void:
	if _buffer.is_empty():
		return
	
	_ensure_file_open()
	if _file == null:
		push_error("[RewardLogger] 写入失败：文件未打开")
		return
	
	# 空文件时写入 CSV Header
	if _file.get_length() == 0:
		_file.store_line("episode_id,player_id,source,value,game_time")
	
	for entry in _buffer:
		var line := "%d,%d,%s,%s,%s" % [
			entry.episode_id,
			entry.player_id,
			_csv_escape(str(entry.source)),
			str(entry.value),
			"%.2f" % entry.game_time
		]
		_file.store_line(line)
	
	_file.flush()
	_file = null  # 关闭文件句柄，释放给其他进程读取
	_buffer.clear()

# This ensures that special characters in the string don't break the CSV structure
func _csv_escape(s: String) -> String:
	if s.contains(",") or s.contains("\"") or s.contains("\n"):
		# Escape double quotes by doubling them
		s = s.replace("\"", "\"\"")
		# Wrap the entire string in double quotes
		return "\"" + s + "\""
	return s
