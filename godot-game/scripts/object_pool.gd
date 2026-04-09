class_name ObjectPool
extends Node

## 对象池（单例）
## 用于复用频繁创建/销毁的游戏对象（特效、子弹等）
## 减少 GC 压力，提升性能
##
## 使用方式：
##   - 获取：ObjectPool.spawn("res://scene.tscn", parent, position)
##   - 归还：ObjectPool.return_to_pool(node, "res://scene.tscn")
##
## 注意：
##   - pool_key 必须是场景资源路径
##   - 归还的对象会被隐藏并从场景树移除

static var pools: Dictionary = {}             ## 对象池字典：路径 → 节点数组

func _ready() -> void:
	pass

func _process(delta: float) -> void:
	pass

## 从对象池获取对象
## @param pool_key: 场景资源路径
## @param parent_node: 父节点
## @param initial_position: 初始位置
## @return: 节点实例
static func spawn(pool_key: String, parent_node: Node, initial_position: Vector2 = Vector2.ZERO) -> Node:
	if not pools.has(pool_key):
		_init_pool(pool_key)
	
	var pool = pools[pool_key]
	var node: Node
	
	if pool.size() > 0:
		## 复用池中对象
		node = pool.pop_front()
		if node.is_inside_tree():
			node.show()
			node.disabled = false
		else:
			parent_node.add_child(node)
			node.global_position = initial_position
	else:
		## 创建新对象
		node = ResourceLoader.load(pool_key).instantiate()
		parent_node.add_child(node)
		node.global_position = initial_position
	
	return node

## 归还对象到对象池
## @param node: 要归还的节点
## @param pool_key: 场景资源路径
static func return_to_pool(node: Node, pool_key: String) -> void:
	node.hide()
	node.disabled = true
	if node.is_inside_tree():
		node.remove_from_parent()
	
	if not pools.has(pool_key):
		pools[pool_key] = []
	pools[pool_key].append(node)

## 初始化对象池
static func _init_pool(pool_key: String) -> void:
	pools[pool_key] = []

## 清空指定对象池
## @param pool_key: 场景资源路径
static func clear_pool(pool_key: String) -> void:
	if pools.has(pool_key):
		pools[pool_key].clear()

## 清空所有对象池
static func clear_all_pools() -> void:
	pools.clear()

## 获取池中对象数量
## @param pool_key: 场景资源路径
## @return: 可用对象数量
static func get_pool_size(pool_key: String) -> int:
	if pools.has(pool_key):
		return pools[pool_key].size()
	return 0

## 预加载对象到池中
## @param pool_key: 场景资源路径
## @param count: 预加载数量
## @return: 实际加载数量
static func prewarm(pool_key: String, count: int) -> int:
	if not pools.has(pool_key):
		_init_pool(pool_key)
	
	var loaded = 0
	for i in range(count):
		var node = ResourceLoader.load(pool_key).instantiate()
		if node != null:
			pools[pool_key].append(node)
			loaded += 1
	
	return loaded

## 检查对象池是否存在
## @param pool_key: 场景资源路径
## @return: true 表示存在
static func has_pool(pool_key: String) -> bool:
	return pools.has(pool_key)

## 获取所有池的键
## @return: 键数组
static func get_all_pool_keys() -> Array:
	return pools.keys()

## 获取池的总大小（所有池的对象总数）
## @return: 对象总数
static func get_total_pool_size() -> int:
	var total = 0
	for key in pools.keys():
		total += pools[key].size()
	return total

## 销毁池中的所有对象
## @param pool_key: 场景资源路径
static func dispose_pool(pool_key: String) -> void:
	if pools.has(pool_key):
		for node in pools[pool_key]:
			if node != null and not node.is_queued_for_deletion():
				node.queue_free()
		pools[pool_key].clear()
		pools.erase(pool_key)

## 销毁所有池
static func dispose_all_pools() -> void:
	for key in pools.keys():
		dispose_pool(key)
	pools.clear()  ## 销毁所有对象池并释放其中的所有对象，用于游戏退出或场景切换时的资源清理
