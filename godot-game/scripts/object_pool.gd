class_name ObjectPool
extends Node

static var pools: Dictionary = {} 
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


# 这是一个通用的工具类，不继承 Node，纯数据处理
static func spawn(pool_key: String, parent_node: Node, initial_position: Vector2 = Vector2.ZERO) -> Node:
	# 1. 检查该类型的池是否存在
	if not pools.has(pool_key):
		_init_pool(pool_key) # 首次调用时初始化

	var pool = pools[pool_key]
	var node: Node

	if pool.size() > 0:
		node = pool.pop_front()
		# 激活节点
		if node.is_inside_tree():
			node.show()
			node.disabled = false
		else:
			parent_node.add_child(node)
			node.global_position = initial_position
	else:
		# 池空了，扩容（或者返回 null）
		node = ResourceLoader.load(pool_key).instantiate()
		parent_node.add_child(node)
		node.global_position = initial_position

	return node

static func return_to_pool(node: Node, pool_key: String):
	# 1. 隐藏/禁用节点
	node.hide()
	node.disabled = true
	if node.is_inside_tree():
		node.remove_from_parent()
	# 2. 放回池中
	if not pools.has(pool_key):
		pools[pool_key] = []
	pools[pool_key].append(node)


static func _init_pool(pool_key:String):
	pools[pool_key]=[]
