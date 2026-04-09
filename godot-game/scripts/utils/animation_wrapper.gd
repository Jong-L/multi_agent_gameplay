class_name AnimationWrapper
extends RefCounted


var name=""
var is_high_priority=false

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func _init(_name:String,_is_high_priority:bool=false):
	self.name=_name
	self.is_high_priority=_is_high_priority
