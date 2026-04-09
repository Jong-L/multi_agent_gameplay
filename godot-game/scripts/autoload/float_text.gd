extends Node

@onready var damage_font=preload("res://resources/damage_font.tres")

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	pass

func show_damage_text(damage:String,position:Vector2,color:Color):
	var label=Label.new()
	label.text=damage
	label.z_index=12
	label.scale=Vector2(0.15,0.15)
	var x_random=randf_range(-10,10)
	var y_random=randf_range(10,15)
	var vector_random=Vector2(x_random,y_random)
	
	label.label_settings=LabelSettings.new()
	label.label_settings.font=damage_font
	label.label_settings.font_color=color
	label.label_settings.font_size=100
	label.label_settings.outline_size=1
	label.label_settings.outline_color="#000"
	
	add_child(label)
	
	var offset=label.size/2
	label.position=position-offset-vector_random
	label.pivot_offset=label.size/2
	
	var tween=create_tween()
	tween.tween_property(label,"position:x",label.position.x+randf_range(-5,5),0.5).set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label,"position:y",label.position.y-12,0.5).set_trans(Tween.TRANS_EXPO).set_ease(Tween.EASE_OUT)
	tween.parallel()
	tween.tween_property(label,"scale",Vector2(0.07,0.07),0.5).set_ease(Tween.EASE_IN)
	
	await tween.finished
	label.queue_free()
	
	
	
	
	
	
	
