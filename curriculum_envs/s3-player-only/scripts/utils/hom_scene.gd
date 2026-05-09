extends Node

## 主菜单场景控制器
## 处理主菜单的按钮交互：开始游戏、退出游戏



## 开始游戏按钮回调
func _on_play_button_pressed() -> void:
	get_tree().change_scene_to_file("res://assets/scenes/PlayScene.tscn")

## 退出游戏按钮回调
func _on_exit_button_pressed() -> void:
	get_tree().quit()

## 设置按钮回调（预留）
func _on_settings_button_pressed() -> void:
	pass  ## 打开设置菜单

## 关于按钮回调（预留）
func _on_about_button_pressed() -> void:
	pass  ## 显示关于信息
