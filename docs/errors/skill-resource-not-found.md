# SKILL_RESOURCE_NOT_FOUND — ResourceNotFoundError

`read_resource` 请求的资源路径不存在。

处理：先调 `list_resources` 确认相对路径，再读取；路径须相对技能根目录。
