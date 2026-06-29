# astrbot_plugin_game_recommender

无需 API Key 即可运行的 Steam/PC 游戏推荐插件。推荐流程使用本地 Steam 索引、归一化标签相似度、排除标签和 Steam 评测口碑；安装 `astrbot_plugin_steam_price_heybox` 时，会额外补充 Steam 当前价、历史最低价、促销状态和小黑盒跨区价格摘要。LLM 只负责抽取用户需求里的额外标签、排除项和相似游戏名，不直接编造推荐事实。

## 功能

- `/gamerec <自然语言需求>`：根据平台、类型、排除项、人数、预算、语言、难度、氛围等偏好推荐游戏；兼容 alias：`/游戏推荐`。
- `/accountbind [steam] <SteamID64|好友码>`：绑定当前聊天用户的 Steam 账号；兼容 alias：`/账号绑定`。
- `/gamedesc <游戏名>`：查询游戏基础资料，并在可用时补充 Steam 价格；兼容 alias：`/游戏详情`。
- 平台覆盖：当前版本仅支持 Steam/PC，检测到 Switch、PlayStation、Xbox 等平台会明确提示。
- `/gamerec` 文本中明确写“排除已玩/过滤已玩/不要已玩/别推荐已玩/排除玩过/过滤玩过”等短语时，会尝试读取已绑定 Steam 游戏库，排除已有且 `playtime_forever > 0` 的候选。
- 推荐参考 SteamPeek 思路，优先按归一化标签相似度、排除标签、评测数量和好评率排序，而不是只看评分。
- 使用 SQLite 缓存 Steam 响应和推荐索引，减少重复请求。
- 价格查询全部通过 `astrbot_plugin_steam_price_heybox`；本插件不直接接入第三方价格 API。

## 安装

1. 将本目录放入 AstrBot 插件目录，或通过 AstrBot 插件管理安装。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 可直接使用；无需填写游戏数据 API Key。

## 配置

可选：

- `llm_provider_id`：用于偏好解析和结果解释；留空时尝试当前会话模型，失败会自动降级。
- `steam_api_key`：Steam Web API Key，用于 `/gamerec` 显式排除已玩时调用 `GetOwnedGames`；留空时账号绑定仍可用，但会跳过已玩排除。
- `max_results`：默认推荐数量，范围 1-10。
- `steam_index_ttl_hours`：Steam 推荐索引缓存有效期，默认 168 小时。
- `steam_min_review_count`：Steam 索引推荐最低评测数量，默认 50。
- `steam_min_positive_ratio`：Steam 索引推荐最低好评率，默认 0.65。
- `cache_ttl_hours`：Steam 缓存有效期。
- `default_region`：默认地区代码，用于 Steam 公开数据源和 Steam 价格增强。
- `timeout_seconds`：HTTP 请求超时时间。
- `steam_price_heybox_notice`：dashboard 静态提示字段，无需填写；安装 `astrbot_plugin_steam_price_heybox` 后自动启用价格增强。

## 示例

```text
/gamerec 推荐几个 Steam 上适合双人的轻松解谜合作游戏，不要恐怖，最好支持中文，预算 100 以内，类似双人成行但别太难。
/accountbind steam 76561198000000000
/gamerec 推荐几个 Steam 合作解谜游戏，排除已玩
/gamedesc It Takes Two
```

## 限制说明

- 价格增强依赖可导入的 `astrbot_plugin_steam_price_heybox`；该插件未安装、不可导入或查询失败时，只展示游戏推荐结果。
- 已玩排除依赖 `steam_api_key` 和公开可读的 Steam 游戏库；未绑定、未配置 key、资料隐私不可见或接口失败时，会跳过排除并继续返回原推荐。
- Steam 索引推荐仅覆盖 Steam/PC；Nintendo Switch、PlayStation、Xbox 等跨平台请求会返回范围提示。
- 预算会参与软排序：当前价在预算内会加分，超预算会提示，但不会直接过滤候选。
- Steam 的中文支持数据可能不完整；结果中未确认时会显示“不确定”或提醒以商店页面为准。
- 多人/合作、难度、氛围主要依据 Steam categories/genres、描述和规则推断，可能不完整。

## 开发验证

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r ../astrbot_plugin_steam_price_heybox/requirements.txt pyyaml
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m compileall -q .
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m unittest discover tests
```
