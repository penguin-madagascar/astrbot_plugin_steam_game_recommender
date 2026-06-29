# astrbot_plugin_game_recommender

无需 API Key 即可运行的游戏推荐插件。Steam/PC 请求默认使用本地 Steam 索引、标签相似度和评测口碑进行推荐；填写 RAWG API Key 后会增强 PlayStation、Xbox、Nintendo Switch 等多平台候选召回和 RAWG 游戏资料。安装 `astrbot_plugin_steam_price_heybox` 时，会额外补充 Steam 当前价、历史最低价、促销状态和小黑盒跨区价格摘要；未安装时仍保持推荐能力，不让 LLM 凭记忆编造事实。

## 功能

- `/gamerec <自然语言需求>`：根据平台、类型、排除项、人数、预算、语言、难度、氛围等偏好推荐游戏；兼容 alias：`/游戏推荐`。
- `/gamedesc <游戏名>`：查询游戏基础资料，并在可用时补充 Steam 价格；兼容 alias：`/游戏详情`。
- 平台覆盖：默认优先支持 Steam/PC 索引型推荐；配置 `rawg_api_key` 后支持 PlayStation、Xbox、Nintendo Switch 的候选召回与筛选。
- Steam/PC 推荐参考 SteamPeek 思路，优先按归一化标签相似度、排除标签、评测数量和好评率排序，而不是只看评分。
- 使用 SQLite 缓存 Steam/RAWG 响应，减少重复请求。
- 价格查询全部通过 `astrbot_plugin_steam_price_heybox`；本插件不直接接入第三方价格 API。

## 安装

1. 将本目录放入 AstrBot 插件目录，或通过 AstrBot 插件管理安装。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 可直接使用；如需增强多平台推荐，可在 AstrBot WebUI 插件配置中填写 `rawg_api_key`。

## 配置

可选：

- `rawg_api_key`：RAWG API Key。留空时使用 Steam 公开数据源；填写后增强多平台候选召回和 RAWG 游戏资料。
- `llm_provider_id`：用于偏好解析和结果解释；留空时尝试当前会话模型，失败会自动降级。
- `max_results`：默认推荐数量，范围 1-10。
- `steam_index_ttl_hours`：Steam 推荐索引缓存有效期，默认 168 小时。
- `steam_min_review_count`：Steam 索引推荐最低评测数量，默认 50。
- `steam_min_positive_ratio`：Steam 索引推荐最低好评率，默认 0.65。
- `cache_ttl_hours`：Steam/RAWG 缓存有效期。
- `default_region`：默认地区代码，用于 Steam 公开数据源和 Steam 价格增强。
- `timeout_seconds`：HTTP 请求超时时间。
- `steam_price_heybox_notice`：dashboard 静态提示字段，无需填写；安装 `astrbot_plugin_steam_price_heybox` 后自动启用价格增强。

## 示例

```text
/gamerec 推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，最好支持中文，预算 100 以内，类似双人成行但别太难。
/gamedesc It Takes Two
```

## 限制说明

- 价格增强依赖可导入的 `astrbot_plugin_steam_price_heybox`；该插件未安装、不可导入或查询失败时，只展示游戏推荐结果。
- Steam 索引推荐仅覆盖 Steam/PC；Nintendo Switch、PlayStation、Xbox 等多平台请求会继续使用 RAWG 流程，未配置 `rawg_api_key` 时覆盖有限，结果中会提示。
- 预算会参与软排序：当前价在预算内会加分，超预算会提示，但不会直接过滤候选。
- Steam/RAWG 的中文支持数据都可能不完整；结果中未确认时会显示“不确定”或提醒以商店页面为准。
- 多人/合作、难度、氛围主要依据 Steam categories/genres、RAWG 标签和规则推断，可能不完整。
- PlayStation、Xbox、Nintendo Switch 的深度价格追踪留待后续接入官方/合规 API。

## 开发验证

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r ../astrbot_plugin_steam_price_heybox/requirements.txt pyyaml
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m compileall -q .
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m unittest discover tests
```
