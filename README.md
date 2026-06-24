# astrbot_plugin_game_recommender

基于 RAWG 数据源、规则过滤/排序和 AstrBot LLM 偏好解析的游戏推荐插件。MVP 目标是先跑通“自然语言需求 -> 结构化偏好 -> RAWG 查询/缓存 -> 可解释推荐”，不追踪实时价格，也不让 LLM 凭记忆编造事实。

## 功能

- `/游戏推荐 <自然语言需求>`：根据平台、类型、排除项、人数、预算、语言、难度、氛围等偏好推荐游戏。
- `/游戏详情 <游戏名>`：查询 RAWG 中的游戏基础资料。
- 使用 SQLite 缓存 RAWG 响应，减少重复请求。
- IGDB、Steam、IsThereAnyDeal client 已预留骨架，首版不调用。

## 安装

1. 将本目录放入 AstrBot 插件目录，或通过 AstrBot 插件管理安装。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 在 AstrBot WebUI 插件配置中填写 `rawg_api_key`。

## 配置

必填：

- `rawg_api_key`：RAWG API Key。未配置时插件会返回清晰错误提示。

可选：

- `llm_provider_id`：用于偏好解析和结果解释；留空时尝试当前会话模型，失败会自动降级。
- `max_results`：默认推荐数量，范围 1-10。
- `cache_ttl_hours`：RAWG 缓存有效期。
- `default_region`：MVP 仅用于文案提示。
- `igdb_client_id`、`igdb_client_secret`、`itad_api_key`：预留字段，MVP 不调用。

## 示例

```text
/游戏推荐 推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，最好支持中文，预算 100 以内，类似双人成行但别太难。
/游戏详情 It Takes Two
```

## 限制说明

- RAWG 不提供可靠实时地区价格，本插件不会编造当前价、史低或折扣。
- RAWG 的中文支持数据不稳定；结果中未确认时会显示“不确定”或提醒以商店页面为准。
- 多人/合作、难度、氛围主要依据 RAWG 标签和规则推断，可能不完整。
- Steam、PlayStation、Nintendo Switch 的深度价格追踪留待后续接入官方/合规 API。

## 开发验证

```bash
python3.12 -m compileall astrbot_plugin_game_recommender
python3.12 -m unittest discover astrbot_plugin_game_recommender/tests
```

