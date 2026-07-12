# astrbot_plugin_game_recommender

无需 API Key 即可运行的 Steam/PC 游戏推荐插件。v0.5.0 使用“偏好与约束解析 → 查询感知召回 → 标签混合精排 → 可选 Embedding 重排 → 过滤与 MMR 最终选择 → 会话反馈”的两阶段流程。安装 `astrbot_plugin_steam_price_heybox` 时，会额外补充 Steam 当前价、历史最低价、促销状态和小黑盒跨区价格摘要。LLM 默认只负责偏好解析；用户可见推荐理由只引用 Steam 数据和结构化证据。仅在启用 LLM 兜底且无结果时，才会生成明确标注的未验证建议。

## 功能

- `/gamerec <自然语言需求>`：根据平台、类型、排除项、人数、预算、语言、难度、氛围等偏好推荐游戏；兼容 alias：`/游戏推荐`。
- `/gamerec_retry [补充要求]`：基于最近一次 `/gamerec` 换一批推荐，并排除已展示结果；兼容 alias：`/重新推荐`、`/换一批`。
- `/accountbind [steam] <SteamID64|好友码>`：绑定当前聊天用户的 Steam 账号；兼容 alias：`/账号绑定`。
- `/unplayedrec`：从已绑定 Steam 游戏库中随机推荐一款未游玩且 Steam 评价过线的游戏；兼容 alias：`/未玩推荐`。
- `/gamedesc <游戏名>`：查询游戏基础资料，并在可用时补充 Steam 价格；兼容 alias：`/游戏详情`。
- 平台覆盖：当前版本仅支持 Steam/PC，检测到 Switch、PlayStation、Xbox 等平台会明确提示；启用 LLM 兜底后，超出范围或无结果时可返回已标注的未验证建议。
- `/gamerec` 和 `/游戏推荐` 支持前置库过滤参数：`排除已有` / `exclude-owned` 会排除 Steam 游戏库中已有的候选；`仅查看已有` / `only-owned` 只保留库内已有的候选。
- 查询感知召回每轮最多使用 8 个搜索词，并在参考游戏可靠解析后继续按其 Steam 有序标签补充候选；索引最多保留 3000 个候选和 256 个搜索词覆盖记录。
- 混合精排结合标签覆盖、正负参考、Steam 游戏库画像和贝叶斯平滑口碑；硬条件确认违反时过滤，证据未知时保留并提示风险。
- 最终列表按层级执行 MMR；严格、平衡和高多样模式的冗余惩罚分别为 0、0.15 和 0.30。
- 使用 SQLite 缓存 Steam 响应和推荐索引，减少重复请求。
- 使用 SQLite 保存最近 30 分钟的推荐上下文、上一批有序结果摘要和最多 10 条结构化反馈。
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
- `enable_llm_fallback`：启用后，空结果或超出 Steam/PC 覆盖范围时允许 LLM 生成已标注的未验证建议；默认关闭。
- `enable_embedding_rerank`：默认关闭；开启后只对约束合格的全局 Top-20 在原层级内按 75% 基础相关度 + 25% 语义相似度重排。
- `embedding_provider_id`：可选的 AstrBot Embedding Provider；留空时使用首个可用 provider。provider 不可用、10 秒超时或向量非法时自动保留原排序。
- `steam_api_key`：Steam Web API Key，用于 `/gamerec` 的已有游戏过滤参数和 `/unplayedrec` 调用 `GetOwnedGames`；留空时账号绑定仍可用，但无法读取游戏库。
- `max_results`：默认推荐数量，范围 1-10。
- `steam_index_ttl_hours`：Steam 推荐索引缓存有效期，默认 168 小时。
- `steam_min_review_count`：索引口碑分的贝叶斯先验强度（最低按 50），同时作为 `/unplayedrec` 的最低评测数量。
- `steam_min_positive_ratio`：仅作为 `/unplayedrec` 的最低好评率，默认 0.65。
- `cache_ttl_hours`：Steam 缓存有效期。
- `default_region`：默认地区代码，用于 Steam 公开数据源和 Steam 价格增强。
- `timeout_seconds`：HTTP 请求超时时间。
- `steam_price_heybox_notice`：dashboard 静态提示字段，无需填写；安装 `astrbot_plugin_steam_price_heybox` 后自动启用价格增强。

## 示例

```text
/gamerec 推荐几个 Steam 上适合双人的轻松解谜合作游戏，不要恐怖，最好支持中文，预算 100 以内，类似双人成行但别太难。
/accountbind steam 76561198000000000
/unplayedrec
/未玩推荐
/gamerec 排除已有 推荐几个 Steam 合作解谜游戏
/游戏推荐 仅查看已有 找几个适合双人的轻松游戏
/gamerec 更多样 推荐几个 Steam 合作游戏，别太同质
/重新推荐
/换一批 不要恐怖，预算 100 以内
/重新推荐 喜欢第 2 款这类
/换一批 不喜欢第 1 款这类，换不同玩法
/换一批 不要第 3 款
/gamedesc It Takes Two
```

## 限制说明

- 价格增强依赖可导入的 `astrbot_plugin_steam_price_heybox`；该插件未安装、不可导入或查询失败时，只展示游戏推荐结果。
- `排除已有` / `仅查看已有` / `/unplayedrec` 依赖 `steam_api_key` 和公开可读的 Steam 游戏库；未绑定、未配置 key、资料隐私不可见或接口失败时，会直接提示错误。
- `/unplayedrec` 将 Steam `playtime_forever` 为 0 的条目视为未游玩，并沿用 `steam_min_review_count` 与 `steam_min_positive_ratio` 作为评价门槛。
- Steam 索引推荐仅覆盖 Steam/PC；Nintendo Switch、PlayStation、Xbox 等跨平台请求会返回范围提示。
- `/重新推荐` / `/换一批` 只复用最近 30 分钟内当前聊天用户的 `/gamerec` 结果，不会复用 `/unplayedrec` 或 `/gamedesc`。
- “喜欢第 N 款这类”会增加正参考，“不喜欢第 N 款这类”会增加负参考；仅说“不要第 N 款”只排除该游戏，不会泛化它的标签。越界序号会被忽略并提示。
- 多样性模式由 LLM 从用户描述中解析；LLM 不可用、字段缺失或返回非法值时固定使用严格匹配。
- 每轮只读取一次 Steam 游戏库，同时用于库过滤和最多占总分 10% 的已游玩画像；未游玩条目不会被视为负反馈。
- 预算只参与软排序：当前价在预算内最多加 0.05，超预算最多减 0.05，未知价格减 0.02，不会因价格单独删除候选；无预算时价格查询不会改变最终顺序。
- Steam 的中文支持数据可能不完整；结果中未确认时会显示“不确定”或提醒以商店页面为准。
- 多人/合作、难度、氛围主要依据 Steam categories/genres、描述和规则推断，可能不完整。

## 开发验证

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r ../astrbot_plugin_steam_price_heybox/requirements.txt pyyaml
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m compileall -q .
PYTHONPATH=/Users/jiangxingda/Projects/QQChatbot .venv/bin/python -m unittest discover tests
uvx ruff check .
uvx ruff format --check .
```
